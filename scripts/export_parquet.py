#!/usr/bin/env python3
"""Export per-file embeddings to the team's patient-level Parquet contract.

Writes ``<modality>.parquet`` with exactly one row per ``case_id`` listed in
``--cases`` (the team's ``data/clinical.csv``):

=============  ===============  ==========================================
column         Arrow type       meaning
=============  ===============  ==========================================
``case_id``    ``string``       TCGA case id, unique
``embedding``  ``list<float>``  float32 mean over that patient's files
``n_sources``  ``int32``        number of files (reports/slides/samples)
=============  ===============  ==========================================

Patient vectors are the float32 mean over the patient's file vectors, exactly
as the authors' modelling notebook does (``np.stack(...).mean(axis=0)``). No
standardisation or PCA: those are fitted per fold in PyHealth.

Contract modality names map to our folders as follows::

    text -> embeddings/report/      (BioMistral, 4096)
    hist -> embeddings/histology/   (UNI2-h, 1536)
    expr -> embeddings/expression/  (BulkRNABert, 256)

Inputs, one of:

* ``--npy-root ROOT``: ``ROOT/embeddings/<folder>/<case_id>/<file>.npy``
  (what embed_reports_biomistral.py and aggregate_uni2_histology.py write);
* ``--h5 FILE``: the authors' layout ``<case_id>/<file_id> -> vector``
  (what embed_expr_bulkrnabert.py writes).

By default every case in ``--cases`` must have at least one vector, because
the contract's validator requires all of them. ``--allow-missing`` writes the
covered cases only and lists how many are missing.

Example::

    python scripts/export_parquet.py --modality expr --h5 /content/expr.h5 \\
        --cases DL4H-project/data/clinical.csv \\
        --output DL4H-project/data/embeddings/expr.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DIMS = {"text": 4096, "hist": 1536, "expr": 256}
FOLDERS = {"text": "report", "hist": "histology", "expr": "expression"}
CASE_RE = re.compile(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}\Z")


def read_cases(path: Path) -> list[str]:
    ids = pd.read_csv(path, dtype=str)["case_id"].tolist()
    if not ids or len(ids) != len(set(ids)) or not all(CASE_RE.fullmatch(i) for i in ids):
        raise ValueError(f"{path}: empty, duplicate or malformed case_id values")
    return ids


def vectors_from_npy(root: Path, modality: str) -> dict[str, dict[str, np.ndarray]]:
    folder = Path(root) / "embeddings" / FOLDERS[modality]
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    out: dict[str, dict[str, np.ndarray]] = {}
    for npy in sorted(folder.glob("*/*.npy")):
        out.setdefault(npy.parent.name, {})[npy.name[: -len(".npy")]] = np.load(npy, allow_pickle=False)
    return out


def vectors_from_h5(path: Path) -> dict[str, dict[str, np.ndarray]]:
    import h5py

    out: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as h5:
        for case_id in h5.keys():
            out[case_id] = {key: h5[case_id][key][()] for key in h5[case_id].keys()}
    return out


def patient_table(per_file: dict[str, dict[str, np.ndarray]], cases: list[str], dim: int,
                  *, allow_missing: bool = False) -> tuple[pd.DataFrame, dict]:
    rows, missing = [], []
    for case_id in cases:  # keeps the order of clinical.csv
        files = per_file.get(case_id) or {}
        if not files:
            missing.append(case_id)
            continue
        stack = []
        for key in sorted(files):
            vec = np.asarray(files[key])
            if vec.shape != (dim,) or vec.dtype != np.float32 or not np.isfinite(vec).all():
                raise ValueError(f"{case_id}/{key}: need finite float32 ({dim},), got {vec.dtype} {vec.shape}")
            stack.append(vec)
        mean = np.stack(stack).mean(axis=0).astype(np.float32)
        rows.append({"case_id": case_id, "embedding": mean, "n_sources": len(stack)})
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} of {len(cases)} cases have no vectors "
                         "(the contract needs all; rerun the embedding step or use --allow-missing)")
    outside = sorted(set(per_file) - set(cases))
    stats = {"cases_expected": len(cases), "cases_written": len(rows), "cases_missing": len(missing),
             "files_used": int(sum(r["n_sources"] for r in rows)),
             "cases_with_several_files": int(sum(r["n_sources"] > 1 for r in rows)),
             "source_cases_outside_list": len(outside)}
    return pd.DataFrame(rows), stats


def write_parquet(table: pd.DataFrame, output: Path, dim: int, metadata: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [("case_id", pa.string()), ("embedding", pa.list_(pa.float32())), ("n_sources", pa.int32())],
        metadata={b"contract": json.dumps(metadata, sort_keys=True).encode()},
    )
    flat = np.concatenate(table["embedding"].to_list()) if len(table) else np.zeros(0, np.float32)
    offsets = np.arange(0, len(table) * dim + 1, dim, dtype=np.int32)
    arrow = pa.Table.from_arrays(
        [pa.array(table["case_id"].tolist(), pa.string()),
         pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat, pa.float32())),
         pa.array(table["n_sources"].to_numpy(np.int32), pa.int32())],
        schema=schema,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output.parent, prefix=".part-", suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(arrow, tmp)
        back = pq.read_table(tmp)
        got = np.asarray(back.column("embedding").combine_chunks().flatten())
        if back.num_rows != len(table) or not np.array_equal(got, flat):
            raise OSError("Parquet read-back verification failed")
        os.replace(tmp, output)
        os.chmod(output, 0o644)  # mkstemp creates 0600
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def check_text_sources(per_file: dict[str, dict[str, np.ndarray]], reports_csv: Path) -> None:
    """Each text vector must come from the report named in reports.csv."""
    reports = pd.read_csv(reports_csv, dtype=str)
    expected = dict(zip(reports["case_id"], reports["patient_filename"]))
    for case_id, files in per_file.items():
        if case_id in expected and set(files) != {expected[case_id]}:
            raise ValueError(f"{case_id}: text vectors {sorted(files)} != reports.csv "
                             f"patient_filename {expected[case_id]}")


def run(modality: str, cases_csv: Path, output: Path, *, npy_root: Path | None = None,
        h5: Path | None = None, reports_csv: Path | None = None,
        allow_missing: bool = False, note: str = "") -> dict:
    if (npy_root is None) == (h5 is None):
        raise ValueError("Give exactly one of npy_root or h5")
    dim = DIMS[modality]
    cases = read_cases(cases_csv)
    per_file = vectors_from_npy(npy_root, modality) if npy_root is not None else vectors_from_h5(h5)
    if modality == "text" and reports_csv is not None:
        check_text_sources(per_file, reports_csv)
    table, stats = patient_table(per_file, cases, dim, allow_missing=allow_missing)
    write_parquet(table, output, dim, {"modality": modality, "dim": dim,
                                       "aggregation": "float32 mean over files", "note": note})
    stats["size_mib"] = round(Path(output).stat().st_size / 2**20, 2)
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=sorted(DIMS), required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--npy-root", type=Path, help="folder holding embeddings/<report|histology|expression>/")
    src.add_argument("--h5", type=Path, help="authors'-layout HDF5 (<case_id>/<file_id>)")
    p.add_argument("--cases", type=Path, required=True, help="team's data/clinical.csv")
    p.add_argument("--output", type=Path, required=True, help="e.g. data/embeddings/expr.parquet")
    p.add_argument("--reports", type=Path, help="team's data/reports.csv (text: check source files)")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--note", default="", help="provenance stored in the Parquet schema metadata")
    args = p.parse_args(argv)
    stats = run(args.modality, args.cases, args.output, npy_root=args.npy_root, h5=args.h5,
                reports_csv=args.reports, allow_missing=args.allow_missing, note=args.note)
    print(json.dumps(stats, indent=2))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
