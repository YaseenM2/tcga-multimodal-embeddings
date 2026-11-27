#!/usr/bin/env python3
"""Pack per-file .npy embeddings into one HDF5 file in the authors' layout.

Input (the per-file embedding layout our scripts use; the PyHealth input
folder is what ``export_pyhealth.py`` writes)::

    <root>/embeddings/<modality>/<TCGA-case-id>/<file_id>.npy

Output (what the authors' notebooks and ``convert_h5_embeddings.py`` read;
e.g. ``summ.h5``, ``expr.h5``)::

    <TCGA-case-id>/<file_id>  -> 1-D float32 dataset

One HDF5 per modality is far easier to commit and share than thousands of
``.npy`` files. ``convert_h5_embeddings.py`` turns it back into the ``.npy``
layout; the round trip is lossless (see tests).

Histology slide vectors are saved by ``aggregate_uni2_histology.py`` as
``<slide>.h5.npy``; the authors key slides without the ``.h5``, so that
suffix is dropped for the histology modality.

Example::

    python scripts/pack_h5.py --root /content/tcga10 --modality report \\
        --splits /content/tcga10/split_cases.csv --output-h5 summ.h5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DIMS = {"report": 4096, "histology": 1536, "expression": 256}
CASE_RE = re.compile(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}\Z")


def file_key(npy: Path, modality: str) -> str:
    name = npy.name[: -len(".npy")]
    if modality == "histology" and name.endswith(".h5"):
        name = name[: -len(".h5")]
    if not name or "/" in name:
        raise ValueError(f"Bad file name {npy.name}")
    return name


def collect(root: Path, modality: str, case_ids: set[str] | None, dim: int) -> dict[str, dict[str, np.ndarray]]:
    folder = Path(root) / "embeddings" / modality
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    data: dict[str, dict[str, np.ndarray]] = {}
    for npy in sorted(folder.glob("*/*.npy")):
        case_id = npy.parent.name
        if not CASE_RE.fullmatch(case_id):
            raise ValueError(f"Not a TCGA case folder: {npy.parent}")
        if case_ids is not None and case_id not in case_ids:
            continue
        vector = np.load(npy, allow_pickle=False)
        if vector.shape != (dim,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
            raise ValueError(f"{npy}: expected finite float32 ({dim},), got {vector.dtype} {vector.shape}")
        key = file_key(npy, modality)
        if key in data.setdefault(case_id, {}):
            raise ValueError(f"{case_id}: duplicate key {key}")
        data[case_id][key] = vector
    return data


def write_h5(data: dict[str, dict[str, np.ndarray]], output: Path, attrs: dict) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output.parent, prefix=".part-", suffix=".h5")
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as h5:
            for key, value in attrs.items():
                h5.attrs[key] = value
            for case_id in sorted(data):
                group = h5.create_group(case_id)
                for key in sorted(data[case_id]):
                    group.create_dataset(key, data=data[case_id][key])
        os.replace(tmp, output)
        os.chmod(output, 0o644)  # mkstemp creates 0600
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def verify(data: dict[str, dict[str, np.ndarray]], output: Path) -> bool:
    with h5py.File(output, "r") as h5:
        if set(h5.keys()) != set(data):
            return False
        for case_id, files in data.items():
            if set(h5[case_id].keys()) != set(files):
                return False
            for key, vector in files.items():
                stored = h5[case_id][key][()]
                if stored.dtype != np.float32 or not np.array_equal(stored, vector):
                    return False
    return True


def run(root: Path, modality: str, output: Path, *, splits: Path | None = None,
        expected_dim: int | None = None, note: str = "") -> dict:
    dim = expected_dim or DIMS[modality]
    case_ids = None
    if splits is not None:
        case_ids = set(pd.read_csv(splits, dtype=str)["case_id"])
    data = collect(root, modality, case_ids, dim)
    if not data:
        raise ValueError("No vectors matched; check --root, --modality and --splits")
    n_vectors = sum(len(v) for v in data.values())
    attrs = {"modality": modality, "embedding_dim": dim, "n_cases": len(data),
             "n_vectors": n_vectors, "layout": "<case_id>/<file_id> -> float32 vector",
             "note": note}
    write_h5(data, output, attrs)
    if not verify(data, output):
        raise OSError(f"Verification of {output} failed")
    result = {"modality": modality, "cases": len(data), "vectors": n_vectors,
              "size_mib": round(Path(output).stat().st_size / 2**20, 2)}
    if case_ids is not None:
        result["cohort_cases"] = len(case_ids)
        result["cohort_cases_missing"] = len(case_ids - set(data))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="folder holding embeddings/<modality>/")
    p.add_argument("--modality", choices=sorted(DIMS), required=True)
    p.add_argument("--output-h5", type=Path, required=True)
    p.add_argument("--splits", type=Path, help="keep only cases in this split_cases.csv")
    p.add_argument("--expected-dim", type=int)
    p.add_argument("--note", default="", help="free-text provenance stored as an HDF5 attribute")
    args = p.parse_args(argv)
    result = run(args.root, args.modality, args.output_h5, splits=args.splits,
                 expected_dim=args.expected_dim, note=args.note)
    print(json.dumps(result, indent=2))
    print("HDF5 PACKED AND VERIFIED!")


if __name__ == "__main__":
    main()
