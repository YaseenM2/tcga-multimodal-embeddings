#!/usr/bin/env python3
"""Write the input folder that PyHealth's TCGAMultimodalDataset loads.

Output layout (``--output``)::

    clinical.csv                                  clinical rows of the study cases
    split_cases.csv                               case_id, split
    embeddings/{expr,hist,text}/<case_id>/<file_id>.npy

The study cases are the ``case_id`` values in ``--splits``. Each case gets one
float32 vector per modality, read from the patient-level Parquet files that
``export_parquet.py`` writes (``<parquet-dir>/{expr,hist,text}.parquet``).
Histology is therefore the per-case mean over slides, not one file per slide.
The ``file_id`` is the report's ``patient_filename`` for ``text`` and the
``case_id`` for ``expr`` and ``hist``.

Inputs:

* ``--clinical``: the authors' ``data/clinical.csv`` or the ``clinical.csv``
  that ``select_cohort.py`` writes. It needs the loader's columns
  (``CLINICAL_COLUMNS``). Rows of the study cases are copied as is, in the
  order of ``--splits``.
* ``--splits``: the authors' ``results/split_cases.csv`` or the subset that
  ``select_cohort.py`` writes. Only ``case_id`` and ``split`` (0 to 4) are kept.
* ``--reports``: the authors' ``data/summarized_reports.csv``; it names the
  report file of each case (exactly one per study case).
* ``--parquet-dir``: your own ``export_parquet.py`` outputs. Each file must
  hold every study case exactly once. UNI2-h (``hist``) vectors are never
  shared; generate them with your own Hugging Face access.

The script refuses an existing export; ``--overwrite`` replaces it together
with the loader's ``tcga_multimodal_*-pyhealth.csv`` index files. It refuses an
output folder that is or contains one of the inputs. It writes the export to a
temporary folder next to ``--output`` and moves it in place only when complete,
so a failed write keeps the old export, or leaves no output. If the final
move fails or is interrupted, the temp folder is kept and the error says where
the old files are. A mount point as ``--output`` is refused, because the move
needs one file system; pass a folder inside it.

Example::

    python scripts/export_pyhealth.py --clinical tcga10/clinical.csv \\
        --splits tcga10/split_cases.csv \\
        --reports multimodal-cancer-modeling/data/summarized_reports.csv \\
        --parquet-dir out --output pyhealth_root
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

DIMS = {"expr": 256, "hist": 1536, "text": 4096}
REPORT_MODALITY = "text"
CLINICAL_CSV = "clinical.csv"
SPLIT_CSV = "split_cases.csv"
SPLIT_COLUMN = "split"
# The loader contract (PyHealth pyhealth/datasets/tcga_survival.py): required clinical columns and fold values.
CLINICAL_COLUMNS = ("case_id", "project", "sex", "age", "race", "ethnicity", "vital_status",
                    "days_to_death", "days_to_last_follow_up")
SPLIT_VALUES = tuple(str(fold) for fold in range(5))
EMBEDDINGS_DIR = "embeddings"
LOADER_CACHE_GLOB = "tcga_multimodal_*-pyhealth.csv"
NEW_EXPORT = "new"
OLD_EXPORT = "old"
EXIT_INTERRUPTED = 130  # 128 + SIGINT, what the shell reports for Ctrl-C


class ExportError(ValueError):
    """Raised when an input or the output folder is not valid."""


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file into its header and one dict per row."""
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def require_columns(path: Path, header: list[str], needed: set[str]) -> None:
    """Fail if the CSV header lacks one of the needed columns."""
    missing = sorted(needed - set(header))
    if missing:
        raise ExportError(f"{path} lacks columns {missing}")


def check_file_id(file_id: str) -> str:
    """Return the name unchanged if it is safe as a file or folder name."""
    if not file_id or "/" in file_id or "\\" in file_id or file_id.startswith("."):
        raise ExportError(f"Unsafe file_id {file_id!r}")
    return file_id


def read_splits(path: Path) -> dict[str, str]:
    """Map each study case_id to its split."""
    header, rows = read_csv(path)
    require_columns(path, header, {"case_id", SPLIT_COLUMN})
    splits = {row["case_id"]: row[SPLIT_COLUMN] for row in rows}
    if not rows or len(splits) != len(rows):
        raise ExportError(f"{path}: empty or repeats case_id values")
    invalid = [(case_id, value) for case_id, value in splits.items() if value not in SPLIT_VALUES]
    if invalid:
        case_id, value = invalid[0]
        raise ExportError(f"{path}: case {case_id} has {SPLIT_COLUMN} {value!r}; every {SPLIT_COLUMN} must be "
                          f"a plain integer from 0 to {len(SPLIT_VALUES) - 1} (no '1.0', no spaces)")
    return splits


def read_clinical(path: Path, cases: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Return the header and the rows of the study cases, in the order of ``cases``."""
    header, rows = read_csv(path)
    require_columns(path, header, set(CLINICAL_COLUMNS))
    by_id = {row["case_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ExportError(f"{path} repeats case_id values")
    missing = [case_id for case_id in cases if case_id not in by_id]
    if missing:
        raise ExportError(f"{path} lacks {len(missing)} study cases, e.g. {missing[0]}")
    return header, [by_id[case_id] for case_id in cases]


def report_file_ids(path: Path, cases: set[str]) -> dict[str, str]:
    """Map each study case_id to its report's patient_filename."""
    header, rows = read_csv(path)
    require_columns(path, header, {"patient_filename"})
    file_ids: dict[str, str] = {}
    for row in rows:
        filename = row["patient_filename"]
        case_id = filename.split(".", 1)[0]
        if case_id not in cases:
            continue
        if case_id in file_ids:
            raise ExportError(f"{path}: case {case_id} has more than one report")
        file_ids[case_id] = check_file_id(filename)
    missing = cases - set(file_ids)
    if missing:
        raise ExportError(f"{path} has no report for {len(missing)} study cases, e.g. {min(missing)}")
    return file_ids


def read_modality(path: Path, dim: int, cases: set[str]) -> tuple[list[str], np.ndarray]:
    """Read case IDs and a (cases, dim) float32 matrix from one Parquet file."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["case_id", "embedding"])
    case_ids = table.column("case_id").to_pylist()
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != cases:
        raise ExportError(f"{path} must hold each case_id in --splits exactly once")
    vectors = table.column("embedding").combine_chunks()
    lengths = pc.list_value_length(vectors)
    if lengths.null_count or pc.any(pc.not_equal(lengths, dim)).as_py():
        raise ExportError(f"{path}: every row needs a vector of length {dim}")
    values = vectors.flatten().to_numpy(zero_copy_only=False)
    matrix = values.astype(np.float32, copy=False).reshape(len(case_ids), dim)
    if not np.isfinite(matrix).all():
        raise ExportError(f"{path}: vectors hold null, NaN or infinite values")
    return case_ids, matrix


def check_inputs(paths: list[Path]) -> None:
    """Fail if an input file does not exist."""
    for path in paths:
        if not Path(path).is_file():
            raise ExportError(f"input {path} does not exist or is not a file")


def check_output(output: Path, inputs: list[Path]) -> None:
    """Refuse an output that is a file, lies below a file, or is or contains one of the inputs."""
    existing = next(p for p in (output, *output.parents) if p.exists())
    if not existing.is_dir():
        raise ExportError(f"{existing} is a file, not a folder; choose another --output")
    if existing != output:
        return
    for path in inputs:
        # Compares file identity, not strings: case-insensitive file systems.
        if any(output.samefile(parent) for parent in Path(path).resolve().parents):
            raise ExportError(f"{output} is or contains the input {path}")


def old_export(output: Path, overwrite: bool) -> list[Path]:
    """Return the entries of an old export that ``--overwrite`` may replace."""
    if not output.exists():
        return []
    generated = [output / name for name in (CLINICAL_CSV, SPLIT_CSV, EMBEDDINGS_DIR)]
    generated += sorted(output.glob(LOADER_CACHE_GLOB))
    existing = [p for p in generated if p.exists() or p.is_symlink()]
    if existing and not overwrite:
        raise ExportError(f"{output} already holds an export ({existing[0].name}); "
                          "pass --overwrite to replace it")
    symlinks = [p for p in existing if p.is_symlink()]
    if symlinks:
        raise ExportError(f"{symlinks[0]} is a symlink; remove it by hand")
    embeddings = output / EMBEDDINGS_DIR
    if embeddings.is_dir():
        foreign = sorted(p.name for p in embeddings.iterdir() if p.name not in DIMS)
        if foreign:
            raise ExportError(f"{embeddings} holds {foreign[0]!r}, which this script does not write; "
                              "choose another --output")
    # The loader never rebuilds its index files, so they go with the files they index.
    return existing


def remove(path: Path) -> None:
    """Remove a file or a folder tree."""
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as error:
        raise ExportError(f"Cannot remove {path}: {error}") from error


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write a CSV file with the csv module's default dialect (CRLF line ends)."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_modality(output: Path, modality: str, case_ids: list[str], matrix: np.ndarray,
                   file_ids: dict[str, str]) -> None:
    """Write one .npy file per case; a case missing from file_ids uses its case_id."""
    folder = output / EMBEDDINGS_DIR / modality
    for case_id, vector in zip(case_ids, matrix, strict=True):
        case_dir = folder / check_file_id(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / f"{check_file_id(file_ids.get(case_id, case_id))}.npy", vector)


def write_export(folder: Path, header: list[str], clinical: list[dict[str, str]], splits: dict[str, str],
                 reports: dict[str, str], vectors: dict[str, tuple[list[str], np.ndarray]]) -> None:
    """Write all export files into a new folder."""
    folder.mkdir()
    write_csv(folder / CLINICAL_CSV, header, [[row[c] for c in header] for row in clinical])
    write_csv(folder / SPLIT_CSV, ["case_id", SPLIT_COLUMN], [list(item) for item in splits.items()])
    for modality, (case_ids, matrix) in vectors.items():
        file_ids = reports if modality == REPORT_MODALITY else {}
        write_modality(folder, modality, case_ids, matrix, file_ids)


def publish(staging: Path, output: Path, old: list[Path]) -> None:
    """Move the old export into ``staging/old``, then the finished export from ``staging/new`` into ``output``."""
    new = staging / NEW_EXPORT
    if not output.exists():
        new.rename(output)
        return
    trash = staging / OLD_EXPORT
    trash.mkdir()
    for entry in old:
        entry.rename(trash / entry.name)
    for entry in sorted(new.iterdir()):
        entry.rename(output / entry.name)


def old_files_place(staging: Path, output: Path, old: list[Path]) -> str:
    """Say where the old export is after a failed move."""
    if not old:
        return ""
    trash = staging / OLD_EXPORT
    moved = len(list(trash.iterdir())) if trash.is_dir() else 0
    if moved == 0:
        return f" The old files are still in {output}."
    if moved == len(old):
        return f" The old files are in {trash}."
    return f" The old files are split between {trash} and {output}."


def run(clinical_csv: Path, splits_csv: Path, reports_csv: Path, parquet_dir: Path, output: Path,
        *, overwrite: bool = False) -> dict:
    """Validate all inputs, write the export to a temp folder, then move it to ``output``."""
    output = Path(output).resolve()
    parquets = {m: Path(parquet_dir) / f"{m}.parquet" for m in DIMS}
    inputs = [clinical_csv, splits_csv, reports_csv, *parquets.values()]
    check_inputs(inputs)
    check_output(output, inputs)

    splits = read_splits(splits_csv)
    cases = set(splits)
    for case_id in cases:
        check_file_id(case_id)
    header, clinical = read_clinical(clinical_csv, list(splits))
    reports = report_file_ids(reports_csv, cases)
    vectors = {m: read_modality(parquets[m], dim, cases) for m, dim in DIMS.items()}
    old_export(output, overwrite)  # fail before the long write
    if os.path.ismount(output):
        raise ExportError(f"{output} is a mount point, and the export cannot be moved across file systems; "
                          "pass a folder inside it as --output")

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent))
    except OSError as error:
        raise ExportError(f"Cannot create a temp folder next to {output}: {error}") from error
    try:
        write_export(staging / NEW_EXPORT, header, clinical, splits, reports, vectors)
        old = old_export(output, overwrite)  # again: index files may have appeared during the write
    except BaseException as error:
        shutil.rmtree(staging, ignore_errors=True)  # the old export is untouched
        if isinstance(error, OSError):
            raise ExportError(f"Cannot write the export to {output}: {error}") from error
        raise
    try:
        publish(staging, output, old)
    except BaseException as error:
        new = staging / NEW_EXPORT
        if not new.exists() or not any(new.iterdir()):
            shutil.rmtree(staging, ignore_errors=True)  # Ctrl-C after the last rename: the export is complete
            raise
        # Any other failure (also Ctrl-C) can leave a mixed output; staging may hold the only copy of old files.
        raise ExportError(f"Cannot move the export into {output}: {error!r}."
                          f"{old_files_place(staging, output, old)} The new files not yet moved are in {new}. "
                          f"Rerun with --overwrite, then delete {staging}") from error
    remove(staging)
    return {"cases": len(cases), "dims": DIMS, "output": str(output)}


def main(argv: list[str] | None = None) -> None:
    """Run the export from the command line; exit with a message on bad input."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clinical", type=Path, required=True, help="authors' data/clinical.csv or its subset")
    p.add_argument("--splits", type=Path, required=True, help="authors' results/split_cases.csv or its subset")
    p.add_argument("--reports", type=Path, required=True, help="authors' data/summarized_reports.csv")
    p.add_argument("--parquet-dir", type=Path, required=True, help="folder with {expr,hist,text}.parquet")
    p.add_argument("--output", type=Path, required=True, help="PyHealth input folder to write")
    p.add_argument("--overwrite", action="store_true", help="replace an existing export")
    args = p.parse_args(argv)
    try:
        stats = run(args.clinical, args.splits, args.reports, args.parquet_dir, args.output,
                    overwrite=args.overwrite)
    except ExportError as error:
        if isinstance(error.__cause__, KeyboardInterrupt):
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(EXIT_INTERRUPTED) from error
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(stats, indent=2))
    print(f"WROTE {stats['output']}")


if __name__ == "__main__":
    main()
