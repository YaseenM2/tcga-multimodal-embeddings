#!/usr/bin/env python3
"""Download open-access GDC RNA-seq files into <out-dir>/<case_id>/<file_name>.

Reads the per-file map written by ``prepare_expression_manifest.py`` and
fetches each file from the public GDC API (``/data/<file_id>``). Every file
is checked against the manifest's size and MD5 before it is moved into
place, so an interrupted run never leaves a half-written file behind. Rerun
the same command to resume; verified files are skipped.

The folder layout matches the authors' ``data-tool.py organize`` output and
is what ``embed_expr_bulkrnabert.py`` expects.

Example::

    python scripts/download_expression.py \\
        --map /content/private/expr_files_tcga10.csv \\
        --out-dir /content/expr_raw --max-files 5
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

GDC_DATA = "https://api.gdc.cancer.gov/data/"
CASE_RE = re.compile(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}\Z")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.tsv\Z")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_verified(path: Path, size: int, md5: str) -> bool:
    return path.is_file() and path.stat().st_size == size and md5_of(path) == md5


def read_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"file_id": str, "file_name": str, "md5": str, "case_id": str})
    need = {"file_id", "file_name", "md5", "size", "case_id"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path}: needs columns {sorted(need)}")
    if df["file_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate file_id rows")
    for rec in df.itertuples(index=False):
        if not CASE_RE.fullmatch(rec.case_id) or not NAME_RE.fullmatch(rec.file_name):
            raise ValueError(f"Unsafe case id or file name in {path}: {rec.case_id}/{rec.file_name}")
    return df


def fetch(session: Any, file_id: str, target: Path, size: int, md5: str,
          *, retries: int = 3, pause: float = 5.0) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".part-", delete=False) as tmp:
                tmp_name = tmp.name
                with session.get(GDC_DATA + file_id, stream=True, timeout=300) as response:
                    response.raise_for_status()
                    for block in response.iter_content(chunk_size=1 << 20):
                        tmp.write(block)
            if not is_verified(Path(tmp_name), size, md5):
                raise IOError(f"{file_id}: size/MD5 check failed")
            os.replace(tmp_name, target)
            tmp_name = None
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(pause * attempt)
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)


def run(map_csv: Path, out_dir: Path, *, session: Any = None, max_files: int | None = None,
        check_only: bool = False) -> dict:
    df = read_map(map_csv)
    counts = {"listed": len(df), "verified_existing": 0, "downloaded": 0, "missing": 0}
    for rec in df.itertuples(index=False):
        target = Path(out_dir) / rec.case_id / rec.file_name
        if is_verified(target, int(rec.size), rec.md5):
            counts["verified_existing"] += 1
            continue
        if check_only or (max_files is not None and counts["downloaded"] >= max_files):
            counts["missing"] += 1
            continue
        if session is None:
            import requests
            session = requests.Session()
        fetch(session, rec.file_id, target, int(rec.size), rec.md5)
        counts["downloaded"] += 1
        if counts["downloaded"] <= 3 or counts["downloaded"] % 50 == 0:
            print(f"downloaded {counts['downloaded']} files", flush=True)
    counts["passed"] = counts["missing"] == 0
    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", type=Path, required=True, help="CSV from prepare_expression_manifest.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-files", type=int, default=None, help="download at most N new files")
    p.add_argument("--check-only", action="store_true", help="verify existing files; no downloads")
    args = p.parse_args(argv)
    result = run(args.map, args.out_dir, max_files=args.max_files, check_only=args.check_only)
    for key, value in result.items():
        print(f"{key}: {value}")
    if result["passed"]:
        print("ALL EXPRESSION FILES PRESENT AND VERIFIED!")
    elif args.check_only:
        raise SystemExit("EXPRESSION DOWNLOAD AUDIT FAILED!")
    else:
        print("Partial download: rerun the same command to continue.")


if __name__ == "__main__":
    main()
