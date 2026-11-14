#!/usr/bin/env python3
"""Map the authors' RNA-seq manifest to TCGA cases and keep the study subset.

The authors' ``data/expr-manifest.txt`` lists GDC file IDs, names, MD5 sums
and sizes, but not which patient each file belongs to. This script asks the
public GDC API for the case and project of every listed file, keeps the files
whose case is in ``--splits`` (e.g. the 10-project subset written by
``select_cohort.py``), and writes:

* ``--out-map``: one row per kept file
  (file_id, file_name, md5, size, case_id, project, sample_ids, sample_types).
  Patient-level metadata: keep it private.
* ``--out-manifest``: the same files in GDC Data Transfer Tool format
  (id, filename, md5, size, state), in case you prefer ``gdc-client``.

Network: needs https://api.gdc.cancer.gov (works in Colab). No login needed;
these are open-access files.

Example::

    python scripts/prepare_expression_manifest.py \\
        --expr-manifest multimodal-cancer-modeling/data/expr-manifest.txt \\
        --splits /content/tcga10/split_cases.csv \\
        --out-map /content/private/expr_files_tcga10.csv \\
        --out-manifest /content/private/expr-manifest-tcga10.txt
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

GDC_FILES = "https://api.gdc.cancer.gov/files"
CASE_RE = re.compile(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}\Z")
FIELDS = [
    "file_id",
    "file_name",
    "md5sum",
    "file_size",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
]


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    need = {"id", "filename", "md5", "size"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path}: needs GDC manifest columns {sorted(need)}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate file ids")
    return df


def query_gdc(file_ids: list[str], session: Any, *, batch_size: int = 500,
              retries: int = 3, pause: float = 2.0) -> dict[str, dict]:
    """Return {file_id: hit} for the given ids, querying in batches."""
    hits: dict[str, dict] = {}
    for start in range(0, len(file_ids), batch_size):
        batch = file_ids[start:start + batch_size]
        payload = {
            "filters": {"op": "in", "content": {"field": "files.file_id", "value": batch}},
            "fields": ",".join(FIELDS),
            "format": "JSON",
            "size": str(len(batch)),
        }
        for attempt in range(1, retries + 1):
            try:
                response = session.post(GDC_FILES, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()["data"]["hits"]
                break
            except Exception:  # network hiccup: retry, then give up loudly
                if attempt == retries:
                    raise
                time.sleep(pause * attempt)
        for hit in data:
            hits[hit["file_id"]] = hit
        print(f"GDC metadata: {min(start + batch_size, len(file_ids))}/{len(file_ids)} files",
              flush=True)
    return hits


def flatten(hit: dict) -> dict:
    cases = hit.get("cases") or []
    if len(cases) != 1:
        raise ValueError(f"File {hit.get('file_id')} is linked to {len(cases)} cases")
    case = cases[0]
    samples = case.get("samples") or []
    return {
        "case_id": case.get("submitter_id"),
        "project": (case.get("project") or {}).get("project_id"),
        "gdc_file_name": hit.get("file_name"),
        "gdc_md5": hit.get("md5sum"),
        "sample_ids": ";".join(sorted(s.get("submitter_id", "") for s in samples)),
        "sample_types": ";".join(sorted({s.get("sample_type", "") for s in samples})),
    }


def build(manifest: pd.DataFrame, study_ids: set[str], hits: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    rows, not_found = [], []
    for rec in manifest.itertuples(index=False):
        hit = hits.get(rec.id)
        if hit is None:
            not_found.append(rec.id)
            continue
        info = flatten(hit)
        if info["gdc_file_name"] != rec.filename or info["gdc_md5"] != rec.md5:
            raise ValueError(f"GDC record for {rec.id} no longer matches the authors' manifest")
        if not CASE_RE.fullmatch(str(info["case_id"])):
            raise ValueError(f"File {rec.id}: unexpected case id {info['case_id']!r}")
        rows.append({"file_id": rec.id, "file_name": rec.filename, "md5": rec.md5,
                     "size": int(rec.size), "case_id": info["case_id"],
                     "project": info["project"], "sample_ids": info["sample_ids"],
                     "sample_types": info["sample_types"]})
    every = pd.DataFrame(rows)
    kept = every[every["case_id"].isin(study_ids)].sort_values(["case_id", "file_name"])
    kept = kept.reset_index(drop=True)
    per_case = kept.groupby("case_id").size()
    stats = {
        "manifest_files": len(manifest),
        "gdc_records_found": len(rows),
        "gdc_records_missing": len(not_found),
        "study_cases_requested": len(study_ids),
        "kept_files": len(kept),
        "kept_cases": int(per_case.size),
        "cases_with_multiple_files": int((per_case > 1).sum()),
        "study_cases_without_file": len(study_ids - set(kept["case_id"])),
        "total_gib": round(kept["size"].sum() / 2**30, 2) if len(kept) else 0.0,
    }
    return kept, {"stats": stats, "missing_file_ids": not_found}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expr-manifest", type=Path, required=True, help="authors' data/expr-manifest.txt")
    p.add_argument("--splits", type=Path, required=True, help="split_cases.csv of the wanted cohort")
    p.add_argument("--out-map", type=Path, required=True, help="private per-file CSV")
    p.add_argument("--out-manifest", type=Path, help="optional gdc-client manifest subset")
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args(argv)

    import requests

    manifest = read_manifest(args.expr_manifest)
    study_ids = set(pd.read_csv(args.splits, dtype=str)["case_id"])
    with requests.Session() as session:
        hits = query_gdc(manifest["id"].tolist(), session, batch_size=args.batch_size)
    kept, report = build(manifest, study_ids, hits)

    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(args.out_map, index=False)
    if args.out_manifest:
        gdc = kept.rename(columns={"file_id": "id", "file_name": "filename"})
        gdc = gdc.assign(state="released")[["id", "filename", "md5", "size", "state"]]
        gdc.to_csv(args.out_manifest, sep="\t", index=False)
    print(json.dumps(report["stats"], indent=2))
    if report["missing_file_ids"]:
        print("NOTE: some manifest files are no longer in GDC; affected study cases are"
              " counted in study_cases_without_file.")
    print(f"Per-file map written to {args.out_map} (patient-level: keep private)")


if __name__ == "__main__":
    main()
