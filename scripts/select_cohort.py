#!/usr/bin/env python3
"""Write the study subset for chosen TCGA projects (default: 10 projects).

Reads the authors' ``data/clinical.csv`` and ``results/split_cases.csv`` and
writes both files, restricted to the chosen projects, into ``--out-dir``. The
written folder is a valid root for the per-file embedding layout of
``validate_tcga_data.py`` and ``convert_h5_embeddings.py``. It is not the
PyHealth input folder; ``export_pyhealth.py`` writes that one.

Only aggregate counts are printed; ``cohort_summary.csv`` holds counts only.

Example::

    python scripts/select_cohort.py \\
        --clinical multimodal-cancer-modeling/data/clinical.csv \\
        --splits multimodal-cancer-modeling/results/split_cases.csv \\
        --out-dir /content/tcga10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Default scope: ten TCGA projects. "UCS" is TCGA-UCS; "USC" is not a TCGA code.
DEFAULT_PROJECTS = ("CESC", "CHOL", "DLBC", "ESCA", "KICH", "MESO", "OV", "READ", "UCS", "UVM")


def normalize_projects(projects: list[str]) -> list[str]:
    out = []
    for p in projects:
        code = p.strip().upper()
        code = code if code.startswith("TCGA-") else f"TCGA-{code}"
        if code not in out:
            out.append(code)
    return out


def select(clinical_csv: Path, splits_csv: Path, projects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(clinical_csv)
    splits = pd.read_csv(splits_csv)
    for need, df, name in (({"case_id", "project"}, clinical, "clinical"),
                           ({"case_id", "split", "dead"}, splits, "split")):
        if not need.issubset(df.columns):
            raise ValueError(f"{name} file lacks columns {sorted(need - set(df.columns))}")
    if clinical["case_id"].duplicated().any() or splits["case_id"].duplicated().any():
        raise ValueError("Duplicate case_id values in the input tables")
    wanted = normalize_projects(projects)
    unknown = sorted(set(wanted) - set(clinical["project"]))
    if unknown:
        raise ValueError(f"Projects not in clinical.csv: {unknown}")

    clin_sub = clinical[clinical["project"].isin(wanted)].reset_index(drop=True)
    split_sub = splits[splits["case_id"].isin(set(clin_sub["case_id"]))].reset_index(drop=True)
    if not set(split_sub["case_id"]) <= set(clinical["case_id"]):
        raise ValueError("split_cases.csv has patients missing from clinical.csv")

    merged = split_sub.merge(clin_sub[["case_id", "project"]], on="case_id")
    dead = merged["dead"].astype(str).str.lower().isin({"true", "1"})
    merged = merged.assign(dead=dead)
    summary = (
        merged.groupby("project")
        .agg(study_patients=("case_id", "size"), deaths=("dead", "sum"))
        .reindex(wanted)
        .fillna(0)
        .astype(int)
    )
    summary["censored"] = summary["study_patients"] - summary["deaths"]
    folds = pd.crosstab(merged["project"], merged["split"]).reindex(wanted).fillna(0).astype(int)
    folds.columns = [f"fold_{c}" for c in folds.columns]
    summary = summary.join(folds)
    summary.loc["TOTAL"] = summary.sum()
    return clin_sub, split_sub, summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clinical", type=Path, required=True, help="authors' data/clinical.csv")
    p.add_argument("--splits", type=Path, required=True, help="authors' results/split_cases.csv")
    p.add_argument("--projects", nargs="+", default=list(DEFAULT_PROJECTS))
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    clin_sub, split_sub, summary = select(args.clinical, args.splits, args.projects)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clin_sub.to_csv(args.out_dir / "clinical.csv", index=False)
    split_sub.to_csv(args.out_dir / "split_cases.csv", index=False)
    summary.to_csv(args.out_dir / "cohort_summary.csv", index_label="project")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary)
    print(f"clinical rows: {len(clin_sub)} | study patients with a fold: {len(split_sub)}")
    print(f"Wrote clinical.csv, split_cases.csv, cohort_summary.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
