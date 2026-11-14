"""Cohort subset tests on synthetic tables (fake TCGA IDs)."""
from __future__ import annotations

import pandas as pd
import pytest

from conftest import load_script

sc = load_script("select_cohort")


def _tables(tmp_path):
    clinical = pd.DataFrame({
        "case_id": ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-BB-0001", "TCGA-CC-0001", "TCGA-CC-0002"],
        "project": ["TCGA-UVM", "TCGA-UVM", "TCGA-OV", "TCGA-BRCA", "TCGA-UVM"],
    })
    splits = pd.DataFrame({
        "case_id": ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-BB-0001", "TCGA-CC-0001"],
        "split": [0, 1, 0, 1],
        "dead": [True, False, True, False],
    })
    c, s = tmp_path / "clinical.csv", tmp_path / "split_cases.csv"
    clinical.to_csv(c, index=False)
    splits.to_csv(s, index=False)
    return c, s


def test_subset_counts_and_prefix_handling(tmp_path):
    c, s = _tables(tmp_path)
    clin, split, summary = sc.select(c, s, ["uvm", "TCGA-OV"])
    # TCGA-CC-0002 is UVM but outside the study: kept in clinical, not in splits.
    assert set(clin["case_id"]) == {"TCGA-AA-0001", "TCGA-AA-0002", "TCGA-BB-0001", "TCGA-CC-0002"}
    assert set(split["case_id"]) == {"TCGA-AA-0001", "TCGA-AA-0002", "TCGA-BB-0001"}
    assert summary.loc["TCGA-UVM", "study_patients"] == 2
    assert summary.loc["TCGA-UVM", "deaths"] == 1
    assert summary.loc["TOTAL", "study_patients"] == 3
    assert summary.loc["TOTAL", "censored"] == 1


def test_unknown_project_rejected(tmp_path):
    c, s = _tables(tmp_path)
    with pytest.raises(ValueError, match="TCGA-USC"):
        sc.select(c, s, ["USC"])  # the proposal's typo for UCS


def test_cli_writes_files(tmp_path):
    c, s = _tables(tmp_path)
    out = tmp_path / "out"
    sc.main(["--clinical", str(c), "--splits", str(s), "--projects", "UVM", "--out-dir", str(out)])
    assert len(pd.read_csv(out / "split_cases.csv")) == 2
    assert {"clinical.csv", "split_cases.csv", "cohort_summary.csv"} <= {p.name for p in out.iterdir()}
