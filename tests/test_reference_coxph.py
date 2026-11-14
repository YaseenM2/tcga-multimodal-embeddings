"""Reference CoxPH baseline: synthetic data contract with a planted signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from conftest import load_script  # noqa: E402

ref = load_script("reference_coxph")


def _contract(root, n=150, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"TCGA-{chr(65 + i % 3)}{chr(65 + i // 26 % 26)}-{i:04d}" for i in range(n)]
    signal = rng.normal(size=n)
    days = np.round(np.exp(6 - signal + rng.normal(scale=0.3, size=n)))
    dead = rng.random(n) < 0.7
    clin = pd.DataFrame({
        "case_id": ids, "project": [f"TCGA-P{i % 3}" for i in range(n)],
        "sex": rng.choice(["male", "female"], n), "age": rng.integers(25, 85, n),
        "race": rng.choice(["white", "asian"], n), "ethnicity": "not hispanic or latino",
        "vital_status": np.where(dead, "Dead", "Alive"),
        "days_to_death": np.where(dead, days, np.nan),
        "days_to_last_follow_up": np.where(dead, np.nan, days)})
    clin.to_csv(root / "clinical.csv", index=False)
    pd.DataFrame({"case_id": ids, "fold": np.arange(n) % 5}).to_csv(root / "folds.csv", index=False)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    # planted prognostic direction, shared by a block of columns so PCA keeps it
    emb[:, :6] = signal[:, None] + rng.normal(scale=0.3, size=(n, 6))
    (root / "embeddings").mkdir()
    table = pa.table({"case_id": ids, "embedding": [list(v) for v in emb],
                      "n_sources": np.ones(n, np.int32)})
    pq.write_table(table, root / "embeddings" / "expr.parquet")


def test_signal_is_found_and_outputs_are_complete(tmp_path, capsys):
    _contract(tmp_path)
    out = tmp_path / "out"
    ref.main(["--data", str(tmp_path), "--pca", "8", "--out", str(out)])
    assert "hist.parquet not found" in capsys.readouterr().out
    overall = pd.read_csv(out / "overall_c_index.csv").set_index("model")
    assert set(overall.index) == {"demo", "canc", "expr", "multimodal"}
    assert overall.loc["expr", "c_index_mean"] > 0.7  # planted signal recovered
    cohort = pd.read_csv(out / "per_cohort_c_index.csv")
    assert set(cohort["project"]) == {"P0", "P1", "P2"}
    assert (cohort["valid_folds"] <= 5).all() and cohort["paper_table17"].isna().all()


def test_missing_embedding_rows_are_an_error(tmp_path):
    _contract(tmp_path)
    t = pq.read_table(tmp_path / "embeddings" / "expr.parquet").slice(0, 100)
    pq.write_table(t, tmp_path / "embeddings" / "expr.parquet")
    with pytest.raises(ValueError, match="50 cases have no embedding"):
        ref.load(tmp_path, ["expr"])
