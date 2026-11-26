"""Parquet contract export tests on synthetic vectors (fake TCGA IDs)."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from conftest import load_script  # noqa: E402

ex = load_script("export_parquet")

CASES = ["TCGA-AA-0001", "TCGA-BB-0002", "TCGA-CC-0003"]


def _cases_csv(tmp_path, ids=CASES):
    path = tmp_path / "clinical.csv"
    pd.DataFrame({"case_id": ids, "project": "TCGA-UVM"}).to_csv(path, index=False)
    return path


def _npy(root, folder, case_id, name, vec):
    path = root / "embeddings" / folder / case_id / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vec)


def _contract_checks(path, expected_ids, dim):
    """The Parquet contract checks: columns, types, one row per case."""
    table = pq.read_table(path)
    assert set(table.column_names) == {"case_id", "embedding", "n_sources"}
    assert table.schema.field("case_id").type == pa.string()
    assert table.schema.field("n_sources").type == pa.int32()
    ids = table.column("case_id").to_pylist()
    assert len(ids) == len(set(ids)) and set(ids) == set(expected_ids)
    vectors = table.column("embedding").combine_chunks()
    assert str(vectors.type.value_type) == "float"
    assert set(vectors.value_lengths().to_pylist()) == {dim}
    flat = vectors.flatten()
    assert flat.null_count == 0 and all(v == v for v in flat.to_pylist())
    return table


def test_hist_npy_means_over_slides_in_clinical_order(tmp_path):
    rng = np.random.default_rng(0)
    slides = {c: [rng.normal(size=1536).astype(np.float32) for _ in range(n)]
              for c, n in zip(CASES, (2, 1, 3))}
    for case_id, vecs in slides.items():
        for i, v in enumerate(vecs):
            _npy(tmp_path, "histology", case_id, f"{case_id}-01Z-00-DX{i + 1}.x.h5", v)
    _npy(tmp_path, "histology", "TCGA-ZZ-9999", "TCGA-ZZ-9999-01Z-00-DX1.y.h5",
         np.zeros(1536, np.float32))  # outside the case list: ignored
    out = tmp_path / "hist.parquet"
    stats = ex.run("hist", _cases_csv(tmp_path), out, npy_root=tmp_path)
    assert stats["cases_written"] == 3 and stats["files_used"] == 6
    assert stats["cases_with_several_files"] == 2 and stats["source_cases_outside_list"] == 1
    table = _contract_checks(out, CASES, 1536)
    assert table.column("case_id").to_pylist() == CASES
    assert table.column("n_sources").to_pylist() == [2, 1, 3]
    got = np.asarray(table.column("embedding").to_pylist()[2], np.float32)
    np.testing.assert_array_equal(got, np.stack(slides["TCGA-CC-0003"]).mean(axis=0))


def test_expr_from_authors_h5(tmp_path):
    rng = np.random.default_rng(1)
    h5_path = tmp_path / "expr.h5"
    with h5py.File(h5_path, "w") as h5:
        for case_id in CASES:
            h5.create_group(case_id).create_dataset("uuid-a", data=rng.normal(size=256).astype(np.float32))
        h5["TCGA-AA-0001"].create_dataset("uuid-b", data=rng.normal(size=256).astype(np.float32))
    out = tmp_path / "expr.parquet"
    ex.run("expr", _cases_csv(tmp_path), out, h5=h5_path)
    table = _contract_checks(out, CASES, 256)
    assert table.column("n_sources").to_pylist() == [2, 1, 1]


def test_missing_cases_fail_unless_allowed(tmp_path):
    for case_id in CASES[:2]:
        _npy(tmp_path, "expression", case_id, "u", np.ones(256, np.float32))
    cases = _cases_csv(tmp_path)
    with pytest.raises(ValueError, match="1 of 3 cases have no vectors"):
        ex.run("expr", cases, tmp_path / "e.parquet", npy_root=tmp_path)
    assert not (tmp_path / "e.parquet").exists()
    stats = ex.run("expr", cases, tmp_path / "e.parquet", npy_root=tmp_path, allow_missing=True)
    assert stats["cases_missing"] == 1 and pq.read_table(tmp_path / "e.parquet").num_rows == 2


def test_text_sources_checked_against_reports_csv(tmp_path):
    for case_id in CASES:
        _npy(tmp_path, "report", case_id, f"{case_id}.uuid1", np.ones(4096, np.float32) / 64)
    reports = tmp_path / "reports.csv"
    pd.DataFrame({"case_id": CASES, "patient_filename": [f"{c}.uuid1" for c in CASES],
                  "summary": "x"}).to_csv(reports, index=False)
    ex.run("text", _cases_csv(tmp_path), tmp_path / "text.parquet", npy_root=tmp_path, reports_csv=reports)
    _contract_checks(tmp_path / "text.parquet", CASES, 4096)
    pd.DataFrame({"case_id": CASES, "patient_filename": [f"{c}.OTHER" for c in CASES],
                  "summary": "x"}).to_csv(reports, index=False)
    with pytest.raises(ValueError, match="reports.csv"):
        ex.run("text", _cases_csv(tmp_path), tmp_path / "t2.parquet", npy_root=tmp_path, reports_csv=reports)


def test_bad_vectors_rejected(tmp_path):
    _npy(tmp_path, "expression", "TCGA-AA-0001", "u", np.ones(255, np.float32))
    with pytest.raises(ValueError, match=r"\(256,\)"):
        ex.run("expr", _cases_csv(tmp_path, CASES[:1]), tmp_path / "e.parquet", npy_root=tmp_path)
    _npy(tmp_path, "expression", "TCGA-AA-0001", "u", np.full(256, np.nan, np.float32))
    with pytest.raises(ValueError, match="finite"):
        ex.run("expr", _cases_csv(tmp_path, CASES[:1]), tmp_path / "e.parquet", npy_root=tmp_path)
