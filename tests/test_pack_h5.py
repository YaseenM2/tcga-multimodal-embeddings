"""pack_h5 <-> convert_h5_embeddings round-trip tests on synthetic vectors."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from conftest import load_script

pack = load_script("pack_h5")
conv = load_script("convert_h5_embeddings")


def _tree(root, modality, files, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    vectors = {}
    for case_id, name in files:
        path = root / "embeddings" / modality / case_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        vec = rng.normal(size=dim).astype(np.float32)
        np.save(path, vec)
        vectors[(case_id, name)] = vec
    return vectors


def _tables(root, ids):
    pd.DataFrame({"case_id": ids}).to_csv(root / "clinical.csv", index=False)
    pd.DataFrame({"case_id": ids, "split": [0] * len(ids)}).to_csv(root / "split_cases.csv", index=False)


def test_report_round_trip_is_lossless(tmp_path):
    src = tmp_path / "src"
    files = [("TCGA-AA-0001", "TCGA-AA-0001.r1.npy"), ("TCGA-AA-0001", "TCGA-AA-0001.r2.npy"),
             ("TCGA-BB-0002", "TCGA-BB-0002.r1.npy"), ("TCGA-CC-0003", "TCGA-CC-0003.r1.npy")]
    vectors = _tree(src, "report", files)
    _tables(src, ["TCGA-AA-0001", "TCGA-BB-0002"])
    out_h5 = tmp_path / "summ.h5"
    res = pack.run(src, "report", out_h5, splits=src / "split_cases.csv", expected_dim=8)
    assert res["cases"] == 2 and res["vectors"] == 3 and res["cohort_cases_missing"] == 0
    with h5py.File(out_h5) as h5:
        assert sorted(h5["TCGA-AA-0001"].keys()) == ["TCGA-AA-0001.r1", "TCGA-AA-0001.r2"]
        assert h5.attrs["modality"] == "report"

    dst = tmp_path / "dst"
    dst.mkdir()
    _tables(dst, ["TCGA-AA-0001", "TCGA-BB-0002"])
    counts = conv.convert(out_h5, "report", dst, expected_dim=8, study_only=True)
    assert counts["written"] == 3
    for (case_id, name), vec in vectors.items():
        if case_id == "TCGA-CC-0003":
            continue
        np.testing.assert_array_equal(np.load(dst / "embeddings/report" / case_id / name), vec)


def test_histology_keys_drop_h5_suffix(tmp_path):
    _tree(tmp_path, "histology", [("TCGA-AA-0001", "TCGA-AA-0001-01Z-00-DX1.x.h5.npy")])
    out_h5 = tmp_path / "hist.h5"
    pack.run(tmp_path, "histology", out_h5, expected_dim=8)
    with h5py.File(out_h5) as h5:
        assert list(h5["TCGA-AA-0001"].keys()) == ["TCGA-AA-0001-01Z-00-DX1.x"]


def test_wrong_dtype_or_dim_rejected(tmp_path):
    path = tmp_path / "embeddings/expression/TCGA-AA-0001/f.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros(8, dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        pack.run(tmp_path, "expression", tmp_path / "e.h5", expected_dim=8)
    np.save(path, np.zeros(9, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(8,\)"):
        pack.run(tmp_path, "expression", tmp_path / "e.h5", expected_dim=8)
    assert not (tmp_path / "e.h5").exists()
