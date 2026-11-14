"""UNI2-h batch runner tests with synthetic archives and a fake downloader."""
from __future__ import annotations

import io
import json
import shutil
import tarfile

import h5py
import numpy as np
import pandas as pd

from conftest import load_script

runner = load_script("run_uni2_projects")

DIM = 1536


def _archive(path, slides):
    with tarfile.open(path, "w:gz") as tar:
        for name, feats in slides.items():
            buf = io.BytesIO()
            with h5py.File(buf, "w") as h5:
                h5.create_dataset("features", data=feats[None].astype(np.float32))
                h5.create_dataset("coords", data=np.zeros((1, len(feats), 2), np.int32))
            data = buf.getvalue()
            info = tarfile.TarInfo(f"TCGA-X/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_download_average_delete_and_resume(tmp_path):
    rng = np.random.default_rng(1)
    feats = {"TCGA-AA-0001-01Z-00-DX1.abc.h5": rng.normal(size=(5, DIM)),
             "TCGA-ZZ-9999-01Z-00-DX1.def.h5": rng.normal(size=(3, DIM))}  # outside study
    source = tmp_path / "src.tar.gz"
    _archive(source, feats)
    splits = tmp_path / "split_cases.csv"
    pd.DataFrame({"case_id": ["TCGA-AA-0001"], "split": [0]}).to_csv(splits, index=False)

    calls = []

    def fake_download(project, download_dir):
        calls.append(project)
        target = download_dir / "TCGA" / f"TCGA-{project}.tar.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
        return target

    out = tmp_path / "root"
    res = runner.run(["uvm"], splits, out, tmp_path / "dl", download=fake_download, min_free_gib=0)
    assert calls == ["UVM"]
    assert res["UVM"]["matched"] == 1 and res["UVM"]["outside_study"] == 1
    vec = np.load(out / "embeddings/histology/TCGA-AA-0001/TCGA-AA-0001-01Z-00-DX1.abc.h5.npy")
    np.testing.assert_allclose(vec, feats["TCGA-AA-0001-01Z-00-DX1.abc.h5"].astype(np.float32).mean(0), rtol=1e-5, atol=1e-6)
    assert not (tmp_path / "dl" / "TCGA" / "TCGA-UVM.tar.gz").exists()  # archive deleted
    marker = json.loads((out / "_uni2_done" / "TCGA-UVM.json").read_text())
    assert marker["new"] == 1

    runner.run(["UVM"], splits, out, tmp_path / "dl", download=fake_download, min_free_gib=0)
    assert calls == ["UVM"]  # skipped on rerun, nothing downloaded again
