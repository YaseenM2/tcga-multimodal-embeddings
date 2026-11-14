"""GDC download tests with a fake HTTP session (no network)."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from conftest import load_script

dl = load_script("download_expression")

CONTENT = {"f1": b"gene_id\tx\n" * 50, "f2": b"other\n" * 70}


class FakeStream:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 7):
            yield self.body[i:i + 7]


class FakeSession:
    def __init__(self, content):
        self.content = content
        self.requests = []

    def get(self, url, stream, timeout):
        fid = url.rsplit("/", 1)[-1]
        self.requests.append(fid)
        return FakeStream(self.content[fid])


def _map(tmp_path, content=CONTENT):
    rows = [{"file_id": fid, "file_name": f"{fid}.rna_seq.augmented_star_gene_counts.tsv",
             "md5": hashlib.md5(body).hexdigest(), "size": len(body),
             "case_id": "TCGA-AA-0001" if fid == "f1" else "TCGA-BB-0002"}
            for fid, body in content.items()]
    path = tmp_path / "map.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_download_layout_and_resume(tmp_path):
    session = FakeSession(CONTENT)
    out = tmp_path / "raw"
    first = dl.run(_map(tmp_path), out, session=session, max_files=1)
    assert first == {"listed": 2, "verified_existing": 0, "downloaded": 1, "missing": 1, "passed": False}
    second = dl.run(_map(tmp_path), out, session=session)
    assert second["downloaded"] == 1 and second["verified_existing"] == 1 and second["passed"]
    assert session.requests == ["f1", "f2"]
    target = out / "TCGA-AA-0001" / "f1.rna_seq.augmented_star_gene_counts.tsv"
    assert target.read_bytes() == CONTENT["f1"]
    assert not list(out.rglob(".part-*"))
    audit = dl.run(_map(tmp_path), out, check_only=True)
    assert audit["passed"] and audit["verified_existing"] == 2


def test_corrupt_download_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    bad = FakeSession({"f1": b"truncated", "f2": CONTENT["f2"]})
    with pytest.raises(IOError, match="MD5"):
        dl.run(_map(tmp_path), tmp_path / "raw", session=bad)
    assert bad.requests.count("f1") == 3  # retried, then gave up
    assert not list((tmp_path / "raw").rglob("*.tsv"))


def test_unsafe_names_rejected(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame([{"file_id": "f1", "file_name": "../x.tsv", "md5": "0", "size": 1,
                   "case_id": "TCGA-AA-0001"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unsafe"):
        dl.read_map(path)
