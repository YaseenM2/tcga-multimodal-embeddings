"""GDC manifest mapping tests with a fake HTTP session (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from conftest import load_script

pm = load_script("prepare_expression_manifest")


class FakeResponse:
    def __init__(self, hits):
        self._hits = hits

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": {"hits": self._hits}}


class FakeSession:
    def __init__(self, records):
        self.records = records
        self.calls = 0

    def post(self, url, json, timeout):
        assert url == pm.GDC_FILES
        self.calls += 1
        ids = json["filters"]["content"]["value"]
        return FakeResponse([self.records[i] for i in ids if i in self.records])


def _hit(fid, name, md5, case, project="TCGA-UVM", samples=("01A",)):
    return {"file_id": fid, "file_name": name, "md5sum": md5, "file_size": 10,
            "cases": [{"submitter_id": case, "project": {"project_id": project},
                       "samples": [{"submitter_id": f"{case}-{s}", "sample_type": "Primary Tumor"}
                                   for s in samples]}]}


def _manifest(tmp_path, rows):
    df = pd.DataFrame(rows, columns=["id", "filename", "md5", "size", "state"])
    path = tmp_path / "expr-manifest.txt"
    df.to_csv(path, sep="\t", index=False)
    return pm.read_manifest(path)


def test_maps_filters_and_counts(tmp_path):
    manifest = _manifest(tmp_path, [
        ("f1", "a.tsv", "m1", 10, "released"),
        ("f2", "b.tsv", "m2", 10, "released"),
        ("f3", "c.tsv", "m3", 10, "released"),
        ("f4", "d.tsv", "m4", 10, "released"),  # gone from GDC
    ])
    session = FakeSession({
        "f1": _hit("f1", "a.tsv", "m1", "TCGA-AA-0001"),
        "f2": _hit("f2", "b.tsv", "m2", "TCGA-AA-0001", samples=("01B",)),
        "f3": _hit("f3", "c.tsv", "m3", "TCGA-ZZ-0009", project="TCGA-BRCA"),
    })
    hits = pm.query_gdc(manifest["id"].tolist(), session, batch_size=2)
    assert session.calls == 2
    kept, report = pm.build(manifest, {"TCGA-AA-0001", "TCGA-AA-0002"}, hits)
    assert kept["file_id"].tolist() == ["f1", "f2"]
    stats = report["stats"]
    assert stats["gdc_records_missing"] == 1
    assert stats["cases_with_multiple_files"] == 1
    assert stats["study_cases_without_file"] == 1
    assert report["missing_file_ids"] == ["f4"]


def test_changed_record_is_an_error(tmp_path):
    manifest = _manifest(tmp_path, [("f1", "a.tsv", "m1", 10, "released")])
    hits = {"f1": _hit("f1", "a.tsv", "DIFFERENT", "TCGA-AA-0001")}
    with pytest.raises(ValueError, match="no longer matches"):
        pm.build(manifest, {"TCGA-AA-0001"}, hits)


def test_file_linked_to_two_cases_is_an_error(tmp_path):
    manifest = _manifest(tmp_path, [("f1", "a.tsv", "m1", 10, "released")])
    hit = _hit("f1", "a.tsv", "m1", "TCGA-AA-0001")
    hit["cases"].append(hit["cases"][0])
    with pytest.raises(ValueError, match="2 cases"):
        pm.build(manifest, {"TCGA-AA-0001"}, {"f1": hit})
