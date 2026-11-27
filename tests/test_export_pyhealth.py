"""PyHealth input-folder export tests on synthetic tables and vectors (fake TCGA IDs)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from conftest import load_script  # noqa: E402

ex = load_script("export_pyhealth")

CASES = ["TCGA-AA-0001", "TCGA-BB-0002", "TCGA-CC-0003"]
OUTSIDE = "TCGA-ZZ-9999"
CLINICAL_HEADER = ["case_id", "project", "sex", "age", "race", "ethnicity", "vital_status",
                   "days_to_death", "days_to_last_follow_up"]


def _write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _clinical_row(case_id, project="TCGA-UVM"):
    return [case_id, project, "female", "61.25205479452055", "white", "not reported", "Dead", "731.0", "731.0"]


def _vectors(seed=0):
    rng = np.random.default_rng(seed)
    return {m: rng.normal(size=(len(CASES), dim)).astype(np.float32) for m, dim in ex.DIMS.items()}


def _parquet(path, case_ids, matrix):
    flat = pa.array(np.asarray(matrix, np.float32).ravel(), pa.float32())
    offsets = pa.array(np.arange(0, len(case_ids) * matrix.shape[1] + 1, matrix.shape[1], dtype=np.int32))
    table = pa.table({"case_id": pa.array(case_ids, pa.string()),
                      "embedding": pa.ListArray.from_arrays(offsets, flat),
                      "n_sources": pa.array(np.ones(len(case_ids), np.int32))})
    pq.write_table(table, path)


def _inputs(tmp_path, vectors=None):
    """Authors'-style inputs: clinical in project order, splits in case_id order."""
    src = tmp_path / "src"
    (src / "parquet").mkdir(parents=True)
    rows = [_clinical_row(CASES[2], "TCGA-OV"), _clinical_row(OUTSIDE, "TCGA-BRCA"),
            _clinical_row(CASES[1]), _clinical_row(CASES[0])]
    rows[0][3] = ""  # empty values must survive the copy
    _write_csv(src / "clinical.csv", CLINICAL_HEADER, rows)
    _write_csv(src / "split_cases.csv", ["case_id", "split", "split_order", "dead"],
               [[c, str(i), "0", "False"] for i, c in enumerate(CASES)])
    _write_csv(src / "summarized_reports.csv", ["patient_filename", "text"],
               [[f"{c}.uuid-{i}", "Line one,\nline two."] for i, c in enumerate(CASES)]
               + [[f"{OUTSIDE}.uuid-a", "x"], [f"{OUTSIDE}.uuid-b", "y"]])
    vectors = _vectors() if vectors is None else vectors
    for modality, matrix in vectors.items():
        _parquet(src / "parquet" / f"{modality}.parquet", CASES, matrix)
    return src


def _run(src, output, **kwargs):
    return ex.run(src / "clinical.csv", src / "split_cases.csv", src / "summarized_reports.csv",
                  src / "parquet", output, **kwargs)


def test_writes_loader_layout(tmp_path):
    vectors = _vectors()
    src, out = _inputs(tmp_path, vectors), tmp_path / "out"
    stats = _run(src, out)
    assert stats["cases"] == 3

    with (out / "clinical.csv").open(newline="") as f:
        clinical = list(csv.reader(f))
    assert clinical[0] == CLINICAL_HEADER
    assert [r[0] for r in clinical[1:]] == CASES  # order of split_cases.csv, outside case dropped
    assert clinical[3][3] == "" and clinical[1][1] == "TCGA-UVM"
    assert (out / "split_cases.csv").read_bytes() == (
        b"case_id,split\r\nTCGA-AA-0001,0\r\nTCGA-BB-0002,1\r\nTCGA-CC-0003,2\r\n")

    npys = sorted(p.relative_to(out / "embeddings").as_posix() for p in (out / "embeddings").rglob("*.npy"))
    assert len(npys) == 9
    assert "text/TCGA-BB-0002/TCGA-BB-0002.uuid-1.npy" in npys
    assert "hist/TCGA-BB-0002/TCGA-BB-0002.npy" in npys and "expr/TCGA-CC-0003/TCGA-CC-0003.npy" in npys
    got = np.load(out / "embeddings" / "hist" / CASES[2] / f"{CASES[2]}.npy")
    assert got.dtype == np.float32 and got.shape == (1536,)
    np.testing.assert_array_equal(got, vectors["hist"][2])


def test_refuses_existing_export_unless_overwrite(tmp_path):
    src, out = _inputs(tmp_path), tmp_path / "out"
    _run(src, out)
    stale = out / "embeddings" / "expr" / "TCGA-OLD-0000"
    stale.mkdir()
    cache = out / "tcga_multimodal_expr-pyhealth.csv"
    cache.write_text("index\n")
    with pytest.raises(ex.ExportError, match="--overwrite"):
        _run(src, out)
    _run(src, out, overwrite=True)
    assert not stale.exists() and not cache.exists()
    assert len(list((out / "embeddings").rglob("*.npy"))) == 9


def test_refuses_output_that_holds_an_input(tmp_path):
    src = _inputs(tmp_path)
    for output in (src, tmp_path):
        with pytest.raises(ex.ExportError, match="is or contains the input"):
            _run(src, output, overwrite=True)
    assert (src / "clinical.csv").exists()


def test_refuses_to_remove_foreign_embeddings(tmp_path):
    src, out = _inputs(tmp_path), tmp_path / "out"
    (out / "embeddings" / "histology" / CASES[0]).mkdir(parents=True)
    with pytest.raises(ex.ExportError, match="'histology'"):
        _run(src, out, overwrite=True)
    assert (out / "embeddings" / "histology" / CASES[0]).is_dir()


def test_refuses_symlink_at_generated_path(tmp_path):
    src, out = _inputs(tmp_path), tmp_path / "out"
    out.mkdir()
    (out / "clinical.csv").symlink_to(src / "clinical.csv")
    with pytest.raises(ex.ExportError, match="symlink"):
        _run(src, out, overwrite=True)


def test_report_must_be_unique_and_present(tmp_path):
    src = _inputs(tmp_path)
    reports = src / "summarized_reports.csv"
    _write_csv(reports, ["patient_filename", "text"], [[f"{c}.u", "x"] for c in CASES] + [[f"{CASES[0]}.v", "y"]])
    with pytest.raises(ex.ExportError, match="more than one report"):
        _run(src, tmp_path / "out")
    _write_csv(reports, ["patient_filename", "text"], [[f"{c}.u", "x"] for c in CASES[:2]])
    with pytest.raises(ex.ExportError, match="no report for 1"):
        _run(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_clinical_case_rejected(tmp_path):
    src = _inputs(tmp_path)
    _write_csv(src / "clinical.csv", CLINICAL_HEADER, [_clinical_row(c) for c in CASES[:2]])
    with pytest.raises(ex.ExportError, match="lacks 1 study cases"):
        _run(src, tmp_path / "out")


def test_bad_splits_rejected(tmp_path):
    src = _inputs(tmp_path)
    _write_csv(src / "split_cases.csv", ["case_id", "fold"], [[c, "0"] for c in CASES])
    with pytest.raises(ex.ExportError, match="lacks columns"):
        _run(src, tmp_path / "out")
    _write_csv(src / "split_cases.csv", ["case_id", "split"], [[CASES[0], "0"], [CASES[0], "1"]])
    with pytest.raises(ex.ExportError, match="repeats case_id"):
        _run(src, tmp_path / "out")
    _write_csv(src / "split_cases.csv", ["case_id", "split"], [["../TCGA-AA-0001", "0"]])
    with pytest.raises(ex.ExportError, match="Unsafe file_id"):
        _run(src, tmp_path / "out")


@pytest.mark.parametrize("value", ["", "5", "-1", "1.5", "1.0", " 1", "a"])
def test_split_value_outside_folds_rejected(tmp_path, value):
    src = _inputs(tmp_path)
    _write_csv(src / "split_cases.csv", ["case_id", "split"], [[CASES[0], "0"], [CASES[1], value], [CASES[2], "2"]])
    with pytest.raises(ex.ExportError, match=f"case {CASES[1]} has split {value!r}; .* plain integer from 0 to 4"):
        _run(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("column", ["project", "vital_status", "days_to_last_follow_up"])
def test_clinical_without_loader_column_rejected(tmp_path, column):
    src = _inputs(tmp_path)
    keep = [i for i, name in enumerate(CLINICAL_HEADER) if name != column]
    _write_csv(src / "clinical.csv", [CLINICAL_HEADER[i] for i in keep],
               [[_clinical_row(c)[i] for i in keep] for c in CASES])
    with pytest.raises(ex.ExportError, match=f"lacks columns \\['{column}'\\]"):
        _run(src, tmp_path / "out")


def test_loader_contract_is_pinned():
    """Copied from PyHealth pyhealth/datasets/tcga_survival.py; change both together."""
    assert ex.CLINICAL_COLUMNS == ("case_id", "project", "sex", "age", "race", "ethnicity", "vital_status",
                                   "days_to_death", "days_to_last_follow_up")
    assert tuple(ex.DIMS) == ("expr", "hist", "text")
    assert ex.SPLIT_VALUES == ("0", "1", "2", "3", "4")
    assert (ex.CLINICAL_CSV, ex.SPLIT_CSV, ex.SPLIT_COLUMN, ex.EMBEDDINGS_DIR) == (
        "clinical.csv", "split_cases.csv", "split", "embeddings")
    assert ex.LOADER_CACHE_GLOB == "tcga_multimodal_*-pyhealth.csv"


def test_export_matches_installed_pyhealth_loader(tmp_path):
    """Runs only where PyHealth is importable; checks the contract against the real loader."""
    tcga = pytest.importorskip("pyhealth.datasets.tcga_survival")
    assert ex.CLINICAL_COLUMNS == tuple(tcga.REQUIRED_CLINICAL_COLUMNS)
    assert tuple(ex.DIMS) == tuple(tcga.EMBEDDING_TABLES)
    assert ex.SPLIT_VALUES == tuple(str(fold) for fold in range(tcga.NUM_FOLDS))
    src, out = _inputs(tmp_path), tmp_path / "out"
    _run(src, out)
    tcga.TCGAMultimodalDataset.prepare_metadata(str(out))

    index_files = {p.name for p in out.glob(ex.LOADER_CACHE_GLOB)}
    assert index_files == {"tcga_multimodal_clinical-pyhealth.csv"} | {
        f"tcga_multimodal_{m}-pyhealth.csv" for m in ex.DIMS}
    with (out / "tcga_multimodal_clinical-pyhealth.csv").open(newline="") as f:
        folds = {row["case_id"]: row["fold"] for row in csv.DictReader(f)}
    assert folds == {c: str(i) for i, c in enumerate(CASES)}
    with (out / "tcga_multimodal_text-pyhealth.csv").open(newline="") as f:
        text = [(row["case_id"], row["file_id"]) for row in csv.DictReader(f)]
    assert text == [(c, f"{c}.uuid-{i}") for i, c in enumerate(CASES)]


@pytest.mark.parametrize("problem, match", [
    ("dim", "length 256"), ("nan", "NaN"), ("missing", "exactly once"),
])
def test_bad_parquet_rejected(tmp_path, problem, match):
    src = _inputs(tmp_path)
    matrix, ids = _vectors()["expr"], CASES
    if problem == "dim":
        matrix = matrix[:, :255]
    elif problem == "nan":
        matrix[1, 3] = np.nan
    else:
        matrix, ids = matrix[:2], CASES[:2]
    _parquet(src / "parquet" / "expr.parquet", ids, matrix)
    with pytest.raises(ex.ExportError, match=match):
        _run(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cli_reports_errors_and_writes(tmp_path, capsys):
    src, out = _inputs(tmp_path), tmp_path / "out"
    argv = ["--clinical", str(src / "clinical.csv"), "--splits", str(src / "split_cases.csv"),
            "--reports", str(src / "summarized_reports.csv"), "--parquet-dir", str(src / "parquet"),
            "--output", str(out)]
    ex.main(argv)
    assert "WROTE" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="error: .*--overwrite"):
        ex.main(argv)
    ex.main([*argv, "--overwrite"])


def _snapshot(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def _temp_folders(parent):
    return sorted(p.name for p in parent.iterdir() if ".export-" in p.name)


def test_refuses_dotdot_and_symlinked_output(tmp_path, monkeypatch):
    src = _inputs(tmp_path)
    monkeypatch.chdir(src / "parquet")
    link = tmp_path / "link"
    link.symlink_to(src, target_is_directory=True)
    for output in (Path(".."), src / "parquet" / ".." / "..", link):
        with pytest.raises(ex.ExportError, match="is or contains the input"):
            _run(src, output, overwrite=True)
    assert _temp_folders(tmp_path) == [] and (src / "clinical.csv").exists()


def test_symlinked_output_writes_into_target(tmp_path):
    src, target = _inputs(tmp_path), tmp_path / "target"
    target.mkdir()
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    stats = _run(src, tmp_path / "link")
    assert stats["output"] == str(target.resolve())
    assert len(list((target / "embeddings").rglob("*.npy"))) == 9
    assert (tmp_path / "link").is_symlink() and _temp_folders(tmp_path) == []


@pytest.mark.parametrize("name", ["clinical.csv", "split_cases.csv", "summarized_reports.csv",
                                  "parquet/hist.parquet"])
def test_missing_input_rejected(tmp_path, name):
    src, out = _inputs(tmp_path), tmp_path / "out"
    _run(src, out)
    before = _snapshot(out)
    (src / name).unlink()
    with pytest.raises(ex.ExportError, match=f"input .*{name} does not exist"):
        _run(src, out, overwrite=True)
    assert _snapshot(out) == before


def test_missing_input_folder_rejected(tmp_path, capsys):
    src = _inputs(tmp_path)
    argv = ["--clinical", str(src / "clinical.csv"), "--splits", str(src / "split_cases.csv"),
            "--reports", str(src / "summarized_reports.csv"), "--parquet-dir", str(tmp_path / "nowhere"),
            "--output", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="error: input .*nowhere.* does not exist"):
        ex.main(argv)
    assert not (tmp_path / "out").exists()


def test_output_that_is_a_file_rejected(tmp_path):
    src = _inputs(tmp_path)
    blocker = tmp_path / "out.txt"
    blocker.write_text("keep me")
    for output in (blocker, blocker / "sub"):
        with pytest.raises(ex.ExportError, match="is a file, not a folder"):
            _run(src, output, overwrite=True)
    assert blocker.read_text() == "keep me" and _temp_folders(tmp_path) == []


def _failing_save(monkeypatch, after):
    calls = []
    real_save = np.save

    def save(path, array):
        calls.append(path)
        if len(calls) > after:
            raise OSError(28, "No space left on device")
        real_save(path, array)

    monkeypatch.setattr(ex.np, "save", save)


def test_crash_during_write_keeps_old_export(tmp_path, monkeypatch):
    src, out = _inputs(tmp_path), tmp_path / "out"
    _run(src, out)
    (out / "notes.txt").write_text("user file")
    before = _snapshot(out)
    _failing_save(monkeypatch, after=4)
    with pytest.raises(ex.ExportError, match="Cannot write the export .*No space left"):
        _run(src, out, overwrite=True)
    assert _snapshot(out) == before and _temp_folders(tmp_path) == []


def test_crash_during_first_write_leaves_no_output(tmp_path, monkeypatch):
    src, out = _inputs(tmp_path), tmp_path / "out"
    _failing_save(monkeypatch, after=4)
    with pytest.raises(ex.ExportError, match="Cannot write the export"):
        _run(src, out)
    assert not out.exists() and _temp_folders(tmp_path) == []
    monkeypatch.undo()
    _run(src, out)  # a failed run leaves nothing that needs --overwrite
    assert len(list((out / "embeddings").rglob("*.npy"))) == 9


def _failing_rename(monkeypatch, fails, failure, after=False):
    """Make Path.rename raise ``failure`` at the first call where ``fails(source, target)`` is true."""
    real_rename = Path.rename

    def rename(self, target):
        if not fails(self, Path(target)):
            return real_rename(self, target)
        if after:
            real_rename(self, target)
        raise failure

    monkeypatch.setattr(Path, "rename", rename)


def _nth_move_into_old(n):
    moves = []

    def fails(source, target):
        if target.parent.name == "old":
            moves.append(target)
        return len(moves) == n
    return fails


def _new_embeddings_moved_in():
    return lambda source, target: target.name == "embeddings" and source.parent.name == "new"


@pytest.mark.parametrize("failure", [OSError(5, "Input/output error"), KeyboardInterrupt()])
@pytest.mark.parametrize("make_fails, place", [
    (_new_embeddings_moved_in, "old files are in .*old\\."),
    (lambda: _nth_move_into_old(1), "old files are still in "),
    (lambda: _nth_move_into_old(2), "old files are split between "),
])
def test_failed_swap_keeps_old_files_in_temp_folder(tmp_path, monkeypatch, failure, make_fails, place):
    src, out = _inputs(tmp_path), tmp_path / "out"
    _run(src, out)
    (out / "tcga_multimodal_expr-pyhealth.csv").write_text("index\n")
    before = _snapshot(out)
    _failing_rename(monkeypatch, make_fails(), failure)
    with pytest.raises(ex.ExportError, match=f"{place}.*--overwrite") as raised:
        _run(src, out, overwrite=True)
    assert raised.value.__cause__ is failure
    [temp] = _temp_folders(tmp_path)
    old = tmp_path / temp / "old"
    assert {**_snapshot(out), **(_snapshot(old) if old.is_dir() else {})}.items() >= before.items()  # none lost
    monkeypatch.undo()
    _run(src, out, overwrite=True)  # the documented recovery
    assert len(list((out / "embeddings").rglob("*.npy"))) == 9


def test_failed_first_move_names_no_old_files(tmp_path, monkeypatch):
    src, out = _inputs(tmp_path), tmp_path / "out"
    _failing_rename(monkeypatch, lambda source, target: target == out, OSError(5, "Input/output error"))
    with pytest.raises(ex.ExportError, match="new files not yet moved") as raised:
        _run(src, out)
    assert "old files" not in str(raised.value) and not out.exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_ctrl_c_after_last_rename_is_no_failure(tmp_path, monkeypatch, overwrite):
    src, out = _inputs(tmp_path), tmp_path / "out"
    if overwrite:
        _run(src, out)
    def last(source, target):  # the last rename of publish()
        return target.name == "split_cases.csv" and source.parent.name == "new" if overwrite else target == out

    _failing_rename(monkeypatch, last, KeyboardInterrupt(), after=True)
    with pytest.raises(KeyboardInterrupt):
        _run(src, out, overwrite=overwrite)
    assert len(list((out / "embeddings").rglob("*.npy"))) == 9 and _temp_folders(tmp_path) == []


def test_cli_ctrl_c_during_swap_exits_130(tmp_path, monkeypatch, capsys):
    src, out = _inputs(tmp_path), tmp_path / "out"
    _run(src, out)
    _failing_rename(monkeypatch, lambda source, target: target.parent.name == "old", KeyboardInterrupt())
    argv = ["--clinical", str(src / "clinical.csv"), "--splits", str(src / "split_cases.csv"),
            "--reports", str(src / "summarized_reports.csv"), "--parquet-dir", str(src / "parquet"),
            "--output", str(out), "--overwrite"]
    with pytest.raises(SystemExit) as raised:
        ex.main(argv)
    assert raised.value.code == ex.EXIT_INTERRUPTED
    assert "error: Cannot move the export" in capsys.readouterr().err
    assert len(_temp_folders(tmp_path)) == 1


def test_mount_point_output_rejected(tmp_path, monkeypatch):
    src, out = _inputs(tmp_path), tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(ex.os.path, "ismount", lambda path: Path(path) == out)
    with pytest.raises(ex.ExportError, match="is a mount point.*folder inside it"):
        _run(src, out)
    assert list(out.iterdir()) == [] and _temp_folders(tmp_path) == []


def test_index_file_written_during_export_is_replaced(tmp_path, monkeypatch):
    src, out = _inputs(tmp_path), tmp_path / "out"
    _run(src, out)
    cache = out / "tcga_multimodal_hist-pyhealth.csv"
    real_save = np.save

    def save(path, array):
        cache.write_text("index\n")  # e.g. a PyHealth run in parallel
        real_save(path, array)

    monkeypatch.setattr(ex.np, "save", save)
    _run(src, out, overwrite=True)
    assert not cache.exists()
