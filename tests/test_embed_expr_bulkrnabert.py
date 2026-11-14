"""Expression embedding tests: file listing, the fork's preprocessing on one
synthetic GDC-style TSV, settings guard, and a full run with a tiny random
checkpoint (no download, CPU only). Model tests skip without jax/haiku/fork.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from conftest import load_script

ee = load_script("embed_expr_bulkrnabert")


def _touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_listing_is_sorted_and_one_per_case_is_deterministic(tmp_path):
    _touch(tmp_path / "TCGA-BB-0002" / "b2.rna.tsv")
    _touch(tmp_path / "TCGA-AA-0001" / "zz.rna.tsv")
    _touch(tmp_path / "TCGA-AA-0001" / "aa.rna.tsv")
    _touch(tmp_path / "TCGA-AA-0001" / "notes.txt")
    pairs = ee.list_expression_files(tmp_path)
    assert [(c, p.name) for c, p in pairs] == [
        ("TCGA-AA-0001", "aa.rna.tsv"), ("TCGA-AA-0001", "zz.rna.tsv"), ("TCGA-BB-0002", "b2.rna.tsv")]
    one = ee.list_expression_files(tmp_path, one_per_case=True)
    assert [p.name for _, p in one] == ["aa.rna.tsv", "b2.rna.tsv"]
    assert ee.file_identifier("6bac-f0.rna_seq.augmented_star_gene_counts.tsv") == "6bac-f0"


def test_non_case_folder_rejected(tmp_path):
    _touch(tmp_path / "TCGA-AA-0001-01A" / "x.tsv")
    with pytest.raises(ValueError, match="Not a TCGA case folder"):
        ee.list_expression_files(tmp_path)


def test_settings_guard(tmp_path):
    path = tmp_path / "e.h5"
    with h5py.File(path, "a") as h5:
        ee.check_settings(h5, {"model_name": "a", "layer": 4})
        ee.check_settings(h5, {"model_name": "a", "layer": 4})
        with pytest.raises(ValueError, match="different settings"):
            ee.check_settings(h5, {"model_name": "b", "layer": 4})


def test_audit_counts(tmp_path):
    _touch(tmp_path / "raw" / "TCGA-AA-0001" / "f1.rna.tsv")
    _touch(tmp_path / "raw" / "TCGA-AA-0001" / "f2.rna.tsv")
    pairs = ee.list_expression_files(tmp_path / "raw")
    out = tmp_path / "e.h5"
    assert ee.audit(out, pairs)["missing"] == 2
    with h5py.File(out, "w") as h5:
        h5.create_group("TCGA-AA-0001").create_dataset("f1", data=np.ones(256, np.float32))
    res = ee.audit(out, pairs)
    assert (res["present"], res["missing"], res["passed"]) == (1, 1, False)


# --------------------------------------------------------------------------
# Tests below need the BulkRNABert fork (and, for the model, jax + haiku).
# --------------------------------------------------------------------------

def _gdc_tsv(path, genes, tpm, extra_par_y=True):
    lines = ["# gene-model: GENCODE v36",
             "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second"
             "\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded",
             "N_unmapped\t\t\t1\t1\t1\t\t\t"]
    for gene, value in zip(genes, tpm):
        lines.append(f"{gene}.3\tG\tprotein_coding\t1\t1\t1\t{value}\t0\t0")
    if extra_par_y:  # PAR_Y duplicates must be dropped, as in the fork
        lines.append(f"{genes[0]}.3_PAR_Y\tG\tprotein_coding\t1\t1\t1\t999\t0\t0")
    _touch(path, "\n".join(lines) + "\n")


def test_preprocess_one_matches_fork_semantics(tmp_path):
    pytest.importorskip("multiomics_open_research.bulk_rna_bert")
    genes = [f"ENSG{i:011d}" for i in range(18_100)]
    present = genes[:-50]  # 50 reference genes absent from the file -> filled with 0
    tpm = np.round(np.random.default_rng(3).gamma(1.0, 5.0, size=len(present)), 4)
    rng_order = np.random.default_rng(4).permutation(len(present))
    tsv = tmp_path / "TCGA-AA-0001" / "uuid1.rna_seq.augmented_star_gene_counts.tsv"
    _gdc_tsv(tsv, [present[i] for i in rng_order], tpm[rng_order])
    values = ee.preprocess_one(tsv, "TCGA-AA-0001", genes, "tpm_unstranded")
    assert values.shape == (len(genes),)
    np.testing.assert_allclose(values[:-50], tpm)  # back in reference order
    assert (values[-50:] == 0).all()


def test_preprocess_one_rejects_too_few_genes(tmp_path):
    pytest.importorskip("multiomics_open_research.bulk_rna_bert")
    genes = [f"ENSG{i:011d}" for i in range(18_100)]
    tsv = tmp_path / "TCGA-AA-0001" / "uuid1.tsv"
    _gdc_tsv(tsv, genes[:100], np.ones(100))
    with pytest.raises(ValueError, match="18,000"):
        ee.preprocess_one(tsv, "TCGA-AA-0001", genes, "tpm_unstranded")


def _tiny_checkpoint(tmp_path, n_genes):
    jax = pytest.importorskip("jax")
    hk = pytest.importorskip("haiku")
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("multiomics_open_research.bulk_rna_bert")
    import jax.numpy as jnp
    from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig

    bc = load_script("bulkrnabert_chunked")
    jax.config.update("jax_default_matmul_precision", "highest")  # see test_bulkrnabert_chunked
    cfg = dict(n_genes=n_genes, n_expressions_bins=64, embed_dim=256, init_gene_embed_dim=16,
               project_gene_embedding=True, use_gene_embedding=True, num_attention_heads=8,
               key_size=32, ffn_embed_dim=64, num_layers=2, use_memory_efficient_attention=False,
               use_gradient_checkpointing=True, gene2vec_weights_path="unused",
               embeddings_layers_to_save=[], attention_layers_to_save=[],
               use_log_normalization=True, use_max_normalization=True,
               normalization_factor=5.873553674586928)
    folder = tmp_path / "checkpoints" / "tiny"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(cfg))
    model = hk.transform(bc.build_forward_fn(BulkRNABertConfig(**cfg), query_chunk=None))
    params = model.init(jax.random.PRNGKey(1), jnp.zeros((1, n_genes), jnp.int32))
    joblib.dump(params, folder / "params.joblib")
    return tmp_path / "checkpoints"


def test_embedder_chunked_equals_full_and_run_is_resumable(tmp_path, monkeypatch):
    n_genes = 40
    weights = _tiny_checkpoint(tmp_path, n_genes)
    genes = [f"ENSG{i:011d}" for i in range(n_genes)]
    values = np.random.default_rng(5).gamma(1.0, 50.0, size=n_genes)

    full = ee.Embedder(weights, "tiny", layer=2, aggregation="mean", query_chunk=0)
    chunked = ee.Embedder(weights, "tiny", layer=2, aggregation="mean", query_chunk=7)
    v_full, v_chunk = full(values.copy(), genes), chunked(values.copy(), genes)
    assert v_full.shape == (256,) and v_full.dtype == np.float32
    np.testing.assert_allclose(v_chunk, v_full, rtol=1e-5, atol=1e-6)
    # Calling twice must not change the input (the tokenizer divides in place).
    np.testing.assert_array_equal(chunked(values, genes), v_chunk)

    # End-to-end run() with preprocessing stubbed to the synthetic vector.
    gene_list = tmp_path / "genes.txt"
    gene_list.write_text("\n".join(genes) + "\n")
    raw = tmp_path / "raw"
    for case, name in [("TCGA-AA-0001", "u1.rna.tsv"), ("TCGA-AA-0001", "u2.rna.tsv"),
                       ("TCGA-BB-0002", "u3.rna.tsv")]:
        _touch(raw / case / name)
    monkeypatch.setattr(ee, "EXPECTED_N_GENES", n_genes)
    monkeypatch.setattr(ee, "preprocess_one", lambda path, case, g, col: values.copy())
    args = argparse.Namespace(expr_dir=raw, gene_list=gene_list, weights_folder=weights,
                              output_h5=tmp_path / "expr.h5", model_name="tiny", layer=2,
                              aggregation="mean", rna_seq_column="tpm_unstranded",
                              query_chunk=7, one_per_case=False, max_new=2, check_only=False)
    assert ee.run(args) == {"files": 3, "new": 2, "skipped": 0, "remaining": 1}
    with h5py.File(args.output_h5, "a") as h5:
        # A partial run must not leave an empty group for the case not yet done
        # (convert_h5_embeddings.py rejects empty groups) ...
        assert "TCGA-BB-0002" not in h5
        h5.create_group("TCGA-CC-0003")  # ... and stale empty groups get removed.
    args.max_new = None
    assert ee.run(args) == {"files": 3, "new": 1, "skipped": 2, "remaining": 0}
    with h5py.File(args.output_h5) as h5:
        assert sorted(h5.keys()) == ["TCGA-AA-0001", "TCGA-BB-0002"]
        assert sorted(h5["TCGA-AA-0001"].keys()) == ["u1", "u2"]
        assert h5["TCGA-BB-0002"]["u3"].attrs["source_file"] == "u3.rna.tsv"
        np.testing.assert_allclose(h5["TCGA-AA-0001"]["u1"][()], v_full, rtol=1e-5, atol=1e-6)
        assert json.loads(h5.attrs[ee.SETTINGS_ATTR])["layer"] == 2
    args.check_only = True
    assert ee.run(args)["passed"]
    args.check_only, args.layer = False, 1
    with pytest.raises(ValueError, match="different settings"):
        ee.run(args)
