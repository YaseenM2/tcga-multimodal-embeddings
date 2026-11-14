"""Chunked attention must reproduce the upstream BulkRNABert forward pass.

Needs jax, dm-haiku and the BulkRNABert fork; skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
hk = pytest.importorskip("haiku")
pytest.importorskip("multiomics_open_research.bulk_rna_bert")

import jax.numpy as jnp  # noqa: E402
from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig  # noqa: E402

from conftest import load_script  # noqa: E402

bc = load_script("bulkrnabert_chunked")

# Compare in true float32 on every device. On Ampere+ GPUs (A100, H100) JAX
# otherwise uses TF32 for float32 matmuls, and cuBLAS may pick TF32 kernels
# for some matrix shapes but not others, so chunked and full attention
# would differ by TF32 rounding (~1e-3), not by anything in the chunking.
# embed_expr_bulkrnabert.py sets the same precision for real runs.
jax.config.update("jax_default_matmul_precision", "highest")


def small_config(**overrides):
    values = dict(n_genes=120, n_expressions_bins=64, embed_dim=32, init_gene_embed_dim=16,
                  project_gene_embedding=True, use_gene_embedding=True, num_attention_heads=4,
                  key_size=8, ffn_embed_dim=64, num_layers=2, use_gradient_checkpointing=False,
                  gene2vec_weights_path="unused", embeddings_layers_to_save=(1, 2),
                  use_log_normalization=True, use_max_normalization=True,
                  normalization_factor=5.87)
    values.update(overrides)
    return BulkRNABertConfig(**values)


@pytest.mark.parametrize("remat", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 50, 120, 500])
def test_chunked_equals_full_attention(chunk, remat):
    cfg = small_config(use_gradient_checkpointing=remat)
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 64, size=(2, 120)), jnp.int32)
    key = jax.random.PRNGKey(0)
    full = hk.transform(bc.build_forward_fn(cfg, query_chunk=None))
    params = full.init(key, tokens)
    reference = full.apply(params, key, tokens)

    chunked = hk.transform(bc.build_forward_fn(cfg, query_chunk=chunk))
    # Same parameter names and shapes: the pretrained checkpoint loads unchanged.
    own = chunked.init(key, tokens)
    assert jax.tree_util.tree_map(np.shape, own) == jax.tree_util.tree_map(np.shape, params)
    result = chunked.apply(params, key, tokens)
    for name in ("embeddings_1", "embeddings_2", "logits"):
        np.testing.assert_allclose(result[name], reference[name], rtol=1e-5, atol=1e-5)


def test_context_manager_restores_upstream_class():
    from multiomics_open_research.bulk_rna_bert import layers

    original = layers.MultiHeadAttention
    with bc.chunked_attention(16):
        assert layers.MultiHeadAttention is bc.ChunkedMultiHeadAttention
    assert layers.MultiHeadAttention is original
    with pytest.raises(ValueError):
        with bc.chunked_attention(0):
            pass
