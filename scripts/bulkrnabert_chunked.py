"""Memory-bounded (query-chunked) self-attention for the BulkRNABert JAX model.

Why this exists
---------------
BulkRNABert attends over all 19,062 gene tokens at once. The upstream
implementation materialises the full attention matrix, which for one sample is
``8 heads x 19,062 x 19,062`` float32 values (about 11.6 GB) and needs roughly
twice that at peak. That does not fit on a free Colab Tesla T4 (15 GB).

Softmax attention is computed independently for every query row, so splitting
the *query* positions into blocks and concatenating the results gives the same
output as the full computation, up to float32 rounding in the matrix multiply.
Nothing about the model, the weights, the tokenisation or the pooling changes.

How it is used
--------------
``ChunkedMultiHeadAttention`` subclasses the fork's ``MultiHeadAttention`` and
creates its Haiku sub-modules under the same names (``query``, ``key``,
``value``, ``mha_output``), so the pretrained ``params.joblib`` loads
unchanged. ``build_forward_fn`` swaps it into the model for the duration of the
Haiku transform. ``tests/test_bulkrnabert_chunked.py`` checks that a small
randomly initialised model gives the same layer outputs with and without
chunking.

Upstream code: https://github.com/StevenSong/BulkRNABert (fork of
instadeepai/multiomics-open-research, CC BY-NC-SA 4.0). This module imports
that package; it does not copy it.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Iterator, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from haiku import initializers

from multiomics_open_research.bulk_rna_bert import layers as _layers
from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig
from multiomics_open_research.bulk_rna_bert.model import (
    build_bulk_rna_bert_forward_fn,
)

DEFAULT_QUERY_CHUNK = 2048


class ChunkedMultiHeadAttention(_layers.MultiHeadAttention):
    """Drop-in replacement that processes query positions in blocks.

    The query chunk size is read from the class attribute ``query_chunk`` so
    the module signature (and therefore Haiku parameter naming) is unchanged.
    """

    query_chunk: int = DEFAULT_QUERY_CHUNK

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        attention_weight_bias: Optional[jnp.ndarray] = None,
    ) -> dict:
        if self._bias_k is not None or self._bias_v is not None:
            raise NotImplementedError("add_bias_kv is not used by BulkRNABert")
        if attention_weight_bias is not None:
            raise NotImplementedError("attention_weight_bias is not used by BulkRNABert")

        # Same projections, same sub-module names as the upstream class.
        query_heads = self._linear_projection_he_init(query, self.key_size, "query")
        key_heads = self._linear_projection_he_init(key, self.key_size, "key")
        value_heads = self._linear_projection_he_init(value, self.value_size, "value")
        sqrt_key_size = jnp.sqrt(self.key_size).astype(query.dtype)

        seq_len = query_heads.shape[-3]
        chunk = max(1, int(self.query_chunk))
        pieces = []
        for start in range(0, seq_len, chunk):
            end = min(seq_len, start + chunk)
            q = query_heads[..., start:end, :, :]
            logits = jnp.einsum("...thd,...Thd->...htT", q, key_heads)
            logits = logits / sqrt_key_size
            if attention_mask is not None:
                mask = attention_mask[..., start:end, :]
                logits = jnp.where(mask, logits, -1e30)
            weights = jax.nn.softmax(logits)
            pieces.append(jnp.einsum("...htT,...Thd->...thd", weights, value_heads))
        attention = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=-3)

        attention_vec = jnp.reshape(attention, (*attention.shape[:-2], -1))
        w_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        b_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        embeddings = hk.Linear(
            self.model_size, w_init=w_init, b_init=b_init, name="mha_output"
        )(attention_vec)
        # Full attention maps are deliberately not returned (that is the point).
        return {"embeddings": embeddings, "attention_weights": None}


@contextlib.contextmanager
def chunked_attention(query_chunk: int = DEFAULT_QUERY_CHUNK) -> Iterator[None]:
    """Temporarily make BulkRNABert build ChunkedMultiHeadAttention modules."""
    if query_chunk <= 0:
        raise ValueError("query_chunk must be positive")
    original = _layers.MultiHeadAttention
    previous_chunk = ChunkedMultiHeadAttention.query_chunk
    ChunkedMultiHeadAttention.query_chunk = int(query_chunk)
    _layers.MultiHeadAttention = ChunkedMultiHeadAttention
    try:
        yield
    finally:
        _layers.MultiHeadAttention = original
        ChunkedMultiHeadAttention.query_chunk = previous_chunk


def build_forward_fn(
    config: BulkRNABertConfig,
    *,
    query_chunk: Optional[int] = DEFAULT_QUERY_CHUNK,
) -> Callable:
    """Return an ``hk.transform``-ready forward function.

    ``query_chunk=None`` builds the unmodified upstream model (full attention).
    Module construction happens inside ``apply``, so the swap is applied there.
    """
    base = build_bulk_rna_bert_forward_fn(
        model_config=config,
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32,
        output_dtype=jnp.float32,
        model_name="bulk_bert",
    )
    if query_chunk is None:
        return base

    def forward_fn(tokens: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None):
        with chunked_attention(query_chunk):
            return base(tokens, attention_mask)

    return forward_fn
