#!/usr/bin/env python3
"""Embed TCGA RNA-seq files with BulkRNABert, as in Song et al. (ML4H 2025).

Reproduces the authors' ``embed/embed_expr_bulkrnabert.py`` recipe:

* input: GDC STAR "augmented gene counts" TSVs, column ``tpm_unstranded``;
* genes: the authors' 19,062-gene list (``data/bulkrnabert_gene_list.txt``),
  missing genes filled with 0, using the fork's own preprocessing function;
* model: ``bulk_rna_bert_gtex_encode`` (not trained on TCGA), float32;
* embedding: layer-4 token embeddings, mean over the gene axis -> 256 values.

Output is the authors' HDF5 layout, so ``convert_h5_embeddings.py`` can read it
directly::

    expr.h5 / <TCGA-case-id> / <file identifier>   -> float32 (256,)

where the identifier is the GDC file name up to its first ".", exactly as the
authors' preprocessing names it.

Differences from the authors' script (numerics unchanged):

1. Attention is computed in blocks of query rows (``--query-chunk``) so one
   sample fits on a 15 GB Colab T4. ``--query-chunk 0`` uses full attention.
   Matrix multiplies use true float32 on every GPU (``--matmul-precision``),
   so A100/H100 results match a T4 or CPU run.
2. Every listed file is embedded. The authors' preprocessing keeps only the
   *first* ``*.tsv`` that the filesystem returns in each case folder, which is
   arbitrary when a case has several tumour RNA-seq files. Their modelling
   notebook averages all vectors stored under a case, so keeping all files
   and letting the loader average is the deterministic choice. Use
   ``--one-per-case`` to keep only the alphabetically first file instead.
3. Runs are resumable and refuse to mix outputs from different settings.

Layout expected under ``--expr-dir`` (written by download_expression.py)::

    <expr-dir>/<TCGA-case-id>/<gdc-file-name>.tsv

Examples::

    # GPU: embed five new files, then rerun without --max-new to finish
    python scripts/embed_expr_bulkrnabert.py --expr-dir /content/expr_raw \\
        --gene-list multimodal-cancer-modeling/data/bulkrnabert_gene_list.txt \\
        --weights-folder BulkRNABert/checkpoints --output-h5 expr.h5 --max-new 5

    # CPU, no model: audit which files are already embedded
    python scripts/embed_expr_bulkrnabert.py --expr-dir /content/expr_raw \\
        --gene-list ... --weights-folder ... --output-h5 expr.h5 --check-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

# The fork's preprocessing shows a progress bar per call (one per file here).
os.environ.setdefault("TQDM_DISABLE", "1")

CASE_RE = re.compile(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}\Z")
EXPECTED_N_GENES = 19062
EXPECTED_DIM = 256
SETTINGS_ATTR = "embedding_settings"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gene_list(path: Path) -> list[str]:
    genes = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(genes) != len(set(genes)) or not all(g.startswith("ENSG") for g in genes):
        raise ValueError(f"{path}: gene list must hold unique ENSG ids")
    return genes


def file_identifier(file_name: str) -> str:
    """The authors' preprocessing uses the file name up to the first '.'."""
    identifier = file_name.split(".")[0]
    if not identifier:
        raise ValueError(f"Cannot derive an identifier from {file_name!r}")
    return identifier


def list_expression_files(expr_dir: Path, *, one_per_case: bool = False) -> list[tuple[str, Path]]:
    """Return sorted (case_id, tsv_path) pairs from <expr_dir>/<case_id>/*.tsv."""
    expr_dir = Path(expr_dir)
    if not expr_dir.is_dir():
        raise FileNotFoundError(expr_dir)
    pairs: list[tuple[str, Path]] = []
    for case_dir in sorted(p for p in expr_dir.iterdir() if p.is_dir()):
        if not CASE_RE.fullmatch(case_dir.name):
            raise ValueError(f"Not a TCGA case folder: {case_dir}")
        files = sorted(p for p in case_dir.iterdir() if p.is_file() and p.suffix == ".tsv")
        if one_per_case:
            files = files[:1]
        seen: set[str] = set()
        for path in files:
            identifier = file_identifier(path.name)
            if identifier in seen:
                raise ValueError(f"{case_dir.name}: two files share identifier {identifier}")
            seen.add(identifier)
            pairs.append((case_dir.name, path))
    if not pairs:
        raise ValueError(f"No <case_id>/*.tsv files found under {expr_dir}")
    return pairs


def preprocess_one(tsv: Path, case_id: str, genes: list[str], column: str):
    """Run the fork's own preprocessing on exactly one file.

    ``preprocess_tcga_rna_seq_dataset`` reads one TSV per sub-folder and takes
    the case ID from the folder name, so we give it a temporary folder holding
    a single symlink. Returns a float64 array of shape (n_genes,).
    """
    from multiomics_open_research.bulk_rna_bert.preprocess import (
        preprocess_tcga_rna_seq_dataset,
    )

    with tempfile.TemporaryDirectory(prefix="bulkrnabert-one-") as tmp:
        case_dir = Path(tmp) / case_id
        case_dir.mkdir()
        (case_dir / tsv.name).symlink_to(tsv.resolve())
        df = preprocess_tcga_rna_seq_dataset(
            dataset_path=Path(tmp),
            output_file=None,
            reference_gene_ids=genes,
            rna_seq_column=column,
        )
    if len(df) != 1:
        raise ValueError(
            f"{tsv}: preprocessing kept {len(df)} rows (file unreadable, wrong column, "
            "or fewer than 18,000 reference genes)"
        )
    if df.loc[0, "case_id"] != case_id or df.loc[0, "identifier"] != file_identifier(tsv.name):
        raise ValueError(f"{tsv}: unexpected identifier/case from preprocessing")
    values = df[genes].to_numpy(dtype=np.float64)[0]
    if values.shape != (len(genes),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{tsv}: expression values must be finite and non-negative")
    return values


class Embedder:
    """Loads the checkpoint once and embeds one expression vector at a time."""

    def __init__(self, weights_folder: Path, model_name: str, layer: int,
                 aggregation: str, query_chunk: int, matmul_precision: str = "highest"):
        import haiku as hk
        import jax

        if matmul_precision == "highest":
            # Ampere+ GPUs (A100, H100) otherwise use TF32 for float32 matrix
            # multiplies; "highest" keeps true float32, as on a T4 or a CPU.
            jax.config.update("jax_default_matmul_precision", "highest")
        import jax.numpy as jnp
        import joblib
        from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig
        from multiomics_open_research.bulk_rna_bert.preprocess import (
            preprocess_rna_seq_for_bulkrnabert,
        )
        from multiomics_open_research.bulk_rna_bert.tokenizer import (
            BinnedExpressionTokenizer,
        )

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bulkrnabert_chunked

        ckpt = Path(weights_folder) / model_name
        raw = (ckpt / "config.json").read_text()
        if hasattr(BulkRNABertConfig, "model_validate_json"):  # pydantic 2
            config = BulkRNABertConfig.model_validate_json(raw)
        else:  # pydantic 1
            config = BulkRNABertConfig.parse_raw(raw)
        if layer < 1 or layer > config.num_layers:
            raise ValueError(f"layer must be in 1..{config.num_layers}")
        # Same tokenizer settings as the fork's get_pretrained_model().
        self.tokenizer = BinnedExpressionTokenizer(
            n_expressions_bins=config.n_expressions_bins,
            use_max_normalization=config.use_max_normalization,
            normalization_factor=config.normalization_factor,
            prepend_cls_token=False,
        )
        config.embeddings_layers_to_save = (layer,)
        # Gradient checkpointing only matters for training; it has no effect on
        # forward values and complicates eager inference.
        config.use_gradient_checkpointing = False
        self.config = config
        self.layer = layer
        self.aggregation = aggregation
        self.params = joblib.load(ckpt / "params.joblib")
        forward = bulkrnabert_chunked.build_forward_fn(
            config, query_chunk=(query_chunk or None)
        )
        self.forward = hk.transform(forward)
        self.key = jax.random.PRNGKey(0)
        self.jnp = jnp
        self.prep = preprocess_rna_seq_for_bulkrnabert
        n = config.n_genes
        # All-true boolean mask (the model's default is a float32 matrix of
        # ones; boolean is equivalent in jnp.where and 4x smaller).
        self.mask = jnp.ones((1, 1, n, n), dtype=jnp.bool_)
        self.params_sha256 = sha256(ckpt / "params.joblib")
        self.config_sha256 = sha256(ckpt / "config.json")

    def __call__(self, values: np.ndarray, genes: list[str]) -> np.ndarray:
        import pandas as pd

        if values.shape != (self.config.n_genes,):
            raise ValueError(f"Expected {self.config.n_genes} genes, got {values.shape}")
        frame = pd.DataFrame([values], columns=genes)
        array = np.array(self.prep(frame, self.config), dtype=np.float64, copy=True)
        # batch_tokenize divides its input in place; it gets a private copy.
        tokens = self.jnp.asarray(self.tokenizer.batch_tokenize(array), dtype=self.jnp.int32)
        outs = self.forward.apply(self.params, self.key, tokens, self.mask)
        embs = np.asarray(outs[f"embeddings_{self.layer}"])  # (1, genes, hidden)
        emb = embs.mean(axis=1) if self.aggregation == "mean" else embs.max(axis=1)
        vector = np.asarray(emb[0], dtype=np.float32)
        if vector.shape != (EXPECTED_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"Bad embedding shape/values: {vector.shape}")
        return vector


def settings_dict(args: argparse.Namespace, genes_sha: str, embedder: Embedder | None) -> dict:
    settings = {
        "model_name": args.model_name,
        "layer": args.layer,
        "aggregation": args.aggregation,
        "rna_seq_column": args.rna_seq_column,
        "gene_list_sha256": genes_sha,
        "dtype": "float32",
        "one_per_case": bool(args.one_per_case),
        "matmul_precision": getattr(args, "matmul_precision", "highest"),
    }
    if embedder is not None:
        settings["params_sha256"] = embedder.params_sha256
        settings["config_sha256"] = embedder.config_sha256
    return settings


def check_settings(h5: h5py.File, settings: dict) -> None:
    stored = h5.attrs.get(SETTINGS_ATTR)
    if stored is None:
        if len(h5.keys()):
            raise ValueError("Output HDF5 has vectors but no recorded settings; refusing to append")
        h5.attrs[SETTINGS_ATTR] = json.dumps(settings, sort_keys=True)
        return
    old = json.loads(stored)
    clash = {k: (old.get(k), v) for k, v in settings.items() if k in old and old[k] != v}
    if clash:
        raise ValueError(f"Output HDF5 was made with different settings: {clash}")


def audit(output_h5: Path, pairs: Iterable[tuple[str, Path]]) -> dict:
    counts = {"expected": 0, "present": 0, "missing": 0, "invalid": 0}
    with h5py.File(output_h5, "r") if Path(output_h5).exists() else _null() as h5:
        for case_id, path in pairs:
            counts["expected"] += 1
            identifier = file_identifier(path.name)
            if h5 is None or case_id not in h5 or identifier not in h5[case_id]:
                counts["missing"] += 1
                continue
            data = h5[case_id][identifier][()]
            if data.shape != (EXPECTED_DIM,) or not np.isfinite(data).all():
                counts["invalid"] += 1
            else:
                counts["present"] += 1
    counts["passed"] = counts["missing"] == 0 and counts["invalid"] == 0
    return counts


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def run(args: argparse.Namespace) -> dict:
    genes = read_gene_list(args.gene_list)
    if len(genes) != EXPECTED_N_GENES:
        raise ValueError(f"Expected the authors' {EXPECTED_N_GENES}-gene list, got {len(genes)}")
    pairs = list_expression_files(args.expr_dir, one_per_case=args.one_per_case)
    if args.check_only:
        return audit(args.output_h5, pairs)

    embedder = Embedder(args.weights_folder, args.model_name, args.layer,
                        args.aggregation, args.query_chunk,
                        getattr(args, "matmul_precision", "highest"))
    settings = settings_dict(args, sha256(args.gene_list), embedder)
    counts = {"files": len(pairs), "new": 0, "skipped": 0, "remaining": 0}
    started = time.time()
    with h5py.File(args.output_h5, "a") as h5:
        check_settings(h5, settings)
        for case_id, path in pairs:
            identifier = file_identifier(path.name)
            if case_id in h5 and identifier in h5[case_id]:
                if h5[case_id][identifier].attrs.get("source_file") != path.name:
                    raise ValueError(f"{case_id}/{identifier}: stored from a different file")
                counts["skipped"] += 1
                continue
            if args.max_new is not None and counts["new"] >= args.max_new:
                counts["remaining"] += 1
                continue
            values = preprocess_one(path, case_id, genes, args.rna_seq_column)
            vector = embedder(values, genes)
            # Create the case group only when a vector is written: empty groups
            # break convert_h5_embeddings.py and the authors' loading code.
            ds = h5.require_group(case_id).create_dataset(identifier, data=vector)
            ds.attrs["source_file"] = path.name
            h5.flush()
            counts["new"] += 1
            if counts["new"] <= 3 or counts["new"] % 25 == 0:
                rate = (time.time() - started) / counts["new"]
                print(f"embedded {counts['new']} files ({rate:.1f} s/file), last {case_id}",
                      flush=True)
        for case_id in [k for k in h5.keys() if len(h5[k]) == 0]:
            del h5[case_id]  # left behind by an interrupted run
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expr-dir", type=Path, required=True, help="<case_id>/<file>.tsv folders")
    p.add_argument("--gene-list", type=Path, required=True, help="authors' bulkrnabert_gene_list.txt")
    p.add_argument("--weights-folder", type=Path, required=True, help="BulkRNABert checkpoints/ folder")
    p.add_argument("--output-h5", type=Path, required=True)
    p.add_argument("--model-name", default="bulk_rna_bert_gtex_encode")
    p.add_argument("--rna-seq-column", default="tpm_unstranded")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    p.add_argument("--query-chunk", type=int, default=2048,
                   help="query rows per attention block; 0 = full attention (needs ~25 GB)")
    p.add_argument("--matmul-precision", choices=["highest", "default"], default="highest",
                   help="'highest' = true float32 matmuls on every GPU (no TF32 on A100/H100)")
    p.add_argument("--one-per-case", action="store_true",
                   help="keep only the alphabetically first file per case")
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--check-only", action="store_true", help="audit output only; no model")
    args = p.parse_args(argv)
    if args.query_chunk < 0 or (args.max_new is not None and args.max_new < 0):
        p.error("--query-chunk and --max-new must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    for key, value in result.items():
        print(f"{key}: {value}")
    if args.check_only:
        if not result["passed"]:
            raise SystemExit("EXPRESSION AUDIT FAILED!")
        print("EXPRESSION AUDIT PASSED!")
    else:
        print("EXPRESSION EMBEDDING COMPLETE!" if result["remaining"] == 0
              else "Partial run: rerun the same command to continue.")


if __name__ == "__main__":
    main()
