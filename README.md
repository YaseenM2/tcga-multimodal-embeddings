# TCGA multimodal embeddings: data preparation

Scripts that turn public TCGA inputs into the per-patient foundation-model embeddings used by
**Song et al., "Multimodal Cancer Modeling in the Age of Foundation Model Embeddings" (ML4H 2025)**
([paper](https://proceedings.mlr.press/v297/song26a.html),
[authors' code](https://github.com/StevenSong/multimodal-cancer-modeling)).

This repository is the data companion to our CS598 DLH (UIUC, Fall 2026) PyHealth contribution
(`TCGAMultimodalDataset`, `MultimodalCancerSurvival`, `CoxSurvivalModel`). Embedding generation needs
GPUs, gated downloads and extra dependencies (JAX, bitsandbytes, h5py), so it lives here rather than
in PyHealth. **No patient data, embeddings or model weights are stored in this repository.**

## What it produces

Per-file vectors (one per report, slide or RNA-seq sample) as `.npy` folders or the authors' HDF5
layout, then one of two hand-off formats:

- **Patient-level Parquet** (`export_parquet.py`): `case_id` (string), `embedding` (list<float32>,
  the float32 mean over the patient's files, as the authors' notebook computes it), `n_sources`
  (int32). One file per modality: `text.parquet` (4096), `hist.parquet` (1536), `expr.parquet` (256).
  This is the format our PyHealth dataset loads.
- **Authors' HDF5** (`pack_h5.py`): `<TCGA-case-id>/<file_id> -> float32 vector`, as read by the
  authors' notebooks. `convert_h5_embeddings.py` turns it back into `.npy` folders.

No standardisation or PCA is applied; those are fitted per training fold downstream.

| Modality | Input | Model | Script | Vector | Hardware |
|---|---|---|---|---|---|
| Pathology report | authors' `summarized_reports.csv` | BioMistral-7B (4-bit NF4), last token, L2-normalised | `embed_reports_biomistral.py` | 4096 | GPU (T4 ok) |
| Histology | gated `MahmoodLab/UNI2-h-features` tile features | UNI2-h, mean over tiles per slide | `run_uni2_projects.py` → `aggregate_uni2_histology.py` | 1536 | CPU |
| RNA-seq | GDC STAR counts, `tpm_unstranded` | BulkRNABert `bulk_rna_bert_gtex_encode`, layer 4, mean over genes | `prepare_expression_manifest.py` → `download_expression.py` → `embed_expr_bulkrnabert.py` | 256 | GPU (T4 ok) |

Supporting tools: `select_cohort.py` (subset by TCGA project), `export_parquet.py` (per-file vectors →
patient-level Parquet), `pack_h5.py` (`.npy` folders → one HDF5), `convert_h5_embeddings.py` (HDF5 →
`.npy` folders), `validate_tcga_data.py` (read-only coverage and integrity audit of `.npy` folders),
`reference_coxph.py` (the authors' survival models in scikit-survival on the Parquet data, with
per-cohort scores next to the paper's Table 17; `pip install -r requirements-analysis.txt`). Run on
the authors' full 7,982-patient tables, it reproduces their published demographics, cancer-type and
combined C-indices exactly (0.629886, 0.737020, 0.747319).

## Default scope: 10 TCGA projects

`select_cohort.py` defaults to the ten projects in our proposal. Counts are from the authors'
`clinical.csv` and `split_cases.csv`:

| Project | Patients | Deaths | Censored |
|---|---:|---:|---:|
| CESC | 248 | 60 | 188 |
| CHOL | 33 | 17 | 16 |
| DLBC | 43 | 7 | 36 |
| ESCA | 116 | 61 | 55 |
| KICH | 65 | 9 | 56 |
| MESO | 66 | 56 | 10 |
| OV | 42 | 21 | 21 |
| READ | 144 | 20 | 124 |
| UCS | 52 | 34 | 18 |
| UVM | 64 | 21 | 43 |
| **Total** | **873** | **306** | **567** |

Fold sizes: 176 / 174 / 173 / 173 / 177. Pass `--projects` to choose others (the full study has
7,982 patients in 32 projects).

## Quick start

Notebooks run the heavy parts step by step:
[`notebooks/expression_gpu.ipynb`](notebooks/expression_gpu.ipynb) (GDC download + BulkRNABert on a Colab T4),
[`notebooks/expression_icrn.ipynb`](notebooks/expression_icrn.ipynb) (the same on UIUC's free
[Illinois Computes Research Notebooks](https://jupyter.ncsa.illinois.edu) with an A100 slice) and
[`notebooks/histology_cpu.ipynb`](notebooks/histology_cpu.ipynb) (UNI2-h archives, CPU, needs your own
Hugging Face access). The same steps from a shell:

```bash
git clone https://github.com/StevenSong/multimodal-cancer-modeling.git
pip install -r requirements.txt
AUTH=multimodal-cancer-modeling

# 1. Cohort subset (CPU)
python scripts/select_cohort.py --clinical $AUTH/data/clinical.csv \
    --splits $AUTH/results/split_cases.csv --out-dir tcga10

# 2. Reports (GPU; resumable). Only the cohort in --splits is embedded.
pip install -r requirements-report.txt
python scripts/embed_reports_biomistral.py --reports $AUTH/data/summarized_reports.csv \
    --splits tcga10/split_cases.csv --output-dir tcga10/embeddings/report
python scripts/pack_h5.py --root tcga10 --modality report --splits tcga10/split_cases.csv --output-h5 tcga10/summ.h5

# 3. Histology (CPU; needs your own approved Hugging Face account, HF_TOKEN set)
python scripts/run_uni2_projects.py --splits tcga10/split_cases.csv --out-root tcga10 --download-dir /tmp/uni2

# 4. RNA-seq (metadata + download on CPU, embedding on GPU)
python scripts/prepare_expression_manifest.py --expr-manifest $AUTH/data/expr-manifest.txt \
    --splits tcga10/split_cases.csv --out-map private/expr_files_tcga10.csv
python scripts/download_expression.py --map private/expr_files_tcga10.csv --out-dir expr_raw
git clone https://github.com/StevenSong/BulkRNABert.git      # code + checkpoints (24 MB model)
pip install -r requirements-expression.txt && pip install --no-deps -e BulkRNABert
python scripts/embed_expr_bulkrnabert.py --expr-dir expr_raw \
    --gene-list $AUTH/data/bulkrnabert_gene_list.txt \
    --weights-folder BulkRNABert/checkpoints --output-h5 tcga10/expr.h5

# 5. Audit the per-file vectors, then export one patient-level Parquet per modality
python scripts/convert_h5_embeddings.py --h5 tcga10/expr.h5 --modality expression --out-root tcga10 --study-only
python scripts/validate_tcga_data.py --root tcga10 --modalities report histology expression --require-complete report
python scripts/export_parquet.py --modality text --npy-root tcga10 --cases tcga10/split_cases.csv --output out/text.parquet
python scripts/export_parquet.py --modality hist --npy-root tcga10 --cases tcga10/split_cases.csv --output out/hist.parquet
python scripts/export_parquet.py --modality expr --h5 tcga10/expr.h5 --cases tcga10/split_cases.csv --output out/expr.parquet
```

`export_parquet.py` fails if any listed case has no vector (`--allow-missing` writes the covered
cases and reports the rest).

Every long-running script is resumable (rerun the same command) and has a `--check-only` or
`--max-*` option for a small pilot first.

## Reproducibility notes: where we differ from the authors

1. **Reports.** The authors embed with vLLM in bfloat16. We load BioMistral-7B in 4-bit NF4 so it
   fits a Colab T4. Pooling (last token, L2-normalised) matches vLLM's default for this model, but the
   vectors are not bit-identical. Report-only CoxPH on all 7,982 patients (paper folds, PCA 256,
   scikit-survival): **C-index 0.7501 ± 0.0100** vs. 0.752 in the paper
   (`results/report_only_5fold.csv`).
2. **Histology.** Same per-slide mean as `embed_hist_uni2.py`, accumulated in float64 (differences
   around 1e-7). Slides are matched to patients by the TCGA barcode in the slide file name. HDF5 keys
   drop the `.h5` suffix, as the authors' keys do.
3. **RNA-seq, attention.** BulkRNABert attends over 19,062 gene tokens, and the upstream code builds
   the full attention matrix (~11.6 GB in float32, ~25 GB at peak). `bulkrnabert_chunked.py` computes
   attention in blocks of query rows instead. Softmax is per row, so the result is the same: tests
   show identical outputs on a small model, and on the real checkpoint chunk sizes 512 and 2048 agree
   to 1e-6. Peak memory for one sample drops to about 4 GB. `--query-chunk 0` restores full attention.
4. **RNA-seq, files per patient.** The authors' preprocessing reads only the *first* `*.tsv` the
   filesystem returns in each case folder, which is arbitrary when a patient has several tumour
   samples. Their modelling code averages every vector stored under a patient. We embed every file
   (deterministic) and let the loader average; `--one-per-case` keeps only the alphabetically first.
5. **RNA-seq, inputs.** Files come from the authors' `expr-manifest.txt`, mapped to patients through
   the public GDC API. If GDC has since removed a file, `prepare_expression_manifest.py` reports it
   rather than substituting another one.

Measured costs: BulkRNABert takes about 140 s per sample on 2 CPU cores, so use a GPU. BioMistral
needed a T4 for the full run.

## Data access and licences

- **TCGA open-access data** (clinical, reports, RNA-seq counts) via the GDC; cite TCGA and the GDC.
- **UNI2-h features** are gated and licensed CC BY-NC-ND 4.0. The access terms forbid redistributing
  the features, and the licence does not allow sharing adapted material, so slide vectors averaged from
  them stay private to the person who downloaded them. Teammates with their own approved access can
  regenerate them with `run_uni2_projects.py` (CPU only).
- **BulkRNABert** code and weights are CC BY-NC-SA 4.0 (InstaDeep; fork by the paper's authors).
  Non-commercial use only.
- **BioMistral-7B** is Apache-2.0.
- Code in this repository: MIT (see `LICENSE`).

Never commit raw TCGA tables, per-patient manifests, `.npy`/`.h5` embeddings, model weights or tokens
here. `.gitignore` blocks the usual file types.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q tests/
```

All tests use synthetic data and fake HTTP sessions, and run on CPU in about 25 seconds. The
BulkRNABert tests run when `jax`, `dm-haiku` and the BulkRNABert fork are installed and are skipped
otherwise.

## Citation

Song, S., Borjigin-Wang, M., Madejski, I. R., Grossman, R. L. Multimodal Cancer Modeling in the Age of
Foundation Model Embeddings. *Proceedings of the Fifth Machine Learning for Health Symposium*
(ML4H 2025), PMLR 297:202–227, 2026. <https://proceedings.mlr.press/v297/song26a.html>
