# TCGA multimodal embeddings: data preparation

Scripts that turn public TCGA inputs into the per-patient foundation-model embeddings used by
**Song et al., "Multimodal Cancer Modeling in the Age of Foundation Model Embeddings" (ML4H 2025)**
([paper](https://proceedings.mlr.press/v297/song26a.html),
[authors' code](https://github.com/StevenSong/multimodal-cancer-modeling)).

This repository is the data companion to a PyHealth contribution (`TCGAMultimodalDataset`,
`MultimodalSurvivalPrediction`, `CoxSurvivalModel`). Embedding generation needs GPUs, gated
downloads and extra dependencies (JAX, bitsandbytes, h5py), so it lives here rather than in PyHealth.
**No patient data, embeddings or model weights are stored in this repository.**

## What it produces

Per-file vectors (one per report, slide or RNA-seq sample) as `.npy` folders or the authors' HDF5
layout, then one of these hand-off formats:

- **Patient-level Parquet** (`export_parquet.py`): `case_id` (string), `embedding` (list<float32>,
  the float32 mean over the patient's files, as in the authors' notebook), `n_sources` (int32).
  One file per modality: report → `text` (4096), histology → `hist` (1536), expression → `expr`
  (256). `reference_coxph.py` and `export_pyhealth.py` read it.
- **PyHealth input folder** (`export_pyhealth.py`): the root that PyHealth's `TCGAMultimodalDataset`
  loads. See [PyHealth input folder](#pyhealth-input-folder).
- **Authors' HDF5** (`pack_h5.py`): `<TCGA-case-id>/<file_id> -> float32 vector`, as read by the
  authors' notebooks. `convert_h5_embeddings.py` turns it back into `.npy` folders.

No standardisation or PCA is applied; those are fitted per training fold downstream.

| Modality         | Input                                            | Model                                                             | Script                                                                                    | Vector | Hardware    |
| ---------------- | ------------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------ | ----------- |
| Pathology report | authors' `summarized_reports.csv`                | BioMistral-7B (4-bit NF4), last token, L2-normalised              | `embed_reports_biomistral.py`                                                             | 4096   | GPU (T4 ok) |
| Histology        | gated `MahmoodLab/UNI2-h-features` tile features | UNI2-h, mean over tiles per slide                                 | `run_uni2_projects.py` (calls `aggregate_uni2_histology.py`)                              | 1536   | CPU         |
| RNA-seq          | GDC STAR counts, `tpm_unstranded`                | BulkRNABert `bulk_rna_bert_gtex_encode`, layer 4, mean over genes | `prepare_expression_manifest.py` → `download_expression.py` → `embed_expr_bulkrnabert.py` | 256    | GPU (T4 ok) |

Supporting tools: `select_cohort.py` (subset by TCGA project), `export_parquet.py` (per-file vectors →
patient-level Parquet), `export_pyhealth.py` (Parquet → PyHealth input folder), `pack_h5.py` (`.npy`
folders → one HDF5), `convert_h5_embeddings.py` (HDF5 → `.npy` folders), `validate_tcga_data.py`
(read-only coverage and integrity audit of `.npy` folders), `reference_coxph.py` (the authors'
survival models in scikit-survival on the Parquet data, with per-cohort scores next to the paper's
Table 17; `pip install -r requirements-analysis.txt`). Run on the authors' full 7,982-patient tables,
it reproduces their published demographics, cancer-type and combined C-indices exactly (0.629886,
0.737020, 0.747319).

## Default scope: 10 TCGA projects

`select_cohort.py` defaults to these ten projects. Its `clinical.csv` has all 884 rows of the
projects, its `split_cases.csv` only the 873 study cases with a fold (below);
`export_pyhealth.py` exports only the 873.

| Project   | Patients |  Deaths | Censored |
| --------- | -------: | ------: | -------: |
| CESC      |      248 |      60 |      188 |
| CHOL      |       33 |      17 |       16 |
| DLBC      |       43 |       7 |       36 |
| ESCA      |      116 |      61 |       55 |
| KICH      |       65 |       9 |       56 |
| MESO      |       66 |      56 |       10 |
| OV        |       42 |      21 |       21 |
| READ      |      144 |      20 |      124 |
| UCS       |       52 |      34 |       18 |
| UVM       |       64 |      21 |       43 |
| **Total** |  **873** | **306** |  **567** |

Fold sizes: 176 / 174 / 173 / 173 / 177. Pass `--projects` to choose others (the full study has
7,982 patients in 32 projects).

## Quick start

Notebooks run the heavy parts step by step:
[`notebooks/expression_gpu.ipynb`](notebooks/expression_gpu.ipynb) (GDC download + BulkRNABert on a Colab T4),
[`notebooks/expression_icrn.ipynb`](notebooks/expression_icrn.ipynb) (the same on UIUC's free
[Illinois Computes Research Notebooks](https://jupyter.ncsa.illinois.edu) with an A100 slice) and
[`notebooks/histology_cpu.ipynb`](notebooks/histology_cpu.ipynb) (UNI2-h archives, CPU, needs your own
Hugging Face access). The Colab notebooks write to the Google Drive folder set by `PRIVATE` in
cell 1 (default `My Drive/tcga-multimodal-embeddings`); the ICRN notebook writes to `~/tcga10`.
The same steps from a shell, from the repository root (Python 3.12 or 3.13: tested, and needed by
`reference_coxph.py`). `AUTH` is your clone of the authors' repository (any path):

```bash
python3 -m venv .venv && source .venv/bin/activate
AUTH=../multimodal-cancer-modeling
git clone https://github.com/StevenSong/multimodal-cancer-modeling.git $AUTH
pip install -r requirements.txt

# 1. Cohort subset (CPU)
python scripts/select_cohort.py --clinical $AUTH/data/clinical.csv \
    --splits $AUTH/results/split_cases.csv --out-dir tcga10

# 2. Reports (GPU). Embeds the authors' Llama summaries (no Llama run) for the --splits cohort
pip install -r requirements-report.txt
python scripts/embed_reports_biomistral.py --reports $AUTH/data/summarized_reports.csv \
    --splits tcga10/split_cases.csv --output-dir tcga10/embeddings/report

# 3. Histology (CPU; HF_TOKEN of an approved account, see "Data access and licences").
#    Writes tcga10/embeddings/histology/<case_id>/<slide>.npy
python scripts/run_uni2_projects.py --splits tcga10/split_cases.csv --out-root tcga10 --download-dir /tmp/uni2

# 4. RNA-seq (metadata + download on CPU, embedding on GPU). The file map is
#    patient-level metadata, so it goes to the git-ignored private/
python scripts/prepare_expression_manifest.py --expr-manifest $AUTH/data/expr-manifest.txt \
    --splits tcga10/split_cases.csv --out-map private/expr_files_tcga10.csv
python scripts/download_expression.py --map private/expr_files_tcga10.csv --out-dir expr_raw
git clone https://github.com/StevenSong/BulkRNABert.git      # code + checkpoints (24 MB model)
pip install -r requirements-expression.txt && pip install --no-deps -e BulkRNABert
python scripts/embed_expr_bulkrnabert.py --expr-dir expr_raw \
    --gene-list $AUTH/data/bulkrnabert_gene_list.txt \
    --weights-folder BulkRNABert/checkpoints --output-h5 tcga10/expr.h5

# 5. Audit the per-file vectors (convert_h5_embeddings.py only feeds this audit), then export Parquet.
#    Modality names: report -> text, histology -> hist, expression -> expr
python scripts/convert_h5_embeddings.py --h5 tcga10/expr.h5 --modality expression --out-root tcga10 --study-only
python scripts/validate_tcga_data.py --root tcga10 --modalities report histology expression --require-complete report
python scripts/export_parquet.py --modality text --npy-root tcga10 --cases tcga10/split_cases.csv --output out/text.parquet
python scripts/export_parquet.py --modality hist --npy-root tcga10 --cases tcga10/split_cases.csv --output out/hist.parquet
python scripts/export_parquet.py --modality expr --h5 tcga10/expr.h5 --cases tcga10/split_cases.csv --output out/expr.parquet

# 6. PyHealth input folder (see "PyHealth input folder" below)
python scripts/export_pyhealth.py --clinical tcga10/clinical.csv --splits tcga10/split_cases.csv \
    --reports $AUTH/data/summarized_reports.csv --parquet-dir out --output pyhealth_root
# reference_coxph.py --data D (not pyhealth_root/): clinical.csv, split_cases.csv, embeddings/{expr,hist,text}.parquet
```

`export_parquet.py` fails if any listed case has no vector (`--allow-missing` writes the covered
cases and reports the rest).

Long-running scripts are resumable and have `--check-only` or `--max-*` for a small pilot.

## PyHealth input folder

PyHealth's `TCGAMultimodalDataset` reads a root folder with this layout:

```
clinical.csv                                          clinical rows of the study cases
split_cases.csv                                       case_id, split (CV fold 0-4)
embeddings/{expr,hist,text}/<case_id>/<file_id>.npy   one float32 vector per case and modality
```

`export_pyhealth.py` (quick start step 6) writes it from the Parquet files of step 5 and three files
of the [authors' repository](https://github.com/StevenSong/multimodal-cancer-modeling) (Apache-2.0):

| Input           | Source                                                                                  |
| --------------- | --------------------------------------------------------------------------------------- |
| `--clinical`    | authors' `data/clinical.csv`, or `tcga10/clinical.csv` from `select_cohort.py`          |
| `--splits`      | authors' `results/split_cases.csv`, or `tcga10/split_cases.csv` from `select_cohort.py` |
| `--reports`     | authors' `data/summarized_reports.csv` (gives the report `file_id`)                     |
| `--parquet-dir` | your own `export_parquet.py` outputs: `expr.parquet`, `hist.parquet`, `text.parquet`    |

- Before it writes, the script checks the loader's contract: each Parquet file holds exactly the
  cases of `--splits`, once each; each case has one report; `--clinical` has the 9 columns the
  loader needs; every `split` is `0` to `4`. The `text` `file_id` is the report's
  `patient_filename`, `expr` and `hist` use the `case_id`, and `hist` is the per-case mean over
  slides.
- It refuses an existing export unless you pass `--overwrite`, and an `--output` that is or contains
  an input, is a mount point or has other `embeddings/` subfolders (such as `tcga10/`). It writes to a
  hidden temp folder next to `--output` and moves it in place at the end. If the move fails or is
  interrupted (Ctrl-C: exit code 130), the error says where the old files are; rerun with
  `--overwrite`, then delete the temp folder. A hard kill or a power loss prints no error, the old
  files can then be in the hidden `.<output>.export-*/old` folder, and files are not fsynced. Run one
  export per `--output` at a time.

**UNI2-h embeddings are always your own.** Nobody may share `hist.parquet` or the export's
`embeddings/hist/` (see [Data access and licences](#data-access-and-licences)), so keep the PyHealth
input folder private.

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
4. **RNA-seq, files per patient.** The authors' preprocessing reads only the _first_ `*.tsv` the
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
  them stay private to the person who downloaded them. To get access, log in to Hugging Face, open
  [MahmoodLab/UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features) and submit
  its access request. Once it is approved, create a read token (Settings → Access Tokens), set it as
  `HF_TOKEN` and run `run_uni2_projects.py` (CPU only).
- **BulkRNABert**: the fork this pipeline uses
  ([StevenSong/BulkRNABert](https://github.com/StevenSong/BulkRNABert), by the paper's authors) is
  still CC BY-NC-SA 4.0 (code and weights, non-commercial use only). Only upstream
  [instadeepai/multiomics-open-research](https://github.com/instadeepai/multiomics-open-research)
  moved to Apache-2.0 (2026-11-19).
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
Foundation Model Embeddings. _Proceedings of the Fifth Machine Learning for Health Symposium_
(ML4H 2025), PMLR 297:202–227, 2026. <https://proceedings.mlr.press/v297/song26a.html>
