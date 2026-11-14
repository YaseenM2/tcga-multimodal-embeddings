# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["pyarrow==23.0.1", "scikit-survival==0.28.0"]
# ///
"""Reference survival baseline on the team's data contract, as in Song et al.

A plain scikit-survival re-run of the authors' ``run_experiment``
(``experiments/1-survival-experiments.ipynb``), used to sanity-check real-data
results and the PyHealth pipeline. For every fold in ``folds.csv``:

* demographics (sex, age bin, race, ethnicity) and cancer type: one-hot
  (``drop="if_binary"``, fitted on all cases, as the authors do), no scaling;
* each embedding modality: ``StandardScaler`` then ``PCA(n, random_state=42)``,
  fitted on the training folds;
* one ``CoxPHSurvivalAnalysis(alpha=0.1)`` per modality;
* fusion ("multimodal" = cancer type + demographics + every embedding given):
  z-score each modality's training risk scores, fit a second CoxPH on them.

Scoring follows the paper: overall C-index per test fold, and per cancer type
the C-index over that type's test cases in each fold, averaged over folds
(Table 17). Folds where a cohort has no comparable pair are skipped and counted.

Run on the authors' full 7,982-case tables, this reproduces their published
demographics, cancer-type and combined C-indices exactly.

Inputs (``--data``, the DL4H-project ``data/`` folder): ``clinical.csv``,
``folds.csv`` and ``embeddings/{expr,hist,text}.parquet``. Missing embedding
files are skipped with a note (e.g. no ``hist.parquet``).

The paper's Table 17 models were trained on all 32 cancer types (7,982
patients); a run on the 10-cohort subset trains on 873 patients only, so the
numbers are comparable in method but not in training data.

Example:
    uv run analysis/reference_coxph.py --data data --out analysis/results/pca256
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDINGS = ("expr", "hist", "text")
LABELS = {
    "demo": "Demographics",
    "canc": "Cancer type",
    "expr": "RNA-seq",
    "hist": "Histology",
    "text": "Text",
    "multimodal": "Multimodal",
}

# Paper Table 17 (PCA 256, pan-cancer models trained on all 32 cohorts), for the
# cohorts in our 10-project subset. DLBC is not reported in the paper.
PAPER_TABLE17 = {
    "CESC": {
        "demo": 0.602,
        "expr": 0.670,
        "hist": 0.619,
        "text": 0.603,
        "multimodal": 0.704,
    },
    "READ": {
        "demo": 0.706,
        "expr": 0.538,
        "hist": 0.587,
        "text": 0.642,
        "multimodal": 0.664,
    },
    "ESCA": {
        "demo": 0.579,
        "expr": 0.447,
        "hist": 0.596,
        "text": 0.553,
        "multimodal": 0.576,
    },
    "MESO": {
        "demo": 0.503,
        "expr": 0.611,
        "hist": 0.643,
        "text": 0.514,
        "multimodal": 0.665,
    },
    "KICH": {
        "demo": 0.615,
        "expr": 0.937,
        "hist": 0.805,
        "text": 0.882,
        "multimodal": 0.937,
    },
    "UVM": {
        "demo": 0.684,
        "expr": 0.652,
        "hist": 0.675,
        "text": 0.530,
        "multimodal": 0.739,
    },
    "UCS": {
        "demo": 0.631,
        "expr": 0.495,
        "hist": 0.440,
        "text": 0.654,
        "multimodal": 0.628,
    },
    "OV": {
        "demo": 0.518,
        "expr": 0.653,
        "hist": 0.530,
        "text": 0.498,
        "multimodal": 0.588,
    },
    "CHOL": {
        "demo": 0.492,
        "expr": 0.466,
        "hist": 0.437,
        "text": 0.557,
        "multimodal": 0.493,
    },
}


def load(
    data: Path, modalities: list[str]
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """Loads clinical data, survival labels and patient-level embeddings.

    Args:
        data: Folder with ``clinical.csv``, ``folds.csv`` and ``embeddings/``.
        modalities: Embedding names to load; absent files are skipped.

    Returns:
        Cases sorted by ``case_id`` (with ``fold`` and ``age_binned``), the
        structured survival array, and one feature matrix per loaded modality.

    Raises:
        ValueError: If folds, survival times or embedding rows are missing.
    """
    import pyarrow.parquet as pq

    clin = pd.read_csv(data / "clinical.csv")
    folds = pd.read_csv(data / "folds.csv")
    df = clin.merge(folds[["case_id", "fold"]], on="case_id", validate="one_to_one")
    df = df.sort_values("case_id").reset_index(drop=True)  # as the authors do
    if len(df) != len(clin):
        raise ValueError("folds.csv does not cover every case in clinical.csv")
    dead = (df["vital_status"] == "Dead").to_numpy()
    days = np.where(dead, df["days_to_death"], df["days_to_last_follow_up"])
    days = days.astype(float)
    if np.isnan(days).any():
        raise ValueError("Missing survival time for some cases")
    y = np.array(
        list(zip(dead, days, strict=True)),
        dtype=[("Status", "?"), ("Survival_in_days", "<f8")],
    )
    df["age_binned"] = pd.cut(
        df["age"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["(0, 20]", "(20, 40]", "(40, 60]", "(60, 80]", "(80, 100]"],
    )
    feats: dict[str, np.ndarray] = {}
    for name in modalities:
        path = data / "embeddings" / f"{name}.parquet"
        if not path.exists():
            print(f"note: {path} not found; skipping {name}")
            continue
        table = pq.read_table(path).to_pandas().set_index("case_id")
        missing = set(df["case_id"]) - set(table.index)
        if missing:
            raise ValueError(f"{name}: {len(missing)} cases have no embedding")
        rows = table.loc[df["case_id"], "embedding"].map(np.asarray).to_numpy()
        feats[name] = np.stack(rows).astype(np.float32)
    return df, y, feats


def one_hot(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """One-hot encodes demographics and cancer type over all cases.

    Args:
        df: Cases with ``sex``, ``age_binned``, ``race``, ``ethnicity`` and
            ``project`` columns.

    Returns:
        ``{"demo": ..., "canc": ...}`` float32 matrices.
    """
    from sklearn.preprocessing import OneHotEncoder

    demo = OneHotEncoder(drop="if_binary", sparse_output=False, dtype=np.float32)
    canc = OneHotEncoder(drop="if_binary", sparse_output=False, dtype=np.float32)
    demo_cols = df[["sex", "age_binned", "race", "ethnicity"]].astype(str)
    return {
        "demo": demo.fit_transform(demo_cols),
        "canc": canc.fit_transform(df[["project"]]),
    }


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    pca: int | None,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits one ridge CoxPH model, optionally after z-scoring and PCA.

    Args:
        x_train: Training features.
        y_train: Training survival array.
        x_test: Test features.
        pca: PCA components, or ``None`` for no reduction.
        standardize: Whether to z-score with training statistics first.

    Returns:
        Risk scores for the training and the test cases.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    if standardize:
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    if pca is not None:
        reducer = PCA(n_components=pca, random_state=42).fit(x_train)
        x_train, x_test = reducer.transform(x_train), reducer.transform(x_test)
    cox = CoxPHSurvivalAnalysis(alpha=0.1).fit(x_train, y_train)
    return cox.predict(x_train), cox.predict(x_test)


def c_index(y: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index, or NaN when no pair is comparable.

    Args:
        y: Structured survival array.
        risk: Risk scores (higher means shorter survival).

    Returns:
        The concordance index.
    """
    from sksurv.metrics import concordance_index_censored

    try:
        result = concordance_index_censored(y["Status"], y["Survival_in_days"], risk)
    except Exception:  # all censored / no comparable pair
        return float("nan")
    return float(result[0])


def run(
    df: pd.DataFrame, y: np.ndarray, feats: dict[str, np.ndarray], *, pca: int
) -> dict[str, np.ndarray]:
    """Cross-validates every unimodal model and the two-stage fusion.

    Args:
        df: Cases from :func:`load`.
        y: Structured survival array.
        feats: Embedding matrices from :func:`load`.
        pca: PCA components for the embedding modalities.

    Returns:
        ``{model: test-set risk per case}``; each case is scored by the model
        of the fold it is a test case in.
    """
    from sklearn.preprocessing import StandardScaler

    x = {**one_hot(df), **feats}
    unimodal = [m for m in ("demo", "canc", *EMBEDDINGS) if m in x]
    risks = {m: np.full(len(df), np.nan) for m in [*unimodal, "multimodal"]}
    for fold in sorted(df["fold"].unique()):
        test = (df["fold"] == fold).to_numpy()
        train = ~test
        train_risk, test_risk = {}, {}
        for m in unimodal:
            emb = m in EMBEDDINGS
            n_comp = min(pca, int(train.sum()) - 1, x[m].shape[1]) if emb else None
            train_risk[m], test_risk[m] = fit_predict(
                x[m][train], y[train], x[m][test], pca=n_comp, standardize=emb
            )
            risks[m][test] = test_risk[m]
        fused_train, fused_test = [], []
        for m in sorted(unimodal):  # authors: sorted combo, z-scored train risks
            scaler = StandardScaler().fit(train_risk[m][:, None])
            fused_train.append(scaler.transform(train_risk[m][:, None]))
            fused_test.append(scaler.transform(test_risk[m][:, None]))
        _, risks["multimodal"][test] = fit_predict(
            np.hstack(fused_train),
            y[train],
            np.hstack(fused_test),
            pca=None,
            standardize=False,
        )
        print(f"fold {fold}: done", flush=True)
    return risks


def score(
    df: pd.DataFrame, y: np.ndarray, risks: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores overall per fold and per cohort next to the paper's Table 17.

    Args:
        df: Cases from :func:`load`.
        y: Structured survival array.
        risks: Output of :func:`run`.

    Returns:
        The overall table (one row per model) and the per-cohort table.
    """
    folds = sorted(df["fold"].unique())
    overall_rows, cohort_rows = [], []
    for model, risk in risks.items():
        per_fold = []
        for f in folds:
            mask = (df["fold"] == f).to_numpy()
            per_fold.append(c_index(y[mask], risk[mask]))
        overall_rows.append(
            {
                "model": model,
                "c_index_mean": np.nanmean(per_fold),
                "c_index_std": np.nanstd(per_fold),
                **{f"fold_{f}": v for f, v in zip(folds, per_fold, strict=True)},
            }
        )
        for project in sorted(df["project"].unique()):
            vals = []
            for f in folds:
                mask = ((df["fold"] == f) & (df["project"] == project)).to_numpy()
                vals.append(c_index(y[mask], risk[mask]))
            code = project.replace("TCGA-", "")
            paper = PAPER_TABLE17.get(code, {}).get(model, np.nan)
            ours = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
            cohort_rows.append(
                {
                    "project": code,
                    "model": model,
                    "c_index_mean": ours,
                    "valid_folds": int(np.isfinite(vals).sum()),
                    "paper_table17": paper,
                    "difference": ours - paper,
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(cohort_rows)


def main(argv: list[str] | None = None) -> None:
    """Runs the baseline and writes ``overall_c_index.csv`` and per-cohort CSVs.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True, help="DL4H-project data/")
    p.add_argument(
        "--modalities", nargs="+", default=list(EMBEDDINGS), choices=EMBEDDINGS
    )
    p.add_argument("--pca", type=int, default=256, help="PCA components (paper: 256)")
    p.add_argument("--out", type=Path, required=True, help="output folder")
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore", category=UserWarning)
    df, y, feats = load(args.data, args.modalities)
    print(
        f"{len(df)} cases, {int(y['Status'].sum())} deaths, "
        f"{df['fold'].nunique()} folds, embeddings: {', '.join(feats) or 'none'}",
        flush=True,
    )
    risks = run(df, y, feats, pca=args.pca)
    overall, cohort = score(df, y, risks)
    args.out.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.out / "overall_c_index.csv", index=False)
    cohort.to_csv(args.out / "per_cohort_c_index.csv", index=False)
    show = {"display.width": 200, "display.max_columns": None, "display.precision": 3}
    with pd.option_context(*[item for pair in show.items() for item in pair]):
        print("\nOverall C-index (mean over folds):")
        cols = ["c_index_mean", "c_index_std"]
        print(overall.set_index("model")[cols].rename(index=LABELS))
        ours = cohort.pivot(index="model", columns="project", values="c_index_mean")
        print("\nPer cohort (ours):")
        print(ours.rename(index=LABELS))
        paper = cohort.pivot(index="model", columns="project", values="paper_table17")
        print("\nPer cohort (paper Table 17, pan-cancer training):")
        print(paper.rename(index=LABELS).dropna(how="all"))
    print(f"\nWrote {args.out}/overall_c_index.csv and per_cohort_c_index.csv")


if __name__ == "__main__":
    main()
