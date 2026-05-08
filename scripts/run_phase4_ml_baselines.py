from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "su2_naca0012_phase3_accepted_dataset_public.csv"
RESULT_DIR = ROOT / "reproduced" / "ml_results"
FIG_DIR = ROOT / "reproduced" / "figures" / "phase4_ml"
REPORT = ROOT / "reproduced" / "phase4_ml_surrogate_baseline_report.md"


@dataclass
class PredictionResult:
    mean: np.ndarray
    std: np.ndarray | None = None


class GaussianProcessModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=8,
            random_state=42,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianProcessModel":
        xs = self.scaler.fit_transform(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(xs, y)
        return self

    def predict(self, x: np.ndarray) -> PredictionResult:
        xs = self.scaler.transform(x)
        mean, std = self.model.predict(xs, return_std=True)
        return PredictionResult(np.asarray(mean), np.asarray(std))


class BootstrapPolynomialModel:
    def __init__(self, degree: int = 3, n_bootstrap: int = 500, random_state: int = 42) -> None:
        self.degree = degree
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.models = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BootstrapPolynomialModel":
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        self.models = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            model = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(self.degree, include_bias=False),
                LinearRegression(),
            )
            model.fit(x[idx], y[idx])
            self.models.append(model)
        return self

    def predict(self, x: np.ndarray) -> PredictionResult:
        preds = np.vstack([model.predict(x) for model in self.models])
        return PredictionResult(preds.mean(axis=0), preds.std(axis=0, ddof=1))


class SklearnMeanModel:
    def __init__(self, model) -> None:
        self.model = model

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SklearnMeanModel":
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> PredictionResult:
        return PredictionResult(np.asarray(self.model.predict(x)), None)


def model_factory(name: str):
    if name.startswith("poly"):
        degree = int(name.replace("poly", ""))
        return SklearnMeanModel(
            make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree, include_bias=False),
                LinearRegression(),
            )
        )
    if name == "gpr_matern":
        return GaussianProcessModel()
    if name == "random_forest":
        return SklearnMeanModel(
            RandomForestRegressor(
                n_estimators=600,
                min_samples_leaf=1,
                random_state=42,
            )
        )
    if name == "bootstrap_poly3":
        return BootstrapPolynomialModel(degree=3, n_bootstrap=500, random_state=42)
    raise ValueError(f"Unknown model: {name}")


def metrics(y_true: np.ndarray, y_pred: np.ndarray, std: np.ndarray | None = None) -> dict[str, float | str]:
    err = y_pred - y_true
    out: dict[str, float | str] = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
    }
    if len(y_true) > 1 and np.var(y_true) > 0:
        out["r2"] = float(1.0 - np.sum(err**2) / np.sum((y_true - np.mean(y_true)) ** 2))
    else:
        out["r2"] = ""
    if std is not None:
        lower = y_pred - 1.96 * std
        upper = y_pred + 1.96 * std
        out["mean_pred_std"] = float(np.mean(std))
        out["picp_95"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        safe_std = np.maximum(std, 1e-12)
        out["mean_gaussian_nll"] = float(
            np.mean(0.5 * np.log(2 * math.pi * safe_std**2) + 0.5 * ((y_true - y_pred) / safe_std) ** 2)
        )
    else:
        out["mean_pred_std"] = ""
        out["picp_95"] = ""
        out["mean_gaussian_nll"] = ""
    return out


def credibility_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "Cl_Exp_pct_diff",
        "Cd_Exp_pct_diff",
        "Cl_drift_pct_last_200",
        "Cd_drift_pct_last_200",
        "Cl_drift_pct_last_500",
        "Cd_drift_pct_last_500",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")

    classes = []
    scores = []
    reasons = []
    for row in out.itertuples(index=False):
        score = 1.0
        reason = []
        alpha = float(row.alpha_deg)
        cd_exp = abs(getattr(row, "Cd_Exp_pct_diff")) if not pd.isna(getattr(row, "Cd_Exp_pct_diff")) else np.nan
        cl_exp = abs(getattr(row, "Cl_Exp_pct_diff")) if not pd.isna(getattr(row, "Cl_Exp_pct_diff")) else np.nan
        cd_drift_200 = abs(getattr(row, "Cd_drift_pct_last_200")) if not pd.isna(getattr(row, "Cd_drift_pct_last_200")) else 0.0
        cd_drift_500 = abs(getattr(row, "Cd_drift_pct_last_500")) if not pd.isna(getattr(row, "Cd_drift_pct_last_500")) else 0.0

        if abs(alpha) < 1e-9:
            reason.append("baseline anchor without direct experimental percent comparison")
        if cd_drift_200 > 0.5:
            score -= 0.15
            reason.append("Cd drift over last 200 iterations exceeds 0.5%")
        if cd_drift_500 > 1.0:
            score -= 0.15
            reason.append("Cd drift over last 500 iterations exceeds 1.0%")
        if not np.isnan(cd_exp) and cd_exp > 10.0:
            score -= 0.20
            reason.append("Cd differs from experiment by more than 10%")
        if not np.isnan(cl_exp) and cl_exp > 5.0:
            score -= 0.10
            reason.append("Cl differs from experiment by more than 5%")
        if alpha >= 11.0:
            score -= 0.10
            reason.append("high-alpha region flagged separately")

        score = max(0.0, min(1.0, score))
        if score >= 0.85:
            tier = "core"
        elif score >= 0.65:
            tier = "usable_with_caveat"
        else:
            tier = "diagnostic_only"
        classes.append(tier)
        scores.append(score)
        reasons.append("; ".join(reason) if reason else "no major automatic credibility penalty")

    return pd.DataFrame(
        {
            "case_id": out["case_id"],
            "alpha_deg": out["alpha_deg"],
            "Cl_SU2": out["Cl_SU2"],
            "Cd_SU2": out["Cd_SU2"],
            "credibility_score": scores,
            "credibility_tier": classes,
            "credibility_reason": reasons,
        }
    )


def evaluate_loocv(df: pd.DataFrame, model_names: list[str], target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[["alpha_deg"]].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    pred_rows = []
    metric_rows = []

    for model_name in model_names:
        pred = np.empty_like(y)
        std = np.full_like(y, np.nan)
        for i in range(len(df)):
            train = np.ones(len(df), dtype=bool)
            train[i] = False
            model = model_factory(model_name).fit(x[train], y[train])
            result = model.predict(x[[i]])
            pred[i] = result.mean[0]
            if result.std is not None:
                std[i] = result.std[0]
            pred_rows.append(
                {
                    "evaluation": "loocv",
                    "target": target,
                    "model": model_name,
                    "case_id": df.iloc[i]["case_id"],
                    "alpha_deg": float(df.iloc[i]["alpha_deg"]),
                    "actual": float(y[i]),
                    "predicted": float(pred[i]),
                    "predicted_std": "" if np.isnan(std[i]) else float(std[i]),
                    "error": float(pred[i] - y[i]),
                    "abs_error": float(abs(pred[i] - y[i])),
                }
            )
        m = metrics(y, pred, None if np.all(np.isnan(std)) else std)
        metric_rows.append({"evaluation": "loocv", "target": target, "model": model_name, "n_train": len(df) - 1, "n_test": len(df), **m})
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def evaluate_high_alpha_holdout(df: pd.DataFrame, model_names: list[str], target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = df["alpha_deg"] <= 10.12
    test_mask = df["alpha_deg"] > 10.12
    x_train = df.loc[train_mask, ["alpha_deg"]].to_numpy(dtype=float)
    y_train = df.loc[train_mask, target].to_numpy(dtype=float)
    x_test = df.loc[test_mask, ["alpha_deg"]].to_numpy(dtype=float)
    y_test = df.loc[test_mask, target].to_numpy(dtype=float)
    test_df = df.loc[test_mask].reset_index(drop=True)

    pred_rows = []
    metric_rows = []
    for model_name in model_names:
        model = model_factory(model_name).fit(x_train, y_train)
        result = model.predict(x_test)
        std = result.std
        metric_rows.append(
            {
                "evaluation": "high_alpha_holdout_train_alpha_le_10p12",
                "target": target,
                "model": model_name,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                **metrics(y_test, result.mean, std),
            }
        )
        for i, row in test_df.iterrows():
            pred_rows.append(
                {
                    "evaluation": "high_alpha_holdout_train_alpha_le_10p12",
                    "target": target,
                    "model": model_name,
                    "case_id": row["case_id"],
                    "alpha_deg": float(row["alpha_deg"]),
                    "actual": float(y_test[i]),
                    "predicted": float(result.mean[i]),
                    "predicted_std": "" if std is None else float(std[i]),
                    "error": float(result.mean[i] - y_test[i]),
                    "abs_error": float(abs(result.mean[i] - y_test[i])),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def dense_predictions(df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    x_grid = np.linspace(df["alpha_deg"].min(), df["alpha_deg"].max(), 400).reshape(-1, 1)
    rows = []
    for target in ["Cl_SU2", "Cd_SU2"]:
        x = df[["alpha_deg"]].to_numpy(dtype=float)
        y = df[target].to_numpy(dtype=float)
        for model_name in model_names:
            model = model_factory(model_name).fit(x, y)
            result = model.predict(x_grid)
            for alpha, mean, std in zip(x_grid.ravel(), result.mean, np.full(len(x_grid), np.nan) if result.std is None else result.std):
                rows.append(
                    {
                        "alpha_deg": float(alpha),
                        "target": target,
                        "model": model_name,
                        "predicted": float(mean),
                        "predicted_std": "" if np.isnan(std) else float(std),
                    }
                )
    return pd.DataFrame(rows)


def plot_surrogates(df: pd.DataFrame, dense: pd.DataFrame, target: str, ylabel: str, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    core = df[~df["credibility_tier"].eq("usable_with_caveat")]
    caveat = df[df["credibility_tier"].eq("usable_with_caveat")]
    ax.scatter(core["alpha_deg"], core[target], s=58, color="#1f5f99", label="accepted CFD")
    ax.scatter(caveat["alpha_deg"], caveat[target], s=68, color="#b55a30", marker="s", label="caveated high-alpha CFD")
    colors = {"poly3": "#222222", "gpr_matern": "#2f7d32", "random_forest": "#7b4aa0", "bootstrap_poly3": "#be8a00"}
    labels = {
        "poly3": "cubic response surface",
        "gpr_matern": "Gaussian process",
        "random_forest": "random forest",
        "bootstrap_poly3": "bootstrap cubic mean",
    }
    for model_name, group in dense[dense["target"].eq(target)].groupby("model"):
        ax.plot(group["alpha_deg"], group["predicted"], lw=2.0, color=colors.get(model_name), label=labels.get(model_name, model_name))
        std = pd.to_numeric(group["predicted_std"], errors="coerce")
        if model_name in {"gpr_matern", "bootstrap_poly3"} and std.notna().any():
            y = group["predicted"].to_numpy(dtype=float)
            x = group["alpha_deg"].to_numpy(dtype=float)
            s = std.to_numpy(dtype=float)
            ax.fill_between(x, y - 1.96 * s, y + 1.96 * s, color=colors.get(model_name), alpha=0.12, linewidth=0)
    ax.set_xlabel("Angle of attack, alpha (deg)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    fig.savefig(out_stem.with_suffix(".svg"))
    plt.close(fig)


def plot_holdout(predictions: pd.DataFrame, target: str, ylabel: str, out_stem: Path) -> None:
    subset = predictions[
        predictions["evaluation"].eq("high_alpha_holdout_train_alpha_le_10p12")
        & predictions["target"].eq(target)
    ].copy()
    models = ["poly3", "gpr_matern", "random_forest", "bootstrap_poly3"]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    actual = subset[subset["model"].eq(models[0])][["alpha_deg", "actual"]].sort_values("alpha_deg")
    ax.plot(actual["alpha_deg"], actual["actual"], marker="o", lw=2.0, color="#111111", label="actual CFD")
    for model_name in models:
        group = subset[subset["model"].eq(model_name)].sort_values("alpha_deg")
        ax.plot(group["alpha_deg"], group["predicted"], marker=".", lw=1.8, label=model_name)
    ax.set_xlabel("Held-out high-alpha case (deg)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    fig.savefig(out_stem.with_suffix(".svg"))
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    view = df[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(df: pd.DataFrame, metrics_df: pd.DataFrame, predictions: pd.DataFrame, credibility: pd.DataFrame) -> None:
    loocv = metrics_df[metrics_df["evaluation"].eq("loocv")].copy()
    high = metrics_df[metrics_df["evaluation"].eq("high_alpha_holdout_train_alpha_le_10p12")].copy()
    best_loocv = loocv.sort_values(["target", "rmse"]).groupby("target", as_index=False).first()
    best_high = high.sort_values(["target", "rmse"]).groupby("target", as_index=False).first()

    report = f"""# Phase 4 ML Surrogate Baseline Report

Date: 2026-05-05

## Purpose

Train first-pass surrogate and uncertainty baselines on the accepted SU2 NACA 0012 Phase 3 dataset. This is a model-selection and feasibility checkpoint, not the final manuscript result.

## Dataset

- Rows: {len(df)}
- Alpha range: {df["alpha_deg"].min():.2f} deg to {df["alpha_deg"].max():.2f} deg
- Targets: `Cl_SU2`, `Cd_SU2`
- Inputs: angle of attack only
- High-alpha rows, `alpha >= 11 deg`: {int((df["alpha_deg"] >= 11.0).sum())}
- Rows with automatic caveat or diagnostic tier: {int((~credibility["credibility_tier"].eq("core")).sum())}

Credibility classes:

{markdown_table(credibility, ["case_id", "alpha_deg", "credibility_score", "credibility_tier", "credibility_reason"])}

## Models tested

- `poly1`, `poly2`, `poly3`, `poly4`: deterministic polynomial response surfaces.
- `gpr_matern`: Gaussian-process surrogate with a Matern kernel and predictive standard deviation.
- `random_forest`: tree-based baseline; expected to be weak for extrapolation.
- `bootstrap_poly3`: bootstrap ensemble of cubic response surfaces for an empirical uncertainty band.

No neural network was trained because {len(df)} rows is too small for a defensible deep-learning claim.

## Leave-one-out validation

Best LOOCV model by target:

{markdown_table(best_loocv, ["target", "model", "n_test", "mae", "rmse", "max_abs_error", "r2", "mean_pred_std", "picp_95"])}

Full metrics are stored in:

- `./06_ml_pipeline\\results\\phase4_model_metrics.csv`

## High-alpha holdout test

This test trains only on cases with `alpha <= 10.12 deg` and predicts the held-out high-alpha cases above 10.12 deg. It is intentionally difficult and checks whether the model can extrapolate into the caveated nonlinear region.

Best high-alpha holdout model by target:

{markdown_table(best_high, ["target", "model", "n_test", "mae", "rmse", "max_abs_error", "r2", "mean_pred_std", "picp_95"])}

Important interpretation:

- A model that performs well in LOOCV but poorly in high-alpha holdout should not be presented as robust outside its sampled regime.
- Tree-based models are included as a cautionary baseline because they do not extrapolate naturally in one-dimensional sparse data.
- Gaussian-process and bootstrap uncertainty bands are useful for figures, but their calibration remains weak with only {len(df)} CFD points.

## Generated artifacts

- `./06_ml_pipeline\\results\\phase4_model_metrics.csv`
- `./06_ml_pipeline\\results\\phase4_model_predictions.csv`
- `./06_ml_pipeline\\results\\phase4_dense_predictions.csv`
- `./06_ml_pipeline\\results\\phase4_credibility_dataset.csv`
- `./07_figures\\phase4_ml\\cl_surrogate_comparison.png`
- `./07_figures\\phase4_ml\\cd_surrogate_comparison.png`
- `./07_figures\\phase4_ml\\cl_high_alpha_holdout.png`
- `./07_figures\\phase4_ml\\cd_high_alpha_holdout.png`

## Phase decision

{phase_decision(df)}
"""
    REPORT.write_text(report, encoding="utf-8")


def phase_decision(df: pd.DataFrame) -> str:
    max_alpha = float(df["alpha_deg"].max())
    if max_alpha >= 14.22:
        return (
            "The current dataset is sufficient for manuscript-grade Phase 4 surrogate figures and tables, "
            "including nonlinear drag-growth coverage through alpha = 14.22 deg. It is still not sufficient "
            "for a broad deep-learning or geometry-general surrogate claim. The next scientifically strongest "
            "move is to refine credibility-aware reporting, select final models, and prepare manuscript result "
            "tables. Do not run alpha = 15.26 deg unless the final model-selection review shows a specific "
            "gap that cannot be handled by the current caveated high-alpha data."
        )
    return (
        "The current dataset is sufficient to develop manuscript-grade ML figures, but not yet sufficient for a broad "
        "deep-learning or general surrogate claim. The scientifically strongest next step is to refine credibility-aware "
        "reporting and decide whether to add one targeted CFD case at alpha = 14.22 deg for nonlinear drag growth, "
        "rather than expanding the sweep mechanically."
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET)
    df = df.sort_values("alpha_deg").reset_index(drop=True)
    credibility = credibility_table(df)
    df = df.merge(credibility[["case_id", "credibility_score", "credibility_tier"]], on="case_id", how="left")

    model_names = ["poly1", "poly2", "poly3", "poly4", "gpr_matern", "random_forest", "bootstrap_poly3"]
    metric_frames = []
    pred_frames = []
    for target in ["Cl_SU2", "Cd_SU2"]:
        m, p = evaluate_loocv(df, model_names, target)
        metric_frames.append(m)
        pred_frames.append(p)
        m, p = evaluate_high_alpha_holdout(df, model_names, target)
        metric_frames.append(m)
        pred_frames.append(p)

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    dense = dense_predictions(df, ["poly3", "gpr_matern", "random_forest", "bootstrap_poly3"])

    credibility.to_csv(RESULT_DIR / "phase4_credibility_dataset.csv", index=False)
    metrics_df.to_csv(RESULT_DIR / "phase4_model_metrics.csv", index=False)
    predictions.to_csv(RESULT_DIR / "phase4_model_predictions.csv", index=False)
    dense.to_csv(RESULT_DIR / "phase4_dense_predictions.csv", index=False)

    plot_surrogates(df, dense, "Cl_SU2", "Lift coefficient, Cl", FIG_DIR / "cl_surrogate_comparison")
    plot_surrogates(df, dense, "Cd_SU2", "Drag coefficient, Cd", FIG_DIR / "cd_surrogate_comparison")
    plot_holdout(predictions, "Cl_SU2", "Lift coefficient, Cl", FIG_DIR / "cl_high_alpha_holdout")
    plot_holdout(predictions, "Cd_SU2", "Drag coefficient, Cd", FIG_DIR / "cd_high_alpha_holdout")
    write_report(df, metrics_df, predictions, credibility)

    print(REPORT)
    print(RESULT_DIR / "phase4_model_metrics.csv")


if __name__ == "__main__":
    main()
