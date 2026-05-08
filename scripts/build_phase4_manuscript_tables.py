from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "su2_naca0012_phase3_accepted_dataset_public.csv"
COMPARISON = ROOT / "data" / "processed" / "su2_naca0012_phase3_reference_comparison_long.csv"
METRICS = ROOT / "data" / "processed" / "ml_results" / "phase4_model_metrics.csv"
PREDICTIONS = ROOT / "data" / "processed" / "ml_results" / "phase4_model_predictions.csv"
CREDIBILITY = ROOT / "data" / "processed" / "ml_results" / "phase4_credibility_dataset.csv"
OUT_DIR = ROOT / "reproduced" / "manuscript_tables"
PACK = ROOT / "reproduced" / "phase4_manuscript_results_pack.md"


def fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 1:
        return f"{number:.{digits}f}".rstrip("0").rstrip(".")
    if abs(number) >= 0.01:
        return f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return f"{number:.3e}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, stem: str, columns: list[str] | None = None) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = columns or list(df.columns)
    csv_path = OUT_DIR / f"{stem}.csv"
    md_path = OUT_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(df, cols), encoding="utf-8")
    return f"- `{csv_path}`\n- `{md_path}`"


def table_dataset_credibility(dataset: pd.DataFrame, credibility: pd.DataFrame) -> pd.DataFrame:
    df = dataset.merge(
        credibility[["case_id", "credibility_score", "credibility_tier"]],
        on="case_id",
        how="left",
    )
    out = df[
        [
            "case_id",
            "alpha_deg",
            "Cl_SU2",
            "Cd_SU2",
            "rms_pressure_log10",
            "continuation_used",
            "credibility_score",
            "credibility_tier",
        ]
    ].copy()
    out = out.rename(
        columns={
            "case_id": "Case",
            "alpha_deg": "Alpha_deg",
            "Cl_SU2": "Cl",
            "Cd_SU2": "Cd",
            "rms_pressure_log10": "log10_rms_pressure",
            "continuation_used": "Continuation",
            "credibility_score": "Credibility_score",
            "credibility_tier": "Credibility_tier",
        }
    )
    return out


def table_validation_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    df = comparison.copy()
    df["abs_percent_difference"] = pd.to_numeric(df["percent_difference"], errors="coerce").abs()
    df = df[df["abs_percent_difference"].notna()]
    grouped = (
        df.groupby(["coefficient", "reference_source"], as_index=False)
        .agg(
            n_cases=("abs_percent_difference", "count"),
            mean_abs_percent_difference=("abs_percent_difference", "mean"),
            max_abs_percent_difference=("abs_percent_difference", "max"),
        )
        .sort_values(["coefficient", "reference_source"])
    )
    grouped = grouped.rename(
        columns={
            "coefficient": "Coefficient",
            "reference_source": "Reference",
            "n_cases": "N",
            "mean_abs_percent_difference": "Mean_abs_percent_difference",
            "max_abs_percent_difference": "Max_abs_percent_difference",
        }
    )
    return grouped


def table_model_performance(metrics: pd.DataFrame, evaluation: str) -> pd.DataFrame:
    df = metrics[metrics["evaluation"].eq(evaluation)].copy()
    df["sort_target"] = df["target"].map({"Cl_SU2": 0, "Cd_SU2": 1})
    df["rmse_numeric"] = pd.to_numeric(df["rmse"], errors="coerce")
    df = df.sort_values(["sort_target", "rmse_numeric"])
    out = df[
        [
            "target",
            "model",
            "n_train",
            "n_test",
            "mae",
            "rmse",
            "max_abs_error",
            "r2",
            "mean_pred_std",
            "picp_95",
        ]
    ].copy()
    out = out.rename(
        columns={
            "target": "Target",
            "model": "Model",
            "n_train": "N_train",
            "n_test": "N_test",
            "mae": "MAE",
            "rmse": "RMSE",
            "max_abs_error": "Max_abs_error",
            "r2": "R2",
            "mean_pred_std": "Mean_pred_std",
            "picp_95": "PICP_95",
        }
    )
    return out


def table_selected_high_alpha_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keep = [
        ("Cl_SU2", "poly3"),
        ("Cd_SU2", "bootstrap_poly3"),
        ("Cl_SU2", "random_forest"),
        ("Cd_SU2", "random_forest"),
    ]
    frames = []
    high = predictions[predictions["evaluation"].eq("high_alpha_holdout_train_alpha_le_10p12")].copy()
    for target, model in keep:
        frames.append(high[high["target"].eq(target) & high["model"].eq(model)])
    out = pd.concat(frames, ignore_index=True)
    out = out[
        ["target", "model", "case_id", "alpha_deg", "actual", "predicted", "predicted_std", "error", "abs_error"]
    ].copy()
    out = out.rename(
        columns={
            "target": "Target",
            "model": "Model",
            "case_id": "Case",
            "alpha_deg": "Alpha_deg",
            "actual": "Actual",
            "predicted": "Predicted",
            "predicted_std": "Predicted_std",
            "error": "Error",
            "abs_error": "Abs_error",
        }
    )
    return out


def table_final_model_selection(metrics: pd.DataFrame) -> pd.DataFrame:
    loocv = metrics[metrics["evaluation"].eq("loocv")].copy()
    high = metrics[metrics["evaluation"].eq("high_alpha_holdout_train_alpha_le_10p12")].copy()

    def pick(evaluation_df: pd.DataFrame, target: str, model: str) -> pd.Series:
        match = evaluation_df[evaluation_df["target"].eq(target) & evaluation_df["model"].eq(model)]
        if match.empty:
            raise RuntimeError(f"Missing metric row for {target} / {model}")
        return match.iloc[0]

    rows = []
    selections = [
        {
            "Role": "Primary interpolation/UQ model",
            "Target": "Cl_SU2",
            "Model": "gpr_matern",
            "Metric_source": "LOOCV",
            "Rationale": "Lowest Cl LOOCV RMSE and provides predictive standard deviation.",
            "row": pick(loocv, "Cl_SU2", "gpr_matern"),
        },
        {
            "Role": "Primary interpolation/UQ model",
            "Target": "Cd_SU2",
            "Model": "gpr_matern",
            "Metric_source": "LOOCV",
            "Rationale": "Lowest Cd LOOCV RMSE and provides predictive standard deviation.",
            "row": pick(loocv, "Cd_SU2", "gpr_matern"),
        },
        {
            "Role": "High-alpha extrapolation stress test",
            "Target": "Cl_SU2",
            "Model": "poly3",
            "Metric_source": "train alpha <= 10.12, test alpha > 10.12",
            "Rationale": "Best high-alpha lift extrapolation among tested baselines.",
            "row": pick(high, "Cl_SU2", "poly3"),
        },
        {
            "Role": "High-alpha extrapolation stress test",
            "Target": "Cd_SU2",
            "Model": "bootstrap_poly3",
            "Metric_source": "train alpha <= 10.12, test alpha > 10.12",
            "Rationale": "Best high-alpha drag extrapolation among tested baselines; uncertainty is conservative but not fully calibrated.",
            "row": pick(high, "Cd_SU2", "bootstrap_poly3"),
        },
        {
            "Role": "Negative control",
            "Target": "Cl_SU2",
            "Model": "random_forest",
            "Metric_source": "train alpha <= 10.12, test alpha > 10.12",
            "Rationale": "Tree model fails to extrapolate and should be shown as a cautionary baseline.",
            "row": pick(high, "Cl_SU2", "random_forest"),
        },
        {
            "Role": "Negative control",
            "Target": "Cd_SU2",
            "Model": "random_forest",
            "Metric_source": "train alpha <= 10.12, test alpha > 10.12",
            "Rationale": "Tree model fails to extrapolate and should be shown as a cautionary baseline.",
            "row": pick(high, "Cd_SU2", "random_forest"),
        },
    ]
    for item in selections:
        metric = item.pop("row")
        rows.append(
            {
                **item,
                "MAE": metric["mae"],
                "RMSE": metric["rmse"],
                "Max_abs_error": metric["max_abs_error"],
                "R2": metric["r2"],
                "PICP_95": metric.get("picp_95", ""),
            }
        )
    return pd.DataFrame(rows)


def table_claim_boundaries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Claim": "Open-source SU2 workflow can reproduce a small validated NACA 0012 coefficient dataset.",
                "Supported": "Yes",
                "Evidence": "12 accepted SU2 cases, internal symmetry at +/-4.04 deg, comparison against local experiment/Fluent/TAU/CFL3D values.",
                "Boundary": "Only NACA 0012, Re = 6e6, low Mach, Spalart-Allmaras, current mesh/settings.",
            },
            {
                "Claim": "Gaussian-process surrogate is a strong interpolation baseline for Cl and Cd.",
                "Supported": "Yes",
                "Evidence": "Best LOOCV RMSE for both targets with predictive standard deviation.",
                "Boundary": "Small-data result; uncertainty calibration is approximate.",
            },
            {
                "Claim": "Model can extrapolate reliably beyond sampled high-alpha range.",
                "Supported": "No",
                "Evidence": "High-alpha holdout is a stress test only; random forest fails strongly and UQ calibration remains weak.",
                "Boundary": "Do not claim robust extrapolation beyond alpha = 14.22 deg.",
            },
            {
                "Claim": "Deep learning improves the surrogate.",
                "Supported": "No",
                "Evidence": "Only 12 CFD rows; no neural network trained.",
                "Boundary": "Mention AI/ML broadly, but do not present deep-learning performance claims for Paper 1.",
            },
            {
                "Claim": "High-alpha drag is fully validated.",
                "Supported": "No",
                "Evidence": "Cd at 14.22 deg is 17.42% above experiment, though closer to Fluent and TAU.",
                "Boundary": "Treat high-alpha points as caveated credibility/failure-boundary evidence.",
            },
        ]
    )


def write_pack(paths: dict[str, str], final_selection: pd.DataFrame) -> None:
    primary = final_selection[final_selection["Role"].eq("Primary interpolation/UQ model")]
    high = final_selection[final_selection["Role"].eq("High-alpha extrapolation stress test")]
    pack = f"""# Phase 4 Manuscript Results Pack

Date: 2026-05-06

## Purpose

Package the Phase 4 CFD+ML outputs into manuscript-ready tables and model-selection statements. This file intentionally avoids literature-review and reference content, following the deferred-literature protocol.

## Final model choice

Primary interpolation/UQ model:

{markdown_table(primary, ["Role", "Target", "Model", "Metric_source", "MAE", "RMSE", "Max_abs_error", "R2", "PICP_95", "Rationale"])}

High-alpha extrapolation stress-test models:

{markdown_table(high, ["Role", "Target", "Model", "Metric_source", "MAE", "RMSE", "Max_abs_error", "R2", "PICP_95", "Rationale"])}

## Main result statements

- Gaussian-process regression with a Matern kernel is the primary interpolation model for both `Cl` and `Cd`.
- LOOCV RMSE is `0.00185` for `Cl` and `7.77e-05` for `Cd`.
- High-alpha holdout confirms that a cubic response surface extrapolates lift more reliably than tree-based models in this sparse one-dimensional setting.
- Bootstrap cubic regression is the best tested drag extrapolation stress-test model, but its uncertainty is conservative and should not be overclaimed.
- Random forest is retained as a negative-control baseline because it fails to extrapolate beyond the sampled training range.
- No neural-network or deep-learning claim is scientifically justified with 12 CFD rows.
- High-alpha drag values, especially alpha = 14.22 deg, must be flagged as caveated validation evidence.

## Generated tables

{chr(10).join(paths.values())}

## Manuscript use

Recommended result-section structure:

1. Dataset credibility and CFD validation table.
2. Surrogate models and validation protocol.
3. LOOCV interpolation performance.
4. High-alpha holdout stress test.
5. Evidence boundaries and why no deep-learning model is claimed.

## Literature/references boundary

Do not fill related work or references from this pack. Those sections remain deferred until the user provides Consensus/pro-search outputs from authentic sources.
"""
    PACK.write_text(pack, encoding="utf-8")


def main() -> None:
    dataset = pd.read_csv(DATASET)
    comparison = pd.read_csv(COMPARISON)
    metrics = pd.read_csv(METRICS)
    predictions = pd.read_csv(PREDICTIONS)
    credibility = pd.read_csv(CREDIBILITY)

    tables = {
        "dataset": table_dataset_credibility(dataset, credibility),
        "validation": table_validation_summary(comparison),
        "loocv": table_model_performance(metrics, "loocv"),
        "high_alpha": table_model_performance(metrics, "high_alpha_holdout_train_alpha_le_10p12"),
        "high_alpha_predictions": table_selected_high_alpha_predictions(predictions),
        "final_selection": table_final_model_selection(metrics),
        "claim_boundaries": table_claim_boundaries(),
    }

    paths = {
        "dataset": save_table(tables["dataset"], "table_1_dataset_credibility"),
        "validation": save_table(tables["validation"], "table_2_validation_summary"),
        "loocv": save_table(tables["loocv"], "table_3_loocv_model_performance"),
        "high_alpha": save_table(tables["high_alpha"], "table_4_high_alpha_holdout_performance"),
        "high_alpha_predictions": save_table(tables["high_alpha_predictions"], "table_5_high_alpha_predictions"),
        "final_selection": save_table(tables["final_selection"], "table_6_final_model_selection"),
        "claim_boundaries": save_table(tables["claim_boundaries"], "table_7_claim_boundaries"),
    }
    write_pack(paths, tables["final_selection"])

    print(PACK)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
