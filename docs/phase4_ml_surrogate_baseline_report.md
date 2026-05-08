# Phase 4 ML Surrogate Baseline Report

Date: 2026-05-05

## Purpose

Train first-pass surrogate and uncertainty baselines on the accepted SU2 NACA 0012 Phase 3 dataset. This is a model-selection and feasibility checkpoint, not the final manuscript result.

## Dataset

- Rows: 12
- Alpha range: -4.04 deg to 14.22 deg
- Targets: `Cl_SU2`, `Cd_SU2`
- Inputs: angle of attack only
- High-alpha rows, `alpha >= 11 deg`: 4
- Rows with automatic caveat or diagnostic tier: 8

Credibility classes:

| case_id | alpha_deg | credibility_score | credibility_tier | credibility_reason |
| --- | --- | --- | --- | --- |
| alpha_m4p04 | -4.04 | 0.7 | usable_with_caveat | Cd drift over last 200 iterations exceeds 0.5%; Cd drift over last 500 iterations exceeds 1.0% |
| alpha_m2p14 | -2.14 | 0.7 | usable_with_caveat | Cd drift over last 200 iterations exceeds 0.5%; Cd drift over last 500 iterations exceeds 1.0% |
| alpha_0_baseline | 0 | 1 | core | baseline anchor without direct experimental percent comparison |
| alpha_p2p05 | 2.05 | 0.6 | diagnostic_only | Cd drift over last 200 iterations exceeds 0.5%; Cd drift over last 500 iterations exceeds 1.0%; Cl differs from experiment by more than 5% |
| alpha_p4p04 | 4.04 | 0.7 | usable_with_caveat | Cd drift over last 200 iterations exceeds 0.5%; Cd drift over last 500 iterations exceeds 1.0% |
| alpha_p6p09 | 6.09 | 1 | core | no major automatic credibility penalty |
| alpha_p8p30 | 8.3 | 0.85 | core | Cd drift over last 200 iterations exceeds 0.5% |
| alpha_p10p12 | 10.12 | 0.85 | core | Cd drift over last 200 iterations exceeds 0.5% |
| alpha_p11p13 | 11.13 | 0.7 | usable_with_caveat | Cd differs from experiment by more than 10%; high-alpha region flagged separately |
| alpha_p12p12 | 12.12 | 0.7 | usable_with_caveat | Cd differs from experiment by more than 10%; high-alpha region flagged separately |
| alpha_p13p08 | 13.08 | 0.55 | diagnostic_only | Cd drift over last 500 iterations exceeds 1.0%; Cd differs from experiment by more than 10%; high-alpha region flagged separately |
| alpha_p14p22 | 14.22 | 0.55 | diagnostic_only | Cd drift over last 500 iterations exceeds 1.0%; Cd differs from experiment by more than 10%; high-alpha region flagged separately |

## Models tested

- `poly1`, `poly2`, `poly3`, `poly4`: deterministic polynomial response surfaces.
- `gpr_matern`: Gaussian-process surrogate with a Matern kernel and predictive standard deviation.
- `random_forest`: tree-based baseline; expected to be weak for extrapolation.
- `bootstrap_poly3`: bootstrap ensemble of cubic response surfaces for an empirical uncertainty band.

No neural network was trained because 12 rows is too small for a defensible deep-learning claim.

## Leave-one-out validation

Best LOOCV model by target:

| target | model | n_test | mae | rmse | max_abs_error | r2 | mean_pred_std | picp_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cd_SU2 | gpr_matern | 12 | 6.62373e-05 | 7.77157e-05 | 0.000138661 | 0.99952 | 9.20326e-05 | 0.916667 |
| Cl_SU2 | gpr_matern | 12 | 0.000993832 | 0.00184796 | 0.00601213 | 0.999991 | 0.00160368 | 0.833333 |

Full metrics are stored in:

- `.\06_ml_pipeline\results\phase4_model_metrics.csv`

## High-alpha holdout test

This test trains only on cases with `alpha <= 10.12 deg` and predicts the held-out high-alpha cases above 10.12 deg. It is intentionally difficult and checks whether the model can extrapolate into the caveated nonlinear region.

Best high-alpha holdout model by target:

| target | model | n_test | mae | rmse | max_abs_error | r2 | mean_pred_std | picp_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cd_SU2 | bootstrap_poly3 | 4 | 0.000162742 | 0.000181681 | 0.000287934 | 0.991438 | 0.00237462 | 1 |
| Cl_SU2 | poly3 | 4 | 0.0038401 | 0.00541868 | 0.0100664 | 0.997357 |  |  |

Important interpretation:

- A model that performs well in LOOCV but poorly in high-alpha holdout should not be presented as robust outside its sampled regime.
- Tree-based models are included as a cautionary baseline because they do not extrapolate naturally in one-dimensional sparse data.
- Gaussian-process and bootstrap uncertainty bands are useful for figures, but their calibration remains weak with only 12 CFD points.

## Generated artifacts

- `.\06_ml_pipeline\results\phase4_model_metrics.csv`
- `.\06_ml_pipeline\results\phase4_model_predictions.csv`
- `.\06_ml_pipeline\results\phase4_dense_predictions.csv`
- `.\06_ml_pipeline\results\phase4_credibility_dataset.csv`
- `.\07_figures\phase4_ml\cl_surrogate_comparison.png`
- `.\07_figures\phase4_ml\cd_surrogate_comparison.png`
- `.\07_figures\phase4_ml\cl_high_alpha_holdout.png`
- `.\07_figures\phase4_ml\cd_high_alpha_holdout.png`

## Phase decision

The current dataset is sufficient for manuscript-grade Phase 4 surrogate figures and tables, including nonlinear drag-growth coverage through alpha = 14.22 deg. It is still not sufficient for a broad deep-learning or geometry-general surrogate claim. The next scientifically strongest move is to refine credibility-aware reporting, select final models, and prepare manuscript result tables. Do not run alpha = 15.26 deg unless the final model-selection review shows a specific gap that cannot be handled by the current caveated high-alpha data.
