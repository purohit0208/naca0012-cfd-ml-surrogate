# Data Dictionary

## `su2_naca0012_phase3_accepted_dataset_public.csv`

- `case_id`: unique local case label.
- `alpha_deg`: angle of attack in degrees.
- `Cl_SU2`, `Cd_SU2`: accepted SU2 lift and drag coefficients.
- `rms_pressure_log10`, `rms_nu_tilde_log10`: final reported residual values.
- `continuation_used`: whether restart continuation was used.
- `Cl_*`, `Cd_*`: comparison values and percent differences where available.

## `phase4_model_metrics.csv`

Metrics for leave-one-out cross-validation and high-alpha holdout testing.

## `phase4_model_predictions.csv`

Per-case surrogate predictions for evaluated models.

## `surface_validation_metrics.csv`

Surface-level validation metrics comparing SU2 surface `Cp`/`Cf` profiles with official TMR/CFL3D reference data.

## `alpha0_mesh_sensitivity_summary.csv`

Two-grid alpha-zero sensitivity summary comparing the `449 x 129` and `897 x 257` nested NASA/TMR grids.
