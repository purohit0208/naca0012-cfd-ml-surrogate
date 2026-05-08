# NACA 0012 Surface Cp/Cf Validation Addendum

Date: 2026-05-07

## Purpose

Strengthen Paper 1 by adding surface-level validation diagnostics for the SU2 NACA 0012 cases. This addendum uses existing SU2 `surface_flow*.vtu` files and official TMR reference data; no new CFD run was required.

## Official reference sources

- TMR NACA 0012 validation page: https://tmbwg.github.io/turbmodels/naca0012_val.html
- TMR SA model results page: https://tmbwg.github.io/turbmodels/naca0012_val_sa.html
- SU2 turbulent NACA 0012 tutorial: https://su2code.github.io/tutorials/Inc_Turbulent_NACA0012/

The TMR page lists surface `Cp` and `Cf` as quantities of interest at alpha = 0, 10, and 15 deg. It also states that no known experimental skin-friction data are available for this case, so `Cf` is compared to CFL3D only.

## Cases processed

| SU2 case | SU2 alpha | Reference alpha | Reason |
|---|---:|---:|---|
| alpha_0_baseline | 0.00 | 0.00 | Exact TMR alpha match. |
| alpha_p10p12 | 10.12 | 10.00 | Closest completed SU2 case to the TMR alpha = 10 surface reference. |

## Surface extraction diagnostics

- Alpha 0 surface points: 512
- Alpha 10.12 surface points: 512
- Alpha 0 maximum y+: 0.2694
- Alpha 10.12 maximum y+: 0.6778

## Error metrics

| quantity | surface | reference_source | reference_alpha_deg | n_reference | mae | rmse | max_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cp | upper | CFL3D_SA_TMR | 0 | 256 | 0.00405 | 0.004748 | 0.01534 |
| cp | lower | CFL3D_SA_TMR | 0 | 254 | 0.004 | 0.004655 | 0.01018 |
| cp | upper | Gregory_OReilly_exp_TMR | 0 | 24 | 0.03142 | 0.06293 | 0.237 |
| cf | upper | CFL3D_SA_TMR | 0 | 255 | 2.419e-05 | 7.869e-05 | 0.00102 |
| cp | upper | CFL3D_SA_TMR | 10 | 256 | 0.01896 | 0.03842 | 0.1301 |
| cp | lower | CFL3D_SA_TMR | 10 | 254 | 0.02988 | 0.05617 | 0.1423 |
| cp | upper | Gregory_OReilly_exp_TMR | 10 | 24 | 0.07324 | 0.1545 | 0.6354 |
| cf | upper | CFL3D_SA_TMR | 10 | 255 | 0.0002198 | 0.0003498 | 0.001385 |

## Generated files

- `.\03_validation_audit\surface_validation\su2_surface_profiles.csv`
- `.\03_validation_audit\surface_validation\tmr_cp_reference_long.csv`
- `.\03_validation_audit\surface_validation\tmr_cf_reference_long.csv`
- `.\03_validation_audit\surface_validation\surface_validation_metrics.csv`
- `.\07_figures\surface_validation`

## Manuscript interpretation

The surface diagnostics strengthen the CFD validation story because the paper no longer relies only on integrated force coefficients. The alpha = 0 comparison is the strongest surface validation because it is an exact angle match. The alpha = 10.12 case should be described as a near-alpha comparison against the TMR alpha = 10 reference, not as an exact match.

## Evidence boundary

- `Cf` is a CFD-to-CFD comparison against CFL3D; it is not experimental validation.
- The `alpha_p10p12` comparison has a 0.12 deg angle mismatch relative to the TMR alpha = 10 reference.
- The highest-alpha SU2 cases remain force-coefficient stress-test points; no alpha = 15 surface validation was added in this step.
