# Reproducibility Notes

## Software

- CFD solver used for the new simulations: SU2 v8.5.0 Windows OpenMP binary.
- ML/post-processing environment: Python with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `joblib`.

## CFD Dataset

The accepted coefficient dataset contains 12 NACA 0012 angle-of-attack cases from `-4.04 deg` to `14.22 deg`.

The public package includes processed coefficient data, selected small surface-output files, and the compact alpha-zero mesh-sensitivity log. Large volume solutions, restart files, private manuscripts, and paid-software raw outputs are not included.

## Mesh-Sensitivity Addendum

The package includes a limited alpha-zero two-grid sensitivity check using the official nested NASA/TMR `449 x 129` and `897 x 257` grids. It reports a conservative two-grid, assumed-order GCI-style drag uncertainty estimate; it is not a full observed-order three-grid GCI study. The summary is available in `docs/alpha0_mesh_sensitivity_report.md`, the compact calculation table is retained in `data/processed/mesh_sensitivity/alpha0_two_grid_gci_summary.csv`, and compact coarse-grid setup/log files are retained in `data/raw/mesh_sensitivity`.

## ML Reproduction

Run:

```bash
python scripts/run_phase4_ml_baselines.py
python scripts/build_phase4_manuscript_tables.py
```

Expected headline result:

- GPR Matern LOOCV RMSE for `Cl`: approximately `0.00185`.
- GPR Matern LOOCV RMSE for `Cd`: approximately `7.77e-05`.

Small numerical differences may occur because Gaussian-process optimization can be sensitive to package versions.
