# Reproducibility Notes

## Software

- CFD solver used for the new simulations: SU2 v8.5.0 Windows OpenMP binary.
- ML/post-processing environment: Python with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `joblib`.

## CFD Dataset

The accepted coefficient dataset contains 12 NACA 0012 angle-of-attack cases from `-4.04 deg` to `14.22 deg`.

The public package includes processed coefficient data and selected small surface-output files. Large volume solutions, restart files, full logs, private manuscripts, and paid-software raw outputs are not included.

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
