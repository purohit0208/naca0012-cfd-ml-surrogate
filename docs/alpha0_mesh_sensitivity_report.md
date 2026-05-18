# Alpha-Zero Two-Grid Mesh-Sensitivity Check

Date: 2026-05-18

## Purpose

This post-package technical addendum checks the alpha-zero coefficient sensitivity between the official nested NASA/TMR 449 x 129 and 897 x 257 NACA 0012 grids. It is a limited two-grid check, not a formal Richardson extrapolation or GCI study.

## Inputs

- Coarse run log: `.\05_cfd_cases\mesh_sensitivity\alpha0_449x129\su2_alpha0_449x129_run.log`
- Fine run log: `.\05_cfd_cases\su2_naca0012_baseline\02_alpha0_restart_clean_exit\su2_alpha0_restart_run.log`
- Coarse grid source: official NASA/TMR `n0012_449-129.p2dfmt` and `n0012_449-129.nmf` from `NACA0012_grids.zip`.
- Coarse grid conversion: `scripts/convert_tmr_plot3d_to_su2.py`.

## Results

| Grid | Final iter | log10 rms(P) | log10 rms(nu) | Cl | Cd | Cd drift last 500 iters |
|---|---:|---:|---:|---:|---:|---:|
| 449 x 129 | 7999 | -9.111585 | -8.839794 | -3e-06 | 0.008188 | -0.110% |
| 897 x 257 | 1999 | -9.546991 | -9.503154 | -3e-06 | 0.008448 | -0.118% |

The 449 x 129 coarse-grid run gives `Cl = -3e-06` and `Cd = 0.008188` at alpha zero. The 897 x 257 accepted fine-grid value is `Cl = -3e-06` and `Cd = 0.008448`. The coarse-minus-fine drag difference is `-0.000260`, or `-3.08%` of the fine-grid value; the lift difference is `0`.

## Interpretation

The alpha-zero lift remains essentially zero on both grids. The drag coefficient changes by about three percent between the two nested grids. This supports retaining a few-percent alpha-zero drag discretization/convergence sensitivity statement. Because only two grids are used and the coefficient variation is not interpreted through a formal asymptotic range, no GCI or Richardson-extrapolated value is claimed.

