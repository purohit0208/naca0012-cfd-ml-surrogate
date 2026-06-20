#!/usr/bin/env python
"""Cross-solver uncertainty floor and stall-region extrapolation test (manuscript Section 3.7).

Reproduces, from the released data only, the two strengthening analyses:
  (A) cross-solver / model-form disagreement among Exp, Fluent, DLR-TAU and CFL3D across the
      attached-flow angles, contrasted with the GP surrogate's leave-one-out interpolation error;
  (B) a cubic surrogate trained on the SU2 lift for alpha <= 14.22 deg, extrapolated against the
      held-out experimental polar through stall (15-19 deg), with a figure.

Inputs (data/processed/):
  - su2_naca0012_phase3_reference_comparison_long.csv  (SU2 values + references, alpha <= 14.22)
  - naca0012_experimental_polar_to_19deg_public.csv    (Exp/Fluent/TAU/CFL3D polar to 19 deg)
Output:
  - figures/manuscript_combined/cross_solver_stall_lift_polar.png

The reference/experimental coefficients are pre-existing comparison values (the author's prior
NACA 0012 study and public Ladson / NASA-TMR data); they are used here as comparison context only.
No new CFD is run.
"""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "processed" / "su2_naca0012_phase3_reference_comparison_long.csv"
POLAR = ROOT / "data" / "processed" / "naca0012_experimental_polar_to_19deg_public.csv"
FIGDIR = ROOT / "figures" / "manuscript_combined"


def load_polar():
    by = {}  # (alpha, coef) -> {source: value}
    for r in csv.DictReader(open(POLAR)):
        a = round(float(r["AoA_deg"]), 2)
        by.setdefault((a, r["coefficient"]), {})[r["source"]] = float(r["value"])
    return by


def load_su2_cl():
    pts = {}
    for r in csv.DictReader(open(REF)):
        if r["coefficient"] == "Cl":
            pts[round(float(r["alpha_deg"]), 2)] = float(r["su2_value"])
    return pts


def cross_solver_floor(polar):
    print("(A) Cross-solver / model-form disagreement (attached flow, alpha <= 14 deg)")
    for coef in ("Cd", "Cl"):
        rels = []
        for (a, c), d in polar.items():
            if c != coef or a > 14.5:
                continue
            vals = {k: v for k, v in d.items() if k in ("Exp", "Fluent", "TAU", "CFL3D")}
            thresh = 0.05 if coef == "Cl" else 0.0  # near-zero guard applies to lift only
            if len(vals) >= 2 and "Exp" in vals and abs(vals["Exp"]) > thresh:
                rels.append((max(vals.values()) - min(vals.values())) / abs(vals["Exp"]))
        if rels:
            print(f"    {coef}: mean spread {100*sum(rels)/len(rels):.1f}% of Exp, max {100*max(rels):.1f}%")
    print("    GP surrogate leave-one-out Cd error ~ 0.7% of mean Cd (see ml_results): an order of")
    print("    magnitude smaller than the cross-solver floor -> solver credibility is the binding uncertainty.")


def polyfit3(x, y):
    n = len(x)
    A = [[xi**j for j in range(4)] for xi in x]
    ATA = [[sum(A[r][i]*A[r][j] for r in range(n)) for j in range(4)] for i in range(4)]
    ATy = [sum(A[r][i]*y[r] for r in range(n)) for i in range(4)]
    M = [row[:] + [ATy[i]] for i, row in enumerate(ATA)]
    for col in range(4):
        piv = max(range(col, 4), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(4):
            if r != col:
                f = M[r][col] / M[col][col]
                M[r] = [M[r][k] - f*M[col][k] for k in range(5)]
    return [M[i][4]/M[i][i] for i in range(4)]


def stall_test(polar, su2):
    xs = sorted(a for a in su2 if a <= 14.22)
    coef = polyfit3(xs, [su2[a] for a in xs])
    pred = lambda a: sum(coef[j]*a**j for j in range(4))
    print("\n(B) Stall-region extrapolation (cubic fit to SU2 Cl, alpha <= 14.22, vs experiment)")
    print(f"    {'alpha':>7}{'surrogate':>11}{'experiment':>12}{'error':>9}")
    for a in [15.26, 16.30, 17.13, 18.02, 19.08]:
        e = polar.get((a, "Cl"), {}).get("Exp")
        if e is not None:
            print(f"    {a:>7.2f}{pred(a):>11.3f}{e:>12.3f}{pred(a)-e:>9.3f}")
    return coef, pred


def make_figure(polar, su2, pred):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"\n[figure skipped: {exc}]")
        return
    ea = sorted(a for (a, c) in polar if c == "Cl" and "Exp" in polar[(a, c)])
    ev = [polar[(a, "Cl")]["Exp"] for a in ea]
    sx = sorted(a for a in su2 if a <= 14.22)
    sy = [su2[a] for a in sx]
    xs = np.linspace(-4.04, 19.08, 250)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ea, ev, "k-o", ms=4, lw=1.3, label="Experiment (Ladson)")
    ax.plot(sx, sy, "bs", ms=5, label="SU2 (training, alpha<=14.22)")
    ax.plot(xs, [pred(x) for x in xs], "r--", lw=1.6, label="Cubic surrogate (fit to alpha<=14.22)")
    ax.axvspan(17.13, 19.2, color="orange", alpha=0.15)
    ax.axvline(14.22, color="gray", ls=":", lw=1)
    ax.annotate("stall", xy=(18.0, 1.0), xytext=(18.0, 1.45), ha="center", arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("Angle of attack (deg)")
    ax.set_ylabel("Lift coefficient Cl")
    ax.set_title("NACA 0012, Re 6e6: surrogate extrapolation vs experiment")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "cross_solver_stall_lift_polar.png"
    fig.savefig(out, dpi=300)
    print(f"\nWrote {out}")


def main():
    polar = load_polar()
    su2 = load_su2_cl()
    cross_solver_floor(polar)
    _, pred = stall_test(polar, su2)
    make_figure(polar, su2, pred)


if __name__ == "__main__":
    main()
