from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "03_validation_audit" / "mesh_sensitivity"
COARSE_LOG = ROOT / "05_cfd_cases" / "mesh_sensitivity" / "alpha0_449x129" / "su2_alpha0_449x129_run.log"
FINE_LOG = ROOT / "05_cfd_cases" / "su2_naca0012_baseline" / "02_alpha0_restart_clean_exit" / "su2_alpha0_restart_run.log"


LINE_RE = re.compile(
    r"^\|\s*(\d+)\|\s*([-+0-9.Ee]+)\|\s*([-+0-9.Ee]+)\|\s*([-+0-9.Ee]+)\|\s*([-+0-9.Ee]+)\|\s*([-+0-9.Ee]+)\|\s*([-+0-9.Ee]+)\|"
)


def parse_su2_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LINE_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "iter": int(match.group(1)),
                "rms_p": float(match.group(4)),
                "rms_nu": float(match.group(5)),
                "cl": float(match.group(6)),
                "cd": float(match.group(7)),
            }
        )
    if not rows:
        raise ValueError(f"No SU2 iteration rows parsed from {path}")
    return rows


def summarize(label: str, grid: str, log_path: Path) -> dict[str, float | str]:
    rows = parse_su2_log(log_path)
    final = rows[-1]
    result: dict[str, float | str] = {
        "case": label,
        "grid": grid,
        "final_iter": final["iter"],
        "final_rms_p": final["rms_p"],
        "final_rms_nu": final["rms_nu"],
        "final_cl": final["cl"],
        "final_cd": final["cd"],
        "log_path": str(log_path),
    }
    for window in (200, 500, 1000):
        segment = rows[-window:]
        cd_delta = segment[-1]["cd"] - segment[0]["cd"]
        cl_delta = segment[-1]["cl"] - segment[0]["cl"]
        result[f"cl_delta_last_{window}"] = cl_delta
        result[f"cd_delta_last_{window}"] = cd_delta
        result[f"cd_pct_delta_last_{window}"] = 100.0 * cd_delta / segment[-1]["cd"]
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize("alpha0_coarse", "449x129", COARSE_LOG),
        summarize("alpha0_fine", "897x257", FINE_LOG),
    ]
    coarse, fine = rows
    cd_delta = float(coarse["final_cd"]) - float(fine["final_cd"])
    cl_delta = float(coarse["final_cl"]) - float(fine["final_cl"])
    cd_pct = 100.0 * cd_delta / float(fine["final_cd"])

    csv_path = OUT_DIR / "alpha0_mesh_sensitivity_summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = OUT_DIR / "alpha0_mesh_sensitivity_report.md"
    report.write_text(
        "\n".join(
            [
                "# Alpha-Zero Two-Grid Mesh-Sensitivity Check",
                "",
                "Date: 2026-05-18",
                "",
                "## Purpose",
                "",
                "This post-package technical addendum checks the alpha-zero coefficient sensitivity between the official nested NASA/TMR 449 x 129 and 897 x 257 NACA 0012 grids. It is a limited two-grid check, not a formal Richardson extrapolation or GCI study.",
                "",
                "## Inputs",
                "",
                f"- Coarse run log: `{COARSE_LOG}`",
                f"- Fine run log: `{FINE_LOG}`",
                "- Coarse grid source: official NASA/TMR `n0012_449-129.p2dfmt` and `n0012_449-129.nmf` from `NACA0012_grids.zip`.",
                "- Coarse grid conversion: `scripts/convert_tmr_plot3d_to_su2.py`.",
                "",
                "## Results",
                "",
                "| Grid | Final iter | log10 rms(P) | log10 rms(nu) | Cl | Cd | Cd drift last 500 iters |",
                "|---|---:|---:|---:|---:|---:|---:|",
                f"| 449 x 129 | {int(coarse['final_iter'])} | {float(coarse['final_rms_p']):.6f} | {float(coarse['final_rms_nu']):.6f} | {float(coarse['final_cl']):.6g} | {float(coarse['final_cd']):.6f} | {float(coarse['cd_pct_delta_last_500']):.3f}% |",
                f"| 897 x 257 | {int(fine['final_iter'])} | {float(fine['final_rms_p']):.6f} | {float(fine['final_rms_nu']):.6f} | {float(fine['final_cl']):.6g} | {float(fine['final_cd']):.6f} | {float(fine['cd_pct_delta_last_500']):.3f}% |",
                "",
                f"The 449 x 129 coarse-grid run gives `Cl = {float(coarse['final_cl']):.6g}` and `Cd = {float(coarse['final_cd']):.6f}` at alpha zero. The 897 x 257 accepted fine-grid value is `Cl = {float(fine['final_cl']):.6g}` and `Cd = {float(fine['final_cd']):.6f}`. The coarse-minus-fine drag difference is `{cd_delta:.6f}`, or `{cd_pct:.2f}%` of the fine-grid value; the lift difference is `{cl_delta:.6g}`.",
                "",
                "## Interpretation",
                "",
                "The alpha-zero lift remains essentially zero on both grids. The drag coefficient changes by about three percent between the two nested grids. This supports retaining a few-percent alpha-zero drag discretization/convergence sensitivity statement. Because only two grids are used and the coefficient variation is not interpreted through a formal asymptotic range, no GCI or Richardson-extrapolated value is claimed.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(report)
    print(csv_path)


if __name__ == "__main__":
    main()
