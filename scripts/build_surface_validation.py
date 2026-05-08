from __future__ import annotations

import csv
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "05_cfd_cases"
OUTDIR = ROOT / "03_validation_audit" / "surface_validation"
REFDIR = OUTDIR / "tmr_reference"
FIGDIR = ROOT / "07_figures" / "surface_validation"


@dataclass(frozen=True)
class SurfaceCase:
    case_id: str
    alpha_deg: float
    vtu_path: Path
    reference_alpha: float | None
    notes: str


TYPE_MAP = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int32": np.int32,
    "UInt8": np.uint8,
}


SURFACE_CASES = [
    SurfaceCase(
        "alpha_0_baseline",
        0.0,
        CASE_ROOT
        / "su2_naca0012_baseline"
        / "02_alpha0_restart_clean_exit"
        / "surface_flow.vtu",
        0.0,
        "Exact alpha match to TMR alpha=0 reference.",
    ),
    SurfaceCase(
        "alpha_p10p12",
        10.12,
        CASE_ROOT
        / "su2_naca0012_sweep"
        / "alpha_p10p12"
        / "surface_flow_alpha_p10p12_continuation.vtu",
        10.0,
        "Compared to TMR alpha=10 reference; SU2 case differs by +0.12 deg.",
    ),
]


def read_su2_vtu_appended(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = path.read_bytes()
    marker = data.index(b"<AppendedData")
    header_text = data[:marker].decode("utf-8", errors="ignore")
    raw_start = data.index(b"_", marker) + 1

    arrays = []
    pattern = re.compile(
        r'<DataArray\s+type="(?P<type>[^"]+)"\s+Name="(?P<name>[^"]*)"\s+'
        r'NumberOfComponents=\s*"?(?P<comps>\d+)"?\s+offset="(?P<offset>\d+)"\s+format="appended"\s*/>'
    )
    for match in pattern.finditer(header_text):
        name = match.group("name") or "Points"
        arrays.append(
            {
                "type": match.group("type"),
                "name": name,
                "components": int(match.group("comps")),
                "offset": int(match.group("offset")),
            }
        )

    parsed: dict[str, np.ndarray] = {}
    for item in arrays:
        dtype = TYPE_MAP[item["type"]]
        block_start = raw_start + item["offset"]
        block_size = struct.unpack("<Q", data[block_start : block_start + 8])[0]
        block = data[block_start + 8 : block_start + 8 + block_size]
        arr = np.frombuffer(block, dtype=dtype).copy()
        comps = item["components"]
        if comps > 1:
            arr = arr.reshape((-1, comps))
        parsed[item["name"]] = arr

    points = parsed.pop("Points")
    return points, parsed


def extract_su2_surface(case: SurfaceCase) -> pd.DataFrame:
    points, pdata = read_su2_vtu_appended(case.vtu_path)
    cp = pdata["Pressure_Coefficient"].astype(float)
    cf_vec = pdata["Skin_Friction_Coefficient"].astype(float)
    cf_mag = np.linalg.norm(cf_vec, axis=1)
    yplus = pdata.get("Y_Plus")
    yplus_arr = yplus.astype(float) if yplus is not None else np.full(len(points), np.nan)

    df = pd.DataFrame(
        {
            "case_id": case.case_id,
            "alpha_deg": case.alpha_deg,
            "reference_alpha_deg": case.reference_alpha,
            "x_over_c": points[:, 0],
            "y_over_c": points[:, 1],
            "cp": cp,
            "cf_x": cf_vec[:, 0],
            "cf_y": cf_vec[:, 1],
            "cf_z": cf_vec[:, 2],
            "cf_mag": cf_mag,
            "y_plus": yplus_arr,
            "surface": np.where(points[:, 1] >= 0.0, "upper", "lower"),
            "notes": case.notes,
        }
    )
    return df.sort_values(["case_id", "surface", "x_over_c"]).reset_index(drop=True)


def read_tecplot_zones(path: Path, value_name: str, source: str) -> pd.DataFrame:
    rows = []
    zone_name = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.lower().startswith("variables"):
            continue
        if line.lower().startswith("zone"):
            m = re.search(r't="([^"]+)"', line)
            zone_name = m.group(1) if m else line
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            value = float(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "source": source,
                "zone": zone_name,
                "x_over_c": x,
                value_name: value,
            }
        )
    return pd.DataFrame(rows)


def reference_alpha(zone: str) -> float | None:
    if not isinstance(zone, str):
        return None
    m = re.search(r"alpha\s*=\s*([+-]?\d+(?:\.\d+)?)", zone)
    return float(m.group(1)) if m else None


def reference_surface(source: str, zone: str, x_series: pd.Series) -> pd.Series:
    if source == "Gregory_OReilly_exp_TMR":
        return pd.Series(["upper"] * len(x_series), index=x_series.index)
    if isinstance(zone, str):
        z = zone.lower()
        if "upper" in z:
            return pd.Series(["upper"] * len(x_series), index=x_series.index)
        if "lower" in z:
            return pd.Series(["lower"] * len(x_series), index=x_series.index)
    values = x_series.to_numpy()
    if len(values) < 3:
        return pd.Series(["unknown"] * len(x_series), index=x_series.index)
    min_idx = int(np.argmin(values))
    labels = np.array(["lower"] * len(values), dtype=object)
    labels[min_idx + 1 :] = "upper"
    return pd.Series(labels, index=x_series.index)


def build_reference_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    cp_cfl3d = read_tecplot_zones(REFDIR / "n0012cp_cfl3d_sa.dat", "cp", "CFL3D_SA_TMR")
    cf_cfl3d = read_tecplot_zones(REFDIR / "n0012cf_cfl3d_sa.dat", "cf", "CFL3D_SA_TMR")
    cp_gregory = read_tecplot_zones(
        REFDIR / "CP_Gregory_expdata.dat", "cp", "Gregory_OReilly_exp_TMR"
    )
    cp_ladson = read_tecplot_zones(REFDIR / "CP_Ladson.dat", "cp", "Ladson_exp_TMR")

    cp_ref = pd.concat([cp_cfl3d, cp_gregory, cp_ladson], ignore_index=True)
    cf_ref = cf_cfl3d.copy()

    for df in [cp_ref, cf_ref]:
        df["alpha_deg"] = df["zone"].map(reference_alpha)
        surfaces = []
        for _, group in df.groupby(["source", "zone"], sort=False):
            surfaces.append(
                reference_surface(
                    str(group["source"].iloc[0]),
                    str(group["zone"].iloc[0]),
                    group["x_over_c"],
                )
            )
        df["surface"] = pd.concat(surfaces).sort_index()

    return cp_ref, cf_ref


def interp_error(
    su2: pd.DataFrame,
    ref: pd.DataFrame,
    su2_value_col: str,
    ref_value_col: str,
    quantity: str,
    surface: str,
    source: str,
    alpha: float,
) -> dict[str, float | str | int]:
    s = su2[(su2["surface"].eq(surface)) & (su2["reference_alpha_deg"].eq(alpha))].copy()
    r = ref[
        (ref["surface"].eq(surface))
        & (ref["source"].eq(source))
        & (ref["alpha_deg"].eq(alpha))
    ].copy()
    if len(s) < 4 or len(r) < 4:
        return {
            "quantity": quantity,
            "surface": surface,
            "reference_source": source,
            "reference_alpha_deg": alpha,
            "n_reference": len(r),
            "mae": np.nan,
            "rmse": np.nan,
            "max_abs_error": np.nan,
        }
    s = s.sort_values("x_over_c")
    r = r.sort_values("x_over_c")
    x_min = max(float(s["x_over_c"].min()), float(r["x_over_c"].min()))
    x_max = min(float(s["x_over_c"].max()), float(r["x_over_c"].max()))
    r = r[(r["x_over_c"] >= x_min) & (r["x_over_c"] <= x_max)].copy()
    pred = np.interp(
        r["x_over_c"].to_numpy(),
        s["x_over_c"].to_numpy(),
        s[su2_value_col].to_numpy(),
    )
    err = pred - r[ref_value_col].to_numpy()
    return {
        "quantity": quantity,
        "surface": surface,
        "reference_source": source,
        "reference_alpha_deg": alpha,
        "n_reference": len(r),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_cp(su2: pd.DataFrame, cp_ref: pd.DataFrame, alpha: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    s = su2[su2["reference_alpha_deg"].eq(alpha)]
    for surface, color in [("upper", "#1f77b4"), ("lower", "#d62728")]:
        part = s[s["surface"].eq(surface)].sort_values("x_over_c")
        ax.plot(part["x_over_c"], part["cp"], color=color, lw=1.8, label=f"SU2 {surface}")

    ref_styles = {
        "CFL3D_SA_TMR": ("-", "black"),
        "Gregory_OReilly_exp_TMR": ("o", "#2ca02c"),
    }
    for source, (style, color) in ref_styles.items():
        ref = cp_ref[(cp_ref["source"].eq(source)) & (cp_ref["alpha_deg"].eq(alpha))]
        for surface in ["upper", "lower"]:
            part = ref[ref["surface"].eq(surface)].sort_values("x_over_c")
            if part.empty:
                continue
            label = source.replace("_TMR", "").replace("_", " ")
            label = f"{label} {surface}" if source == "CFL3D_SA_TMR" else label
            if style == "o":
                ax.scatter(part["x_over_c"], part["cp"], s=16, color=color, alpha=0.75, label=label)
            else:
                ax.plot(part["x_over_c"], part["cp"], color=color, lw=1.0, alpha=0.75, label=label)

    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("Pressure coefficient, Cp")
    ax.set_title(f"NACA 0012 Cp comparison, alpha = {alpha:g} deg reference")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def plot_cf(su2: pd.DataFrame, cf_ref: pd.DataFrame, alpha: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    s = su2[(su2["reference_alpha_deg"].eq(alpha)) & (su2["surface"].eq("upper"))].sort_values(
        "x_over_c"
    )
    ax.plot(s["x_over_c"], s["cf_mag"], color="#1f77b4", lw=1.8, label="SU2 upper |Cf|")
    ref = cf_ref[
        (cf_ref["source"].eq("CFL3D_SA_TMR"))
        & (cf_ref["alpha_deg"].eq(alpha))
        & (cf_ref["surface"].eq("upper"))
    ].sort_values("x_over_c")
    ax.plot(ref["x_over_c"], ref["cf"], color="black", lw=1.0, label="CFL3D SA TMR upper Cf")
    ax.set_xlabel("x/c")
    ax.set_ylabel("Skin-friction coefficient")
    ax.set_title(f"NACA 0012 upper-surface Cf comparison, alpha = {alpha:g} deg reference")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def write_report(metrics: pd.DataFrame, su2: pd.DataFrame) -> None:
    alpha0 = su2[su2["case_id"].eq("alpha_0_baseline")]
    alpha10 = su2[su2["case_id"].eq("alpha_p10p12")]
    report = OUTDIR / "naca0012_surface_validation_report.md"
    lines = [
        "# NACA 0012 Surface Cp/Cf Validation Addendum",
        "",
        "Date: 2026-05-07",
        "",
        "## Purpose",
        "",
        "Strengthen Paper 1 by adding surface-level validation diagnostics for the SU2 NACA 0012 cases. "
        "This addendum uses existing SU2 `surface_flow*.vtu` files and official TMR reference data; no new CFD run was required.",
        "",
        "## Official reference sources",
        "",
        "- TMR NACA 0012 validation page: https://tmbwg.github.io/turbmodels/naca0012_val.html",
        "- TMR SA model results page: https://tmbwg.github.io/turbmodels/naca0012_val_sa.html",
        "- SU2 turbulent NACA 0012 tutorial: https://su2code.github.io/tutorials/Inc_Turbulent_NACA0012/",
        "",
        "The TMR page lists surface `Cp` and `Cf` as quantities of interest at alpha = 0, 10, and 15 deg. "
        "It also states that no known experimental skin-friction data are available for this case, so `Cf` is compared to CFL3D only.",
        "",
        "## Cases processed",
        "",
        "| SU2 case | SU2 alpha | Reference alpha | Reason |",
        "|---|---:|---:|---|",
        "| alpha_0_baseline | 0.00 | 0.00 | Exact TMR alpha match. |",
        "| alpha_p10p12 | 10.12 | 10.00 | Closest completed SU2 case to the TMR alpha = 10 surface reference. |",
        "",
        "## Surface extraction diagnostics",
        "",
        f"- Alpha 0 surface points: {len(alpha0)}",
        f"- Alpha 10.12 surface points: {len(alpha10)}",
        f"- Alpha 0 maximum y+: {alpha0['y_plus'].max():.4g}",
        f"- Alpha 10.12 maximum y+: {alpha10['y_plus'].max():.4g}",
        "",
        "## Error metrics",
        "",
        dataframe_to_markdown(metrics),
        "",
        "## Generated files",
        "",
        "- `./03_validation_audit\\surface_validation\\su2_surface_profiles.csv`",
        "- `./03_validation_audit\\surface_validation\\tmr_cp_reference_long.csv`",
        "- `./03_validation_audit\\surface_validation\\tmr_cf_reference_long.csv`",
        "- `./03_validation_audit\\surface_validation\\surface_validation_metrics.csv`",
        "- `./07_figures\\surface_validation`",
        "",
        "## Manuscript interpretation",
        "",
        "The surface diagnostics strengthen the CFD validation story because the paper no longer relies only on integrated force coefficients. "
        "The alpha = 0 comparison is the strongest surface validation because it is an exact angle match. "
        "The alpha = 10.12 case should be described as a near-alpha comparison against the TMR alpha = 10 reference, not as an exact match.",
        "",
        "## Evidence boundary",
        "",
        "- `Cf` is a CFD-to-CFD comparison against CFL3D; it is not experimental validation.",
        "- The `alpha_p10p12` comparison has a 0.12 deg angle mismatch relative to the TMR alpha = 10 reference.",
        "- The highest-alpha SU2 cases remain force-coefficient stress-test points; no alpha = 15 surface validation was added in this step.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    su2 = pd.concat([extract_su2_surface(case) for case in SURFACE_CASES], ignore_index=True)
    cp_ref, cf_ref = build_reference_tables()

    su2.to_csv(OUTDIR / "su2_surface_profiles.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    cp_ref.to_csv(OUTDIR / "tmr_cp_reference_long.csv", index=False)
    cf_ref.to_csv(OUTDIR / "tmr_cf_reference_long.csv", index=False)

    metrics = []
    for alpha in [0.0, 10.0]:
        for surface in ["upper", "lower"]:
            metrics.append(interp_error(su2, cp_ref, "cp", "cp", "cp", surface, "CFL3D_SA_TMR", alpha))
        metrics.append(
            interp_error(
                su2,
                cp_ref,
                "cp",
                "cp",
                "cp",
                "upper",
                "Gregory_OReilly_exp_TMR",
                alpha,
            )
        )
        metrics.append(
            interp_error(su2, cf_ref, "cf_mag", "cf", "cf", "upper", "CFL3D_SA_TMR", alpha)
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTDIR / "surface_validation_metrics.csv", index=False)

    plot_cp(su2, cp_ref, 0.0, FIGDIR / "naca0012_cp_alpha0_surface_validation")
    plot_cf(su2, cf_ref, 0.0, FIGDIR / "naca0012_cf_alpha0_surface_validation")
    plot_cp(su2, cp_ref, 10.0, FIGDIR / "naca0012_cp_alpha10_surface_validation")
    plot_cf(su2, cf_ref, 10.0, FIGDIR / "naca0012_cf_alpha10_surface_validation")

    write_report(metrics_df, su2)
    print(OUTDIR / "naca0012_surface_validation_report.md")
    print(FIGDIR)


if __name__ == "__main__":
    main()
