from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "07_figures" / "manuscript_workflow"


def add_box(ax, x, y, w, h, text, facecolor, edgecolor="#243447"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    wrapped = "\n".join(textwrap.wrap(text, width=24))
    ax.text(
        x + w / 2,
        y + h / 2,
        wrapped,
        ha="center",
        va="center",
        fontsize=9,
        color="#18212b",
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="#314a5f",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.04, 0.62, "Official NACA 0012 SU2/NASA-TMR setup", "#d9edf7"),
        (0.28, 0.62, "Local SU2 sweep: 12 alpha cases from -4.04 to 14.22 deg", "#e2f0d9"),
        (0.52, 0.62, "Coefficient extraction and convergence-drift checks", "#fff2cc"),
        (0.76, 0.62, "Reference comparison: experiment, Fluent, TAU, NASA/TMR", "#fce4d6"),
        (0.04, 0.28, "Credibility labels: core, usable with caveat, diagnostic only", "#e4dfec"),
        (0.28, 0.28, "Surrogate models: GPR, polynomials, bootstrap, random forest", "#dae8fc"),
        (0.52, 0.28, "Validation: LOOCV and high-alpha holdout stress test", "#d5e8d4"),
        (0.76, 0.28, "Bounded claims, tables, figures, and manuscript draft", "#f8cecc"),
    ]

    w, h = 0.19, 0.16
    centers = []
    for x, y, text, color in boxes:
        add_box(ax, x, y, w, h, text, color)
        centers.append((x + w / 2, y + h / 2))

    # Top-row flow.
    top_y = 0.62
    bottom_y = 0.28
    add_arrow(ax, (0.04 + w, top_y + h / 2), (0.28, top_y + h / 2))
    add_arrow(ax, (0.28 + w, top_y + h / 2), (0.52, top_y + h / 2))
    add_arrow(ax, (0.52 + w, top_y + h / 2), (0.76, top_y + h / 2))

    # Downward transitions.
    add_arrow(ax, (0.76 + w / 2, top_y), (0.04 + w / 2, bottom_y + h))
    add_arrow(ax, (0.04 + w, bottom_y + h / 2), (0.28, bottom_y + h / 2))
    add_arrow(ax, (0.28 + w, bottom_y + h / 2), (0.52, bottom_y + h / 2))
    add_arrow(ax, (0.52 + w, bottom_y + h / 2), (0.76, bottom_y + h / 2))

    ax.text(
        0.5,
        0.92,
        "Credibility-assessed CFD to ML workflow",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#18212b",
    )
    ax.text(
        0.5,
        0.06,
        "All new simulations, processing, and model fitting use free/open-source tools.",
        ha="center",
        va="center",
        fontsize=9,
        color="#465a69",
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(OUTDIR / "paper1_cfd_ml_workflow.png", dpi=300)
    fig.savefig(OUTDIR / "paper1_cfd_ml_workflow.svg")
    plt.close(fig)

    print(OUTDIR / "paper1_cfd_ml_workflow.png")
    print(OUTDIR / "paper1_cfd_ml_workflow.svg")


if __name__ == "__main__":
    main()
