# Open-source CFD+ML surrogate modeling for a validated NACA 0012 benchmark

This repository contains the public reproducibility package for the manuscript:

**Open-source, credibility-assessed surrogate modeling for validated airfoil CFD benchmarks**

The study uses a free/open-source workflow around SU2, Python, and sample-efficient machine-learning models to build and analyze a credibility-labeled NACA 0012 coefficient dataset at `Re = 6.0e6`.

## Repository Contents

- `data/processed`: accepted SU2 coefficient dataset, reference-comparison table, surface-validation tables, mesh-sensitivity summary, and machine-learning outputs.
- `figures`: manuscript figures and supporting figure panels.
- `manuscript_tables`: CSV and Markdown tables used in the manuscript.
- `scripts`: Python scripts used for ML baselines, manuscript tables, surface-validation post-processing, and workflow figure generation.
- `docs`: reproducibility notes, data dictionary, source notes, and local similarity precheck output.

## Reproduction

Create an environment with Python 3.10+ and install the required packages:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then rerun the ML analysis:

```bash
python scripts/run_phase4_ml_baselines.py
python scripts/build_phase4_manuscript_tables.py
```

The surface-validation script is included for transparency. It expects SU2 `surface_flow*.vtu` files and the official TMR reference files in the same relative structure used during manuscript preparation. The processed surface-validation CSVs needed for the paper are included in `data/processed/surface_validation`.

## Evidence Boundaries

This repository supports the manuscript's bounded claims:

- one airfoil: NACA 0012;
- one primary Reynolds-number setting: `Re = 6.0e6`;
- one-dimensional surrogate input: angle of attack;
- sample-efficient surrogate models, not a deep-learning performance claim;
- high-alpha drag is explicitly caveated.

The repository does not include full-text literature PDFs, private manuscripts, paid-software raw output files, large restart files, or local absolute workspace paths.

## Licenses

Code in this repository is released under the MIT License. Data, tables, and generated figures are released under CC BY 4.0 unless a third-party source states otherwise. Official reference data retain their original source terms; see `docs/source_notes.md`.
