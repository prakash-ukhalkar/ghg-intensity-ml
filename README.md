# Predicting Corporate GHG Intensity

Bias-corrected machine learning framework for predicting corporate Scope 1
GHG intensity from SEC financial filings, using a Heckman two-stage
selection correction for non-random emissions-disclosure. Companion code
for the manuscript in [`docs/Main-Manuscript.docx`](docs/Main-Manuscript.docx),
targeting the *Journal of Environmental Economics and Policy*.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`requirements.txt` is a full pinned export of the development environment
(Jupyter/IPython tooling included). Core runtime dependencies: pandas,
numpy, scikit-learn, xgboost, lightgbm, torch, shap, statsmodels,
python-docx, pyxlsb, rapidfuzz, requests, pyyaml.

For running the test suite, also install `pytest` (not currently pinned in
`requirements.txt`):

```bash
pip install pytest
```

## Running the pipeline

`scripts/verify_pipeline.py` orchestrates the full pipeline end-to-end: data
ingestion → entity resolution → feature engineering → cross-validation →
final model training → statistical inference → robustness checks →
publication tables/figures. Run it from the project root.

```bash
# Simulation mode (default) — fast, synthetic data, for smoke-testing the pipeline
python scripts/verify_pipeline.py

# Real data — pulls from EPA GHGRP, SEC EDGAR, and World Bank (requires network,
# and will take significant wall-clock time due to SEC EDGAR rate limiting)
python scripts/verify_pipeline.py --real-data --n-control 800

# Skip inner-loop hyperparameter tuning (faster iteration)
python scripts/verify_pipeline.py --real-data --skip-tuning
```

`--n-control` sets the number of non-reporting "control" companies pulled
alongside EPA-matched reporters (used both as a comparison group and to
give the Heckman Stage-1 Probit selection variation to estimate from).

### Data sources (real-data mode)

- **EPA GHGRP** facility-level Scope 1 emissions (2010–2023) — bulk Excel
  workbooks + parent-company crosswalk, downloaded and cached under
  `data/raw/`.
- **SEC EDGAR** company index + XBRL Company Facts API for matched and
  control CIKs — cached under `data/raw/sec_*`. Respects EDGAR's
  rate-limit/User-Agent etiquette (`src/ingest_sec.py`).
- **World Bank** US macro controls (GDP growth, CO₂/capita, energy/capita).

Entity resolution (`src/entity_resolution.py`) fuzzy-matches EPA parent
company names to SEC filers (RapidFuzz token-set ratio ≥ 70) to build the
reporting/non-reporting panel.

### Leakage-free feature engineering

Winsorization bounds, sector z-score statistics, the Heckman Probit
selection model, and NaN-imputation medians are all **fit on the training
fold only**, then applied to the corresponding test fold — for both the
final holdout split and every fold of the expanding-window CV. See the
module docstring in `src/feature_engineering.py` and the `fit()`/
`transform()` pair; `ModelPipeline` in `src/models.py` drives this
per-split. `FeatureEngineer.create_features()` (fit over the *full* panel)
is reserved for descriptive statistics, correlation tables, and the
in-sample distress regression — never for out-of-sample ML evaluation.

### Beyond the core 8 models

`ModelPipeline` also evaluates:

- **Naive baselines** (`evaluate_naive_baselines`) — a sector-mean and a
  firm's-own-last-value predictor, so the paper can show the ML models
  beat a trivial forecaster, not just each other.
- **`HierarchicalRF`** (`train_hierarchical_rf`) — a sector-shrinkage
  blended Random Forest (`w = n_sector / (n_sector + k_shrink)`), aimed at
  the pooled model's poor fit in thin sectors.
- **`Evaluator.sector_variance_decomposition`** — a random-intercept
  MixedLM quantifying how much intensity variance is between- vs.
  within-sector (ICC), to check whether sector heterogeneity is
  structural or just thin-sample noise.
- **A second exclusion-restriction candidate** (`high_emission_naics_fine`
  in `src/ingest_epa.py`) plus an identification-strength check in
  `Evaluator.run_robustness_checks` comparing it against the main 2-digit
  NAICS flag, with and without sector fixed effects.

## Notebooks

`notebooks/01`–`09` walk through the same pipeline stages interactively, in
execution order: acquisition → linking → cleaning → EDA → feature
engineering → baseline models → advanced ML/DL → interpretability →
figures.

## Outputs

- `outputs/tables/` — CSV/txt result tables (model comparison, CV summary,
  bootstrap CIs, Diebold-Mariano tests, robustness checks, descriptive
  statistics, correlation matrix, distress regressions).
- `outputs/figures/` — publication-quality (300 DPI) PNG figures.
- `outputs/model_artifacts/` — trained model binaries (RF, XGBoost,
  LightGBM, PyTorch MLP).

## Tests

```bash
pytest -q
```

Covers Heckman IMR sign correctness, winsorization/z-score train-only
fitting (no test-fold leakage), sector z-score thin-cell fallback, temporal
split ordering, and expanding-window CV fold monotonicity.

## Project layout

```
src/                  Core pipeline package (ingestion, entity resolution,
                       feature engineering, models, evaluation, viz)
tests/                pytest suite
notebooks/             Interactive walkthrough (01-09)
data/{raw,interim,processed}/   Pipeline data at each stage
outputs/{tables,figures,model_artifacts}/   Pipeline results
docs/                 Manuscript, title page, cover letter, references
scripts/              verify_pipeline.py (orchestrator) and rerun_from_cache.py
etc/paper_config.yaml Journal submission metadata
```
