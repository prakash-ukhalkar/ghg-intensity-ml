"""
Feature Engineering Module.

Constructs publication-ready features from the linked panel:

1. Financial ratios (ROA, leverage, operating margin, CAPEX intensity, R&D
   intensity, revenue growth, log size).
2. GHG intensity target (scope1_intensity_rev, log-transformed variant).
3. Heckman two-stage selection correction:
   - Stage 1: Probit on ``selected`` with the ``high_emission_naics``
     *exclusion restriction* (predicts selection but not intensity).
   - Stage 2: Inverse Mills Ratio (IMR) added as a regressor.
4. Sector-adjusted z-scores (with thin-cell fallback).
5. Winsorization at 1%/99% (configurable).

Leakage-safety
--------------
Winsorization bounds, sector z-score means/stds, Heckman Probit
coefficients, and imputation medians are all *fold-dependent statistics*.
``create_features()`` computes them over the full panel and is intended for
descriptive statistics, correlation tables, and the (in-sample) distress
regression — not for out-of-sample ML evaluation.

For predictive modelling, use the ``fit(train_df)`` / ``transform(df)`` pair:
fit on the training fold only, then transform both the training and test
folds with those frozen parameters. ``ModelPipeline`` (see ``models.py``)
does this internally for both the final holdout split and every
expanding-window CV fold.
"""
import os
import logging
import warnings
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RATIO_COLS = [
    "leverage", "roa", "operating_margin",
    "capex_intensity", "rd_intensity", "revenue_growth",
]
SECTOR_Z_COLS = ["leverage", "roa", "operating_margin", "capex_intensity"]
HECKMAN_SEL_FEATURES = ["size", "leverage", "roa"]


class FeatureEngineer:
    """Transforms the linked panel into model-ready features."""

    def __init__(
        self,
        output_dir: str = "data/processed",
        winsorize_limits: tuple[float, float] = (0.01, 0.01),
    ):
        self.output_dir = output_dir
        self.winsorize_limits = winsorize_limits
        os.makedirs(output_dir, exist_ok=True)

        # Fitted (fold-dependent) parameters — populated by fit()
        self._winsor_bounds: dict[str, tuple[float, float]] = {}
        self._sector_z_stats: dict[str, dict] = {}   # col -> {"pooled": (mu, sd), "sectors": {sector: (mu, sd)}}
        self._probit_model = None
        self._probit_encoder = None
        self._probit_medians: dict[str, float] = {}
        self._feature_medians: dict[str, float] = {}
        self._is_fitted = False

    # ================================================================== #
    #  Full-panel pipeline (descriptive tables / distress regression)     #
    # ================================================================== #
    def create_features(self, linked_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature-engineering pipeline fit and applied over the *entire*
        panel. Suitable for descriptive statistics, correlation tables, and
        the in-sample distress regression. NOT suitable for out-of-sample
        ML evaluation (use fit/transform instead — see module docstring).
        """
        df = self.build_raw_features(linked_df)
        logger.info("Feature engineering (full-panel) on %d rows …", len(df))

        self.fit(df)
        df = self.transform(df)

        out = os.path.join(self.output_dir, "processed_features.csv")
        df.to_csv(out, index=False)

        n_sel = int(df["selected"].sum())
        logger.info(
            "Features saved: %d rows (%d selected, %d non-selected), "
            "%d columns → %s",
            len(df), n_sel, len(df) - n_sel, len(df.columns), out,
        )
        return df

    # ================================================================== #
    #  Stage A — leakage-free per-row transforms                          #
    # ================================================================== #
    def build_raw_features(self, linked_df: pd.DataFrame) -> pd.DataFrame:
        """
        Row-wise transforms that do not depend on any fold's distribution:
        financial ratios, GHG intensity target, and year dummies.

        Safe to compute once on the full panel before splitting.
        """
        df = linked_df.copy()
        df = self._build_financial_ratios(df)
        df = self._build_ghg_intensity(df)
        df = self._add_year_dummies(df)
        return df

    # ================================================================== #
    #  Stage B — fold-dependent fit / transform                           #
    # ================================================================== #
    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fits winsorization bounds, sector z-score stats, the Heckman Probit
        selection model, and imputation medians using ONLY the rows in
        ``train_df``. Call ``transform()`` afterwards on both the training
        and test folds.
        """
        df = train_df

        # --- Winsorization bounds -----------------------------------------
        lo, hi = self.winsorize_limits
        self._winsor_bounds = {}
        for col in RATIO_COLS + ["scope1_intensity_rev"]:
            if col not in df.columns:
                continue
            valid = df[col].dropna()
            if len(valid) < 10:
                continue
            self._winsor_bounds[col] = (
                float(valid.quantile(lo)), float(valid.quantile(1 - hi)),
            )

        # --- Sector z-score stats ------------------------------------------
        self._sector_z_stats = {}
        min_cell_size = 3
        for col in SECTOR_Z_COLS:
            if col not in df.columns:
                continue
            pooled_mu, pooled_sd = df[col].mean(), df[col].std()
            sector_stats: dict = {}
            for sector, grp in df.groupby("sector"):
                valid = grp[col].dropna()
                if len(valid) >= min_cell_size:
                    sector_stats[sector] = (float(valid.mean()), float(valid.std()))
            self._sector_z_stats[col] = {
                "pooled": (float(pooled_mu), float(pooled_sd) if pd.notna(pooled_sd) else 0.0),
                "sectors": sector_stats,
            }

        # --- Heckman Probit (Stage 1) --------------------------------------
        self._fit_heckman(df)

        # --- Imputation medians (post financial-ratio construction) -------
        self._feature_medians = {}
        for col in RATIO_COLS + ["size"]:
            if col in df.columns:
                self._feature_medians[col] = float(df[col].median())

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies previously fitted winsorization bounds, sector z-scores,
        Heckman IMR, and median imputation to ``df``. Must call ``fit()``
        first (on the training fold only).
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer.transform() called before fit().")

        out = df.copy()

        # Impute raw ratios with TRAIN medians before winsorizing/z-scoring
        for col, med in self._feature_medians.items():
            if col in out.columns:
                out[col] = out[col].replace([np.inf, -np.inf], np.nan)
                out[col] = out[col].fillna(med)

        # Winsorize using TRAIN bounds
        for col, (lower, upper) in self._winsor_bounds.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lower, upper=upper)

        # Sector z-scores using TRAIN stats (fallback to pooled for unseen/thin sectors)
        for col, stats_dict in self._sector_z_stats.items():
            z_col = f"{col}_sector_z"
            pooled_mu, pooled_sd = stats_dict["pooled"]
            sector_map = stats_dict["sectors"]

            def _z(row, col=col, sector_map=sector_map, pooled_mu=pooled_mu, pooled_sd=pooled_sd):
                mu_sd = sector_map.get(row["sector"])
                if mu_sd is None:
                    mu, sd = pooled_mu, pooled_sd
                else:
                    mu, sd = mu_sd
                if sd is None or sd <= 1e-9 or pd.isna(sd):
                    return 0.0
                return (row[col] - mu) / sd

            out[z_col] = out.apply(_z, axis=1)

        # Heckman IMR using fitted Probit
        out = self._apply_heckman(out)

        return out

    # ================================================================== #
    #  Financial ratios / target / dummies (leakage-free)                 #
    # ================================================================== #
    def _build_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core financial ratios from SEC data."""
        df["size"] = np.log1p(df["total_assets"].clip(lower=1))
        df["leverage"] = df["total_debt"] / df["total_assets"].replace(0, np.nan)
        df["roa"] = df["net_income"] / df["total_assets"].replace(0, np.nan)
        df["operating_margin"] = (
            df["operating_income"] / df["revenue"].replace(0, np.nan)
        )
        df["capex_intensity"] = df["capex"] / df["total_assets"].replace(0, np.nan)
        df["rd_intensity"] = df["rd_expense"] / df["revenue"].replace(0, np.nan)

        # Revenue growth (year-over-year by firm)
        if "ticker" in df.columns and df["ticker"].notna().any():
            group_col = "ticker"
        else:
            group_col = "cik"
        df = df.sort_values([group_col, "year"])
        df["revenue_growth"] = df.groupby(group_col)["revenue"].pct_change()

        # Replace inf with NaN (median imputation happens later, per-fold)
        for col in RATIO_COLS:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        return df

    def _build_ghg_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """GHG intensity = scope1 / revenue (and log variant)."""
        rev_safe = df["revenue"].replace(0, np.nan)
        df["scope1_intensity_rev"] = df["scope1_emissions"] / rev_safe

        # Log-transformed target (for robustness check)
        df["log_scope1_intensity"] = np.log1p(
            df["scope1_intensity_rev"].clip(lower=0)
        )
        return df

    def _add_year_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary year indicators (dropping first to avoid collinearity)."""
        if "year" in df.columns:
            years = sorted(df["year"].unique())
            for yr in years[1:]:          # drop first year as reference
                df[f"year_{yr}"] = (df["year"] == yr).astype(int)
        return df

    # ================================================================== #
    #  Heckman selection correction                                       #
    # ================================================================== #
    def _fit_heckman(self, df: pd.DataFrame) -> None:
        """
        Fits the Stage-1 Probit selection model on the training fold only.

        selected ~ size + leverage + roa + sector + high_emission_naics
        (exclusion restriction).
        """
        from sklearn.preprocessing import LabelEncoder

        sel_features = list(HECKMAN_SEL_FEATURES)
        if "high_emission_naics" in df.columns:
            sel_features.append("high_emission_naics")

        df_probit = df[sel_features + ["selected", "sector"]].copy()

        le = LabelEncoder()
        df_probit["sector_code"] = le.fit_transform(
            df_probit["sector"].fillna("Unknown").astype(str)
        )

        medians = {}
        for col in sel_features:
            med = df_probit[col].median()
            medians[col] = med
            if df_probit[col].isna().any():
                df_probit[col] = df_probit[col].fillna(med)

        self._probit_medians = medians
        self._probit_encoder = le
        self._probit_sel_features = sel_features

        X = df_probit[sel_features + ["sector_code"]].values
        y = df_probit["selected"].values.astype(int)

        try:
            import statsmodels.api as sm

            X_const = sm.add_constant(X, has_constant="add")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probit = sm.Probit(y, X_const).fit(disp=0, maxiter=300)

            self._probit_model = probit
            logger.info(
                "Heckman Stage 1 (Probit, train-fold fit) — pseudo-R²: %.4f, "
                "N = %d, features = %s",
                probit.prsquared, len(y), sel_features + ["sector_code"],
            )
        except Exception as exc:
            logger.error("Probit fit failed — IMR will be set to 0: %s", exc)
            self._probit_model = None

    def _apply_heckman(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicts IMR for ``df`` using the Probit fitted in ``_fit_heckman``."""
        out = df.copy()

        if self._probit_model is None or self._probit_encoder is None:
            out["inverse_mills_ratio"] = 0.0
            out["probit_pred"] = 0.5
            return out

        sel_features = self._probit_sel_features
        df_probit = out[[c for c in sel_features if c != "high_emission_naics"] +
                         (["high_emission_naics"] if "high_emission_naics" in sel_features else []) +
                         ["sector"]].copy()

        for col in sel_features:
            med = self._probit_medians.get(col, 0.0)
            if df_probit[col].isna().any():
                df_probit[col] = df_probit[col].fillna(med)

        # Map unseen sector labels to a code the encoder has seen (fallback: most frequent train class)
        known_classes = set(self._probit_encoder.classes_)
        sector_vals = out["sector"].fillna("Unknown").astype(str)
        fallback = self._probit_encoder.classes_[0]
        sector_vals = sector_vals.apply(lambda s: s if s in known_classes else fallback)
        df_probit["sector_code"] = self._probit_encoder.transform(sector_vals)

        X = df_probit[sel_features + ["sector_code"]].values

        import statsmodels.api as sm
        X_const = sm.add_constant(X, has_constant="add")
        # add_constant with a single-row / constant-column edge case: ensure shape matches training
        if X_const.shape[1] != self._probit_model.params.shape[0]:
            X_const = np.column_stack([np.ones(len(X)), X])

        pred = self._probit_model.predict(X_const)
        pred = np.clip(pred, 1e-6, 1 - 1e-6)

        y = out["selected"].values.astype(int)
        xb = stats.norm.ppf(pred)
        pdf = stats.norm.pdf(xb)
        cdf = stats.norm.cdf(xb)

        imr = np.where(
            y == 1,
            pdf / np.clip(cdf, 1e-6, None),
            -pdf / np.clip(1 - cdf, 1e-6, None),
        )
        out["inverse_mills_ratio"] = imr
        out["probit_pred"] = pred
        return out


if __name__ == "__main__":
    print("Feature engineering module — import and call create_features(), or fit()/transform() for ML use.")
