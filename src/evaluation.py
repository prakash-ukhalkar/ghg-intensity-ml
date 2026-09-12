"""
Evaluation Module.

Publication-grade evaluation toolkit:

* Bootstrap confidence intervals for all metrics.
* Diebold-Mariano test for pairwise model comparison.
* Battery of robustness checks (winsorization sensitivity, alternative
  targets, sector subsamples, placebo test, leave-one-firm-out,
  with/without Heckman correction, Logit vs Probit selection, and a
  2-digit-vs-6-digit NAICS identification-strength diagnostic for the
  Heckman exclusion restriction).
* Descriptive statistics table (full / reporting / non-reporting panels).
* Variable definitions table.
* Zmijewski-style distress regression with firm-clustered standard errors.
* Sector variance decomposition (random-intercept MixedLM / ICC).
* Emissions-trajectory-volatility distress regression.
"""
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Evaluator:
    """Publication-grade evaluation and robustness testing."""

    def __init__(self, output_dir: str = "outputs/tables"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    # ================================================================== #
    #  Bootstrap confidence intervals                                     #
    # ================================================================== #
    def bootstrap_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_boot: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> dict[str, dict[str, float]]:
        """
        Computes bootstrapped confidence intervals for RMSE, MAE, R².

        Returns
        -------
        {metric_name: {"point": val, "ci_lo": val, "ci_hi": val, "se": val}}
        """
        rng = np.random.RandomState(seed)
        n = len(y_true)
        alpha = (1 - ci) / 2

        boot_rmse, boot_mae, boot_r2 = [], [], []
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            yt, yp = y_true[idx], y_pred[idx]
            boot_rmse.append(np.sqrt(mean_squared_error(yt, yp)))
            boot_mae.append(mean_absolute_error(yt, yp))
            if len(np.unique(yt)) > 1:
                boot_r2.append(r2_score(yt, yp))
            else:
                boot_r2.append(np.nan)

        def _summary(vals, point):
            arr = np.array([v for v in vals if not np.isnan(v)])
            if len(arr) == 0:
                return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan, "se": np.nan}
            return {
                "point": point,
                "ci_lo": float(np.percentile(arr, alpha * 100)),
                "ci_hi": float(np.percentile(arr, (1 - alpha) * 100)),
                "se": float(np.std(arr)),
            }

        return {
            "RMSE": _summary(boot_rmse, np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": _summary(boot_mae, mean_absolute_error(y_true, y_pred)),
            "R2": _summary(boot_r2, r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan),
        }

    # ================================================================== #
    #  Diebold-Mariano test                                               #
    # ================================================================== #
    @staticmethod
    def diebold_mariano(
        y_true: np.ndarray,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        h: int = 1,
    ) -> dict[str, float]:
        """
        Diebold-Mariano test for equal predictive accuracy (squared loss).

        Returns {"dm_stat": float, "p_value": float}.
        """
        e_a = (y_true - preds_a) ** 2
        e_b = (y_true - preds_b) ** 2
        d = e_a - e_b
        n = len(d)

        d_bar = np.mean(d)

        # Newey-West-like variance estimator
        gamma = [np.mean((d[i:] - d_bar) * (d[:-i or None] - d_bar))
                 for i in range(h)]
        var_d = (gamma[0] + 2 * sum(gamma[1:])) / n if len(gamma) > 0 else np.var(d) / n

        if var_d <= 0:
            return {"dm_stat": 0.0, "p_value": 1.0}

        dm = d_bar / np.sqrt(var_d)
        p = 2 * sp_stats.norm.sf(abs(dm))
        return {"dm_stat": float(dm), "p_value": float(p)}

    # ================================================================== #
    #  Pairwise DM test matrix                                            #
    # ================================================================== #
    def pairwise_dm_tests(
        self, y_true: np.ndarray, model_preds: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Runs DM tests between all model pairs.

        Returns a DataFrame with model_a, model_b, dm_stat, p_value.
        """
        names = list(model_preds.keys())
        rows = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                dm = self.diebold_mariano(y_true, model_preds[a], model_preds[b])
                rows.append({"model_a": a, "model_b": b, **dm})
        df = pd.DataFrame(rows)
        out = f"{self.output_dir}/diebold_mariano_tests.csv"
        df.to_csv(out, index=False)
        logger.info("DM tests saved → %s", out)
        return df

    # ================================================================== #
    #  Robustness checks                                                  #
    # ================================================================== #
    def run_robustness_checks(
        self,
        raw_df: pd.DataFrame,
        features: list[str],
        target: str = "scope1_intensity_rev",
        group_col: str = "cik",
    ) -> pd.DataFrame:
        """
        Runs a battery of robustness checks and returns summary results.

        Takes the RAW panel (output of ``FeatureEngineer.build_raw_features``,
        pre-winsorization/z-score/Heckman) so every check below fits its own
        ``FeatureEngineer`` on the training fold only — no test-fold
        statistics leak into training rows.

        Checks
        ------
        1. Winsorization sensitivity (1% vs 5%)
        2. Alternative target (log_scope1_intensity)
        3. Sector subsamples
        4. Placebo (shuffled emissions)
        5. Leave-one-firm-out CV
        6. With vs without Heckman correction (IMR ablation)
        7. Logit vs Probit selection-model specification
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge, LogisticRegression
        from src.feature_engineering import FeatureEngineer

        results: list[dict] = []

        # ---- Helper: temporal split + leakage-free feature fit + eval ----
        def _eval(df_sub_raw, tgt, label, winsor_limits=(0.01, 0.01), drop_imr=False):
            if len(df_sub_raw) < 15:
                return
            years = sorted(df_sub_raw["year"].unique())
            if len(years) < 3:
                return
            split_yr = years[-(len(years) // 3):]
            train_raw = df_sub_raw[~df_sub_raw["year"].isin(split_yr)]
            test_raw = df_sub_raw[df_sub_raw["year"].isin(split_yr)]
            if len(train_raw) < 10 or len(test_raw) < 3:
                return

            fe = FeatureEngineer(winsorize_limits=winsor_limits)
            try:
                fe.fit(train_raw)
            except Exception:
                return
            train_full = fe.transform(train_raw)
            test_full = fe.transform(test_raw)

            train = train_full[train_full["selected"] == 1].dropna(subset=[tgt])
            test = test_full[test_full["selected"] == 1].dropna(subset=[tgt])
            if len(train) < 10 or len(test) < 3:
                return

            feats_clean = [f for f in features if f in train.columns and train[f].notna().any()]
            if drop_imr:
                feats_clean = [f for f in feats_clean if f != "inverse_mills_ratio"]

            train = train.copy()
            test = test.copy()
            for f in feats_clean:
                med = train[f].median()
                train[f] = train[f].fillna(med)
                test[f] = test[f].fillna(med)

            X_tr, y_tr = train[feats_clean].values, train[tgt].values
            X_te, y_te = test[feats_clean].values, test[tgt].values

            for name, mdl in [("Ridge", Ridge(alpha=1.0)), ("RF", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42))]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mdl.fit(X_tr, y_tr)
                preds = mdl.predict(X_te)
                results.append({
                    "check": label,
                    "model": name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "rmse": np.sqrt(mean_squared_error(y_te, preds)),
                    "r2": r2_score(y_te, preds) if len(y_te) > 1 else np.nan,
                    "mae": mean_absolute_error(y_te, preds),
                })

        # 1. Baseline
        _eval(raw_df, target, "Baseline")

        # 2. Winsorization at 5%
        _eval(raw_df, target, "Winsorize 5%", winsor_limits=(0.05, 0.05))

        # 3. Alternative target: log intensity
        if "log_scope1_intensity" in raw_df.columns:
            _eval(raw_df, "log_scope1_intensity", "Log target")

        # 4. Sector subsamples
        for sector in raw_df["sector"].dropna().unique():
            sub = raw_df[raw_df["sector"] == sector]
            if len(sub[sub["selected"] == 1]) >= 20:
                _eval(sub, target, f"Sector: {sector}")

        # 5. Placebo test (shuffled target)
        df_placebo = raw_df.copy()
        df_placebo[target] = np.random.RandomState(99).permutation(df_placebo[target].values)
        _eval(df_placebo, target, "Placebo (shuffled)")

        # 6. Leave-one-firm-out CV (fits FeatureEngineer on the N-1 firms each time)
        reporting_all = raw_df[raw_df["selected"] == 1].dropna(subset=[target])
        if group_col in raw_df.columns:
            firms = reporting_all[group_col].unique()
            if 5 <= len(firms) <= 200:  # cap for runtime; sample larger universes
                lofo_r2 = []
                for firm in firms:
                    tr_raw = raw_df[raw_df[group_col] != firm]
                    te_raw = raw_df[raw_df[group_col] == firm]
                    if len(te_raw[te_raw["selected"] == 1]) < 1 or len(tr_raw) < 10:
                        continue
                    try:
                        fe = FeatureEngineer()
                        fe.fit(tr_raw)
                        tr_full = fe.transform(tr_raw)
                        te_full = fe.transform(te_raw)
                        tr = tr_full[tr_full["selected"] == 1].dropna(subset=[target])
                        te = te_full[te_full["selected"] == 1].dropna(subset=[target])
                        if len(te) < 1:
                            continue
                        feats_clean = [f for f in features if f in tr.columns and tr[f].notna().any()]
                        for f in feats_clean:
                            med = tr[f].median()
                            tr[f] = tr[f].fillna(med)
                            te[f] = te[f].fillna(med)
                        if len(te) <= 1:
                            continue
                        mdl = Ridge(alpha=1.0)
                        mdl.fit(tr[feats_clean].values, tr[target].values)
                        p = mdl.predict(te[feats_clean].values)
                        lofo_r2.append(r2_score(te[target].values, p))
                    except Exception:
                        pass
                valid = [v for v in lofo_r2 if not np.isnan(v)]
                if valid:
                    results.append({
                        "check": "Leave-one-firm-out",
                        "model": "Ridge",
                        "n_train": len(reporting_all),
                        "n_test": len(firms),
                        "rmse": np.nan,
                        "r2": float(np.mean(valid)),
                        "mae": np.nan,
                    })

        # 7. Without Heckman correction (IMR ablation)
        _eval(raw_df, target, "No IMR (ablation)", drop_imr=True)

        # 8. Logit vs Probit selection-model specification
        try:
            sel_features = ["size", "leverage", "roa"]
            if "high_emission_naics" in raw_df.columns:
                sel_features.append("high_emission_naics")
            df_sel = raw_df[sel_features + ["selected"]].copy()
            for c in sel_features:
                df_sel[c] = df_sel[c].fillna(df_sel[c].median())
            X, y = df_sel[sel_features].values, df_sel["selected"].values.astype(int)
            logit = LogisticRegression(max_iter=1000).fit(X, y)
            pred_logit = logit.predict_proba(X)[:, 1]

            import statsmodels.api as sm
            probit = sm.Probit(y, sm.add_constant(X)).fit(disp=0)
            pred_probit = probit.predict(sm.add_constant(X))

            corr = float(np.corrcoef(pred_logit, pred_probit)[0, 1])
            results.append({
                "check": "Logit vs Probit selection (predicted-prob correlation)",
                "model": "Logit/Probit",
                "n_train": len(df_sel), "n_test": len(df_sel),
                "rmse": np.nan, "r2": corr, "mae": np.nan,
            })
        except Exception as exc:
            logger.warning("Logit vs Probit check failed: %s", exc)

        # 9. Identification strength: broad (2-digit) vs. fine (6-digit)
        # exclusion restriction. The main Heckman fit (feature_engineering.py)
        # always uses the broad `high_emission_naics` flag; this check tests
        # whether a less-collinear, more granular instrument still predicts
        # selection after conditioning on sector fixed effects — i.e. that
        # the exclusion restriction isn't just standing in for `sector`.
        try:
            import statsmodels.api as sm

            sector_dum = pd.get_dummies(raw_df["sector"], prefix="sector", drop_first=True).astype(float) \
                if "sector" in raw_df.columns else pd.DataFrame(index=raw_df.index)

            for flag_col, label in [
                ("high_emission_naics", "2-digit NAICS (broad)"),
                ("high_emission_naics_fine", "6-digit NAICS (fine)"),
            ]:
                if flag_col not in raw_df.columns:
                    continue
                base_cols = ["size", "leverage", "roa"]
                df_id = raw_df[base_cols + [flag_col, "selected"]].copy()
                for c in base_cols:
                    df_id[c] = df_id[c].fillna(df_id[c].median())
                y = df_id["selected"].values.astype(int)

                # Without sector FE
                X_no_fe = sm.add_constant(df_id[base_cols + [flag_col]].astype(float))
                m_no_fe = sm.Probit(y, X_no_fe).fit(disp=0)
                p_no_fe = float(m_no_fe.pvalues[flag_col])

                # With sector FE — tests whether the flag adds information
                # beyond the broad sector categorical already in the model.
                X_fe = pd.concat([df_id[base_cols + [flag_col]].astype(float), sector_dum], axis=1)
                X_fe = sm.add_constant(X_fe)
                m_fe = sm.Probit(y, X_fe).fit(disp=0, maxiter=200)
                p_fe = float(m_fe.pvalues[flag_col])

                results.append({
                    "check": f"Identification strength: {label} — pseudo-R2 (no FE / with sector FE)",
                    "model": "Probit",
                    "n_train": len(df_id), "n_test": len(df_id),
                    "rmse": np.nan,
                    "r2": m_no_fe.prsquared,
                    "mae": m_fe.prsquared,
                })
                results.append({
                    "check": f"Identification strength: {label} — flag p-value (no FE / with sector FE)",
                    "model": "Probit",
                    "n_train": len(df_id), "n_test": len(df_id),
                    "rmse": np.nan,
                    "r2": p_no_fe,
                    "mae": p_fe,
                })
        except Exception as exc:
            logger.warning("Identification-strength check failed: %s", exc)

        df_rob = pd.DataFrame(results)
        out = f"{self.output_dir}/robustness_checks.csv"
        df_rob.to_csv(out, index=False)
        logger.info("Robustness checks (%d rows) saved → %s", len(df_rob), out)
        return df_rob

    # ================================================================== #
    #  Descriptive statistics table                                       #
    # ================================================================== #
    def descriptive_statistics(
        self, df: pd.DataFrame, target: str = "scope1_intensity_rev",
    ) -> pd.DataFrame:
        """
        Generates a descriptive statistics table:
        Panel A (full sample), Panel B (reporting), Panel C (non-reporting).
        """
        stat_cols = [
            "total_assets", "revenue", "net_income", "leverage",
            "roa", "operating_margin", "capex_intensity", "rd_intensity",
            target,
        ]
        stat_cols = [c for c in stat_cols if c in df.columns]

        def _stats(sub, label):
            desc = sub[stat_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
            desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
            desc["panel"] = label
            desc.index.name = "variable"
            return desc.reset_index()

        reporting = df[df["selected"] == 1]
        non_reporting = df[df["selected"] == 0]

        parts = [_stats(df, "A: Full Sample")]
        if len(reporting) > 0:
            parts.append(_stats(reporting, "B: Reporting"))
        if len(non_reporting) > 0:
            parts.append(_stats(non_reporting, "C: Non-Reporting"))

        desc_df = pd.concat(parts, ignore_index=True)
        out = f"{self.output_dir}/descriptive_statistics.csv"
        desc_df.to_csv(out, index=False)
        logger.info("Descriptive statistics saved → %s", out)
        return desc_df

    # ================================================================== #
    #  Variable definitions table                                         #
    # ================================================================== #
    def variable_definitions(self) -> pd.DataFrame:
        """Returns a publication-ready variable definitions table."""
        defs = [
            ("scope1_intensity_rev", "Scope 1 GHG emissions / revenue (metric tons CO₂e per $M)", "EPA GHGRP, SEC EDGAR"),
            ("log_scope1_intensity", "ln(1 + scope1_intensity_rev)", "Derived"),
            ("size", "ln(1 + total assets in $M)", "SEC EDGAR (Assets)"),
            ("leverage", "Total debt / total assets", "SEC EDGAR"),
            ("roa", "Net income / total assets", "SEC EDGAR"),
            ("operating_margin", "Operating income / revenue", "SEC EDGAR (OperatingIncomeLoss)"),
            ("capex_intensity", "Capital expenditure / total assets", "SEC EDGAR"),
            ("rd_intensity", "R&D expense / revenue", "SEC EDGAR"),
            ("revenue_growth", "Year-over-year revenue growth rate", "SEC EDGAR"),
            ("inverse_mills_ratio", "Heckman Stage-1 IMR from Probit selection model", "Derived (Heckman 1979)"),
            ("high_emission_naics", "=1 if firm's NAICS 2-digit code is in {21,22,31-33,48-49} (exclusion restriction)", "EPA GHGRP"),
            ("selected", "=1 if firm reports to EPA GHGRP in year t", "EPA GHGRP"),
            ("us_gdp_growth", "US real GDP growth rate (%)", "World Bank (NY.GDP.MKTP.KD.ZG)"),
            ("us_co2_per_capita", "US CO₂ emissions per capita (metric tons)", "World Bank (EN.ATM.CO2E.PC)"),
            ("us_energy_use_per_capita", "US energy use per capita (kg oil equivalent)", "World Bank (EG.USE.PCAP.KG.OE)"),
        ]
        df = pd.DataFrame(defs, columns=["Variable", "Definition", "Source"])
        out = f"{self.output_dir}/variable_definitions.csv"
        df.to_csv(out, index=False)
        logger.info("Variable definitions saved → %s", out)
        return df

    # ================================================================== #
    #  Distress trajectory regression                                     #
    # ================================================================== #
    def distress_regression(
        self,
        df: pd.DataFrame,
        target: str = "scope1_intensity_rev",
    ) -> str:
        """
        Zmijewski-style financial distress regression with
        firm-clustered standard errors.

        Returns OLS summary text.
        """
        import statsmodels.api as sm

        reporting = df[df["selected"] == 1].dropna(subset=[target]).copy()

        dep = target
        indep = ["leverage", "roa", "operating_margin", "size", "revenue_growth"]
        indep = [c for c in indep if c in reporting.columns]

        for col in indep:
            reporting[col] = reporting[col].fillna(reporting[col].median())

        X = sm.add_constant(reporting[indep])
        y = reporting[dep]

        # Identify cluster groups
        cluster_col = "cik" if "cik" in reporting.columns else "ticker"
        groups = reporting[cluster_col] if cluster_col in reporting.columns else None

        ols = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups} if groups is not None else {},
        )

        summary_txt = ols.summary().as_text()
        out = f"{self.output_dir}/distress_regression.txt"
        with open(out, "w") as f:
            f.write(summary_txt)
        logger.info("Distress regression saved → %s (N=%d, firms=%d)",
                     out, len(reporting),
                     reporting[cluster_col].nunique() if groups is not None else 0)
        return summary_txt

    # ================================================================== #
    #  Sector variance decomposition                                      #
    # ================================================================== #
    def sector_variance_decomposition(
        self,
        df: pd.DataFrame,
        target: str = "scope1_intensity_rev",
        sector_col: str = "sector",
    ) -> float | None:
        """
        Quantifies how much of the variation in ``target`` is between-sector
        vs. within-sector, via a random-intercept MixedLM (variance-
        components / intraclass-correlation decomposition).

        This turns the sector-heterogeneity finding from
        ``run_robustness_checks`` (strong fit in Manufacturing, poor fit in
        thin sectors) into a quantified statistic — the intraclass
        correlation (ICC) — rather than only a descriptive caveat, and
        motivates the sector-hierarchical shrinkage model
        (``models.HierarchicalRF``).

        Returns the ICC (share of variance between-sector), or None if the
        sample is too small to estimate.
        """
        import statsmodels.formula.api as smf

        reporting = df[df["selected"] == 1].dropna(subset=[target, sector_col]).copy()
        if reporting[sector_col].nunique() < 3 or len(reporting) < 30:
            logger.warning(
                "Sector variance decomposition skipped — insufficient data "
                "(N=%d, sectors=%d).", len(reporting), reporting[sector_col].nunique(),
            )
            return None

        md = smf.mixedlm(f"{target} ~ 1", reporting, groups=reporting[sector_col])
        mdf = md.fit(reml=True)

        group_var = float(mdf.cov_re.iloc[0, 0])
        resid_var = float(mdf.scale)
        icc = group_var / (group_var + resid_var) if (group_var + resid_var) > 0 else np.nan

        lines = [
            "Sector Variance Decomposition (Random-Intercept MixedLM)",
            "=" * 60,
            f"Target: {target}",
            f"N = {len(reporting)}, Sectors = {reporting[sector_col].nunique()}",
            f"Between-sector variance: {group_var:.6g}",
            f"Within-sector (residual) variance: {resid_var:.6g}",
            f"Intraclass correlation (share of variance between-sector): {icc:.4f}",
            "",
            mdf.summary().as_text(),
        ]
        out = f"{self.output_dir}/sector_variance_decomposition.txt"
        with open(out, "w") as f:
            f.write("\n".join(lines))
        logger.info("Sector variance decomposition saved → %s (ICC=%.4f)", out, icc)
        return icc

    # ================================================================== #
    #  Emissions-trajectory-volatility distress regression                #
    # ================================================================== #
    def trajectory_volatility_regression(
        self,
        df: pd.DataFrame,
        target: str = "scope1_intensity_rev",
        group_col: str = "cik",
    ) -> str:
        """
        Tests whether the VOLATILITY of a firm's emissions-intensity
        trajectory over time is associated with a proxy for financial
        distress, beyond the level effects already captured in
        ``distress_regression()``.

        Distress proxy: a Zmijewski (1984) -style score computed from ROA
        and leverage. NOTE: the classic Zmijewski score also includes a
        current-assets/current-liabilities liquidity term; that data is not
        available from EPA/SEC XBRL Company Facts for this sample, so the
        liquidity term is omitted here. This is a limitation, disclosed in
        the manuscript, not a full reproduction of Zmijewski (1984).

        Trajectory volatility: within-firm standard deviation of
        ``target`` across all reporting years (firms need >= 3 reporting
        years to get a non-degenerate estimate).

        IMPORTANT: ``leverage`` is deliberately excluded from this
        regression's independent variables, even though it is one of the
        two inputs used to construct ``zmijewski_score``. Including it as a
        regressor here would be a mechanical "bad control" — its
        coefficient would partly just reflect how the dependent variable
        was built, not an independent association. ROA is embedded in the
        score the same way and is likewise excluded.

        Runs three firm-clustered OLS specifications (baseline, year FE,
        sector FE), matching the manuscript's Table 6 layout.
        """
        import statsmodels.api as sm

        reporting = df[df["selected"] == 1].dropna(subset=[target]).copy()

        # Zmijewski-style distress proxy (liquidity term omitted — see docstring)
        for col in ["roa", "leverage"]:
            reporting[col] = reporting[col].fillna(reporting[col].median())
        reporting["zmijewski_score"] = (
            -4.336 - 4.513 * reporting["roa"] + 5.679 * reporting["leverage"]
        )

        # Per-firm trajectory volatility (needs >= 3 reporting years)
        vol = (
            reporting.groupby(group_col)[target]
            .agg(["std", "count"])
            .rename(columns={"std": "emissions_trajectory_volatility", "count": "n_years"})
        )
        vol = vol[vol["n_years"] >= 3]
        reporting = reporting.merge(vol[["emissions_trajectory_volatility"]], on=group_col, how="inner")

        if len(reporting) < 20 or reporting[group_col].nunique() < 5:
            logger.warning(
                "Trajectory volatility regression skipped — insufficient firms "
                "with >= 3 reporting years (n_firms=%d).",
                reporting[group_col].nunique(),
            )
            out = f"{self.output_dir}/trajectory_volatility_regression.txt"
            with open(out, "w") as f:
                f.write("Trajectory volatility regression skipped: insufficient "
                        "firms with >= 3 reporting years in this sample.\n")
            return ""

        # leverage/roa excluded deliberately — they are inputs to zmijewski_score
        # itself; including them here would be a mechanical bad control (see docstring).
        indep_base = ["emissions_trajectory_volatility", "size"]
        indep_base = [c for c in indep_base if c in reporting.columns]
        for col in indep_base:
            reporting[col] = reporting[col].fillna(reporting[col].median())

        groups = reporting[group_col]
        lines: list[str] = []
        lines.append("=" * 79)
        lines.append("Dependent Variable: Zmijewski-style Distress Proxy Score")
        lines.append("(roa/leverage omitted from regressors — liquidity term also")
        lines.append(" omitted from the proxy itself; see method notes)")
        lines.append("=" * 79)

        specs = [
            ("Model (1) Baseline", indep_base, False, False),
            ("Model (2) Year FE", indep_base, True, False),
            ("Model (3) Sector FE", indep_base, False, True),
        ]

        for label, indep, year_fe, sector_fe in specs:
            X_cols = list(indep)
            df_spec = reporting.copy()
            if year_fe:
                year_dummies = [c for c in df_spec.columns if c.startswith("year_")]
                X_cols += year_dummies
            if sector_fe and "sector" in df_spec.columns:
                sector_dum = pd.get_dummies(df_spec["sector"], prefix="sector", drop_first=True)
                df_spec = pd.concat([df_spec, sector_dum], axis=1)
                X_cols += list(sector_dum.columns)

            X = sm.add_constant(df_spec[X_cols].astype(float))
            y = df_spec["zmijewski_score"].astype(float)
            try:
                model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
                lines.append(f"\n--- {label} (N={int(model.nobs)}, R2={model.rsquared:.4f}) ---")
                lines.append(model.summary().as_text())
            except Exception as exc:
                lines.append(f"\n--- {label} FAILED: {exc} ---")

        summary_txt = "\n".join(lines)
        out = f"{self.output_dir}/trajectory_volatility_regression.txt"
        with open(out, "w") as f:
            f.write(summary_txt)
        logger.info(
            "Trajectory volatility regression saved → %s (N=%d, firms=%d)",
            out, len(reporting), reporting[group_col].nunique(),
        )
        return summary_txt

    # ================================================================== #
    #  Correlation matrix                                                 #
    # ================================================================== #
    def correlation_matrix(
        self, df: pd.DataFrame, target: str = "scope1_intensity_rev",
    ) -> pd.DataFrame:
        """Pearson correlation matrix with significance stars."""
        cols = [
            "size", "leverage", "roa", "operating_margin",
            "capex_intensity", "rd_intensity", "revenue_growth",
            target,
        ]
        cols = [c for c in cols if c in df.columns]
        reporting = df[df["selected"] == 1][cols].dropna()

        corr = reporting.corr()
        n = len(reporting)

        # Add significance stars
        def _star(r, n):
            if n < 3:
                return ""
            t = r * np.sqrt((n - 2) / (1 - r ** 2 + 1e-10))
            p = 2 * sp_stats.t.sf(abs(t), n - 2)
            if p < 0.01:
                return "***"
            elif p < 0.05:
                return "**"
            elif p < 0.10:
                return "*"
            return ""

        corr_str = corr.copy().astype(str)
        for i in corr.index:
            for j in corr.columns:
                if i != j:
                    r = corr.loc[i, j]
                    star = _star(r, n)
                    corr_str.loc[i, j] = f"{r:.3f}{star}"
                else:
                    corr_str.loc[i, j] = "1.000"

        out = f"{self.output_dir}/correlation_matrix.csv"
        corr_str.to_csv(out)
        logger.info("Correlation matrix saved → %s", out)
        return corr_str


if __name__ == "__main__":
    print("Evaluation module — import Evaluator.")
