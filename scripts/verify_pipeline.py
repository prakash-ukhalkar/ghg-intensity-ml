"""
Full Research Pipeline — Q1 Publication Version.

Orchestrates the complete workflow:

1. Data ingestion     (EPA GHGRP, SEC EDGAR, World Bank)
2. Entity resolution  (full-universe fuzzy matching)
3. Dataset linking    (reporting + control panel)
4. Feature engineering (Heckman correction, winsorization, exclusion restriction)
5. Expanding-window temporal cross-validation
6. Final model training with hyperparameter tuning
7. Bootstrap confidence intervals + Diebold-Mariano tests
8. SHAP feature importance
9. Robustness checks battery
10. Descriptive statistics + variable definitions
11. Publication-quality figures

Usage (run from the project root)
-----
    python scripts/verify_pipeline.py                   # simulation mode
    python scripts/verify_pipeline.py --real-data       # real EPA/SEC/WB data
"""
import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd

# Add project root to path (this file lives in scripts/, one level below root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest_epa import EPAIngester
from src.ingest_sec import SECIngester
from src.ingest_worldbank import WorldBankIngester
from src.entity_resolution import EntityResolver
from src.feature_engineering import FeatureEngineer
from src.models import ModelPipeline
from src.evaluation import Evaluator
from src.viz import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")

YEARS = list(range(2010, 2024))


def parse_args():
    parser = argparse.ArgumentParser(description="GHG Intensity Prediction Pipeline")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real EPA/SEC/World Bank data (requires network).")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip hyperparameter tuning (faster).")
    parser.add_argument("--n-control", type=int, default=500,
                        help="Number of non-reporting control companies.")
    return parser.parse_args()


def main():
    args = parse_args()
    simulate = not args.real_data
    t0 = time.time()

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # ================================================================ #
    #  STEP 1 — Data Ingestion                                          #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 1: Data Ingestion")
    logger.info("=" * 70)

    epa = EPAIngester()
    sec = SECIngester()
    wb = WorldBankIngester()

    epa_df = epa.fetch_epa_emissions(years=YEARS, force_simulate=simulate)
    logger.info("EPA: %d facility records, %d unique parents",
                len(epa_df), epa_df["reported_parent"].nunique())

    # SEC company index (for entity resolution matching)
    if simulate:
        sec_df = sec.fetch_sec_financials(years=YEARS, force_simulate=True)
        sec_index = sec_df[["cik", "company_name"]].drop_duplicates(subset=["cik"])
        logger.info("SEC (simulated): %d companies", sec_index["cik"].nunique())
    else:
        sec_index = sec.download_company_index()
        logger.info("SEC index: %d companies", len(sec_index))

    wb_df = wb.fetch_macro_controls(years=YEARS, force_simulate=simulate)
    logger.info("World Bank: %d year-records", len(wb_df))

    # ================================================================ #
    #  STEP 2 — Entity Resolution                                       #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 2: Entity Resolution")
    logger.info("=" * 70)

    resolver = EntityResolver()
    mapping_df = resolver.resolve_entities(epa_df, sec_index, fuzzy_threshold=70)

    n_resolved = int(mapping_df["is_resolved"].sum())
    logger.info("Resolved: %d / %d EPA parents (%.1f%%)",
                n_resolved, len(mapping_df),
                100 * n_resolved / max(1, len(mapping_df)))

    # ================================================================ #
    #  STEP 3 — SEC financials for matched + control                     #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 3: SEC Financials Pull")
    logger.info("=" * 70)

    matched_ciks = mapping_df[mapping_df["is_resolved"]]["cik"].unique().tolist()

    if not simulate:
        # For real data, pull financials for matched + sampled control
        all_sec_ciks = sec_index["cik"].unique()
        non_matched = [c for c in all_sec_ciks if c not in set(matched_ciks)]
        np.random.seed(42)
        n_ctrl = min(args.n_control, len(non_matched))
        control_ciks = list(np.random.choice(non_matched, n_ctrl, replace=False))
        target_ciks = matched_ciks + control_ciks
        logger.info("Pulling SEC financials for %d matched + %d control = %d CIKs",
                     len(matched_ciks), len(control_ciks), len(target_ciks))
        sec_df = sec.fetch_sec_financials(ciks=target_ciks, years=YEARS)
    else:
        # In simulation, SEC data was already generated
        control_ciks = [
            c for c in sec_df["cik"].unique() if c not in set(matched_ciks)
        ]

    logger.info("SEC financials: %d firm-years, %d unique companies",
                len(sec_df), sec_df["cik"].nunique())

    # ================================================================ #
    #  STEP 4 — Link Datasets                                            #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 4: Dataset Linking")
    logger.info("=" * 70)

    linked_df = resolver.link_datasets(
        epa_df, sec_df, mapping_df, wb_df,
        control_ciks=control_ciks,
    )
    logger.info("Linked panel: %d firm-years (%d reporting, %d non-reporting)",
                len(linked_df),
                int(linked_df["selected"].sum()),
                int((linked_df["selected"] == 0).sum()))

    # ================================================================ #
    #  STEP 5 — Feature Engineering                                      #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 5: Feature Engineering")
    logger.info("=" * 70)

    fe = FeatureEngineer()
    processed_df = fe.create_features(linked_df)
    logger.info("Processed features (full-panel, descriptive use): %d rows × %d cols", *processed_df.shape)

    # Leakage-free raw features for the ML pipeline: ratios/target/year
    # dummies only — winsorization, sector z-scores, and the Heckman Probit
    # are fit per-split/per-fold inside ModelPipeline (see models.py).
    raw_df = fe.build_raw_features(linked_df)

    # ================================================================ #
    #  STEP 6 — Expanding-Window CV                                      #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 6: Expanding-Window Cross-Validation")
    logger.info("=" * 70)

    pipeline = ModelPipeline(target="scope1_intensity_rev", group_col="cik")
    cv_df = pipeline.expanding_window_cv(raw_df, min_train_years=3)

    # CV summary
    if not cv_df.empty:
        cv_summary = cv_df.groupby("model").agg(
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            n_folds=("fold", "nunique"),
        ).round(4)
        cv_summary.to_csv("outputs/tables/cv_summary.csv")
        logger.info("CV Summary:\n%s", cv_summary.to_string())

    # ================================================================ #
    #  STEP 7 — Final Model Training                                     #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 7: Final Model Training")
    logger.info("=" * 70)

    train_df, test_df, features = pipeline.prepare_train_test(raw_df)
    results_df = pipeline.train_all_models(
        train_df, test_df, features,
        tune=not args.skip_tuning,
    )

    # Naive baselines — establishes whether the ML models beat a trivial
    # forecaster, not just each other.
    baseline_df = pipeline.evaluate_naive_baselines(train_df, test_df)

    # Sector-hierarchical shrinkage RF — addresses the pooled model's poor
    # fit in thin sectors (Financials/Mining/Wholesale) by blending
    # per-sector and pooled RF predictions.
    hrf_row = pipeline.train_hierarchical_rf(train_df, test_df, features, k_shrink=100, min_sector_n=30)

    results_df = pd.concat(
        [results_df, baseline_df, pd.DataFrame([hrf_row])], ignore_index=True,
    )
    results_df.to_csv("outputs/tables/model_performance_comparison.csv", index=False)
    logger.info("Model results:\n%s",
                results_df[["model", "test_rmse", "test_r2", "test_mae"]].to_string(index=False))

    # ================================================================ #
    #  STEP 8 — Bootstrap CIs + Diebold-Mariano Tests                    #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 8: Statistical Inference")
    logger.info("=" * 70)

    evaluator = Evaluator()
    boot_results = {}
    model_preds = {}

    for name, res in pipeline.results.items():
        boot = evaluator.bootstrap_metrics(res["y_test"], res["preds_test"])
        boot_results[name] = boot
        model_preds[name] = res["preds_test"]
        logger.info(
            "  %s — RMSE: %.4f [%.4f, %.4f], R²: %.4f [%.4f, %.4f]",
            name,
            boot["RMSE"]["point"], boot["RMSE"]["ci_lo"], boot["RMSE"]["ci_hi"],
            boot["R2"]["point"], boot["R2"]["ci_lo"], boot["R2"]["ci_hi"],
        )

    # Save bootstrap results
    boot_rows = []
    for model, metrics in boot_results.items():
        for metric, vals in metrics.items():
            boot_rows.append({"model": model, "metric": metric, **vals})
    pd.DataFrame(boot_rows).to_csv("outputs/tables/bootstrap_confidence_intervals.csv", index=False)

    # DM tests
    if len(model_preds) >= 2:
        y_test_common = list(pipeline.results.values())[0]["y_test"]
        dm_df = evaluator.pairwise_dm_tests(y_test_common, model_preds)
        logger.info("DM tests:\n%s", dm_df.to_string(index=False))

    # ================================================================ #
    #  STEP 9 — SHAP Feature Importance                                  #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 9: SHAP Feature Importance")
    logger.info("=" * 70)

    viz = Visualizer()
    X_test_arr = test_df[features].values
    for model_name in ["XGBoost", "RandomForest", "LightGBM"]:
        shap_df = pipeline.compute_shap(model_name, X_test_arr, features)
        if shap_df is not None:
            viz.plot_shap(shap_df, model_name)
            logger.info("SHAP plot saved for %s", model_name)

    # ================================================================ #
    #  STEP 10 — Robustness Checks                                       #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 10: Robustness Checks")
    logger.info("=" * 70)

    rob_df = evaluator.run_robustness_checks(raw_df, features)
    logger.info("Robustness results:\n%s", rob_df.to_string(index=False))

    # ================================================================ #
    #  STEP 11 — Descriptive Statistics & Tables                         #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 11: Publication Tables")
    logger.info("=" * 70)

    evaluator.descriptive_statistics(processed_df)
    evaluator.variable_definitions()
    evaluator.correlation_matrix(processed_df)
    evaluator.distress_regression(processed_df)
    evaluator.trajectory_volatility_regression(processed_df)
    evaluator.sector_variance_decomposition(processed_df)

    # ================================================================ #
    #  STEP 12 — Publication Figures                                      #
    # ================================================================ #
    logger.info("=" * 70)
    logger.info("STEP 12: Publication Figures")
    logger.info("=" * 70)

    # Model comparison
    viz.plot_model_comparison(results_df)

    # CV performance over time
    if not cv_df.empty:
        viz.plot_cv_performance(cv_df)

    # Actual vs predicted for best model
    if pipeline.results:
        # Restrict to models pipeline.results actually tracks (fitted models,
        # not the naive baselines) to avoid picking an untracked model name.
        fitted_perf = results_df[results_df["model"].isin(pipeline.results.keys())]
        best_model = fitted_perf.loc[fitted_perf["test_rmse"].idxmin(), "model"]
        res = pipeline.results[best_model]
        viz.plot_actual_vs_predicted(res["y_test"], res["preds_test"], best_model)
        viz.plot_residuals(res["y_test"], res["preds_test"], best_model)

    # Robustness forest plot
    viz.plot_robustness(rob_df)

    # Bootstrap CI comparison
    viz.plot_bootstrap_ci(boot_results)

    # Intensity distribution
    viz.plot_intensity_distribution(processed_df)

    # ================================================================ #
    #  Summary                                                            #
    # ================================================================ #
    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("=" * 70)
    logger.info("Data:   %d firm-years (%d reporting, %d non-reporting)",
                len(processed_df),
                int(processed_df["selected"].sum()),
                int((processed_df["selected"] == 0).sum()))
    logger.info("Models: %d trained and evaluated", len(results_df))
    logger.info("CV:     %d folds", cv_df["fold"].nunique() if not cv_df.empty else 0)
    logger.info("Tables: outputs/tables/")
    logger.info("Figures: outputs/figures/")


if __name__ == "__main__":
    main()
