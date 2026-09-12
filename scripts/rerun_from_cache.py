"""
Re-runs the downstream pipeline (linking → feature engineering → modelling →
evaluation → figures) using already-cached raw data on disk, without
re-hitting EPA/SEC/World Bank over the network. Used to pick up code changes
(e.g. a new evaluation table) without repeating the rate-limited SEC pull.
"""
import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_resolution import EntityResolver
from src.feature_engineering import FeatureEngineer
from src.models import ModelPipeline
from src.evaluation import Evaluator
from src.viz import Visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("rerun")


def main():
    epa_df = pd.read_csv("data/raw/epa_ghgrp_facilities.csv")
    epa_df = epa_df.rename(columns={
        "total_ghg_emissions": "total_ghg_emissions",
        "co2_emissions_non_biogenic": "co2_emissions_non_biogenic",
    })
    sec_df = pd.read_csv("data/raw/sec_financials.csv", dtype={"cik": str})
    wb_df = pd.read_csv("data/raw/worldbank_macro.csv")
    mapping_df = pd.read_csv("data/interim/epa_sec_mapping.csv", dtype={"cik": str})

    resolver = EntityResolver()
    matched_ciks = mapping_df[mapping_df["is_resolved"]]["cik"].dropna().unique().tolist()
    control_ciks = [c for c in sec_df["cik"].unique() if c not in set(matched_ciks)]

    linked_df = resolver.link_datasets(epa_df, sec_df, mapping_df, wb_df, control_ciks=control_ciks)
    logger.info("Linked panel: %d firm-years (%d reporting)", len(linked_df), int(linked_df["selected"].sum()))

    fe = FeatureEngineer()
    processed_df = fe.create_features(linked_df)
    raw_df = fe.build_raw_features(linked_df)

    pipeline = ModelPipeline(target="scope1_intensity_rev", group_col="cik")
    cv_df = pipeline.expanding_window_cv(raw_df, min_train_years=3)
    if not cv_df.empty:
        cv_summary = cv_df.groupby("model").agg(
            mean_r2=("r2", "mean"), std_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
            n_folds=("fold", "nunique"),
        ).round(4)
        cv_summary.to_csv("outputs/tables/cv_summary.csv")

    train_df, test_df, features = pipeline.prepare_train_test(raw_df)
    results_df = pipeline.train_all_models(train_df, test_df, features, tune=True)

    baseline_df = pipeline.evaluate_naive_baselines(train_df, test_df)
    hrf_row = pipeline.train_hierarchical_rf(train_df, test_df, features, k_shrink=100, min_sector_n=30)
    results_df = pd.concat(
        [results_df, baseline_df, pd.DataFrame([hrf_row])], ignore_index=True,
    )
    results_df.to_csv("outputs/tables/model_performance_comparison.csv", index=False)
    logger.info("Model results:\n%s", results_df[["model", "test_rmse", "test_r2", "test_mae"]].to_string(index=False))

    evaluator = Evaluator()
    boot_results, model_preds = {}, {}
    for name, res in pipeline.results.items():
        boot = evaluator.bootstrap_metrics(res["y_test"], res["preds_test"])
        boot_results[name] = boot
        model_preds[name] = res["preds_test"]

    boot_rows = [{"model": m, "metric": met, **v} for m, mm in boot_results.items() for met, v in mm.items()]
    pd.DataFrame(boot_rows).to_csv("outputs/tables/bootstrap_confidence_intervals.csv", index=False)

    if len(model_preds) >= 2:
        y_test_common = list(pipeline.results.values())[0]["y_test"]
        evaluator.pairwise_dm_tests(y_test_common, model_preds)

    viz = Visualizer()
    X_test_arr = test_df[features].values
    for model_name in ["XGBoost", "RandomForest", "LightGBM"]:
        shap_df = pipeline.compute_shap(model_name, X_test_arr, features)
        if shap_df is not None:
            viz.plot_shap(shap_df, model_name)

    rob_df = evaluator.run_robustness_checks(raw_df, features)

    evaluator.descriptive_statistics(processed_df)
    evaluator.variable_definitions()
    evaluator.correlation_matrix(processed_df)
    evaluator.distress_regression(processed_df)
    evaluator.trajectory_volatility_regression(processed_df)
    evaluator.sector_variance_decomposition(processed_df)

    viz.plot_model_comparison(results_df)
    if not cv_df.empty:
        viz.plot_cv_performance(cv_df)
    if pipeline.results:
        # Restrict to models pipeline.results actually tracks (fitted models,
        # not the naive baselines) to avoid picking an untracked model name.
        fitted_perf = results_df[results_df["model"].isin(pipeline.results.keys())]
        best_model = fitted_perf.loc[fitted_perf["test_rmse"].idxmin(), "model"]
        res = pipeline.results[best_model]
        viz.plot_actual_vs_predicted(res["y_test"], res["preds_test"], best_model)
        viz.plot_residuals(res["y_test"], res["preds_test"], best_model)
    viz.plot_robustness(rob_df)
    viz.plot_bootstrap_ci(boot_results)
    viz.plot_intensity_distribution(processed_df)

    logger.info("Done. Data: %d firm-years (%d reporting, %d non-reporting)",
                len(processed_df), int(processed_df["selected"].sum()),
                int((processed_df["selected"] == 0).sum()))


if __name__ == "__main__":
    main()
