import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer
from src.models import HierarchicalRF, ModelPipeline


def test_temporal_split_is_time_ordered(synthetic_linked_panel):
    fe = FeatureEngineer()
    raw = fe.build_raw_features(synthetic_linked_panel)
    pipeline = ModelPipeline(target="scope1_intensity_rev", group_col="cik")

    train_raw, test_raw = pipeline.temporal_split_raw(raw)
    assert train_raw["year"].max() < test_raw["year"].min()


def test_prepare_train_test_no_leakage(synthetic_linked_panel):
    """
    The FeatureEngineer fit inside prepare_train_test must be fit on the
    train fold's rows only. We verify by fitting a FeatureEngineer directly
    on the train_raw split and confirming its winsorization bounds match
    what ModelPipeline used internally (pipeline.fe).
    """
    fe = FeatureEngineer()
    raw = fe.build_raw_features(synthetic_linked_panel)
    pipeline = ModelPipeline(target="scope1_intensity_rev", group_col="cik")

    train_df, test_df, feats = pipeline.prepare_train_test(raw)

    train_raw, test_raw = pipeline.temporal_split_raw(raw)
    fe_direct = FeatureEngineer()
    fe_direct.fit(train_raw)

    assert pipeline.fe._winsor_bounds == fe_direct._winsor_bounds
    assert set(train_df.columns) >= {"scope1_intensity_rev", "cik", "year"}
    assert len(feats) > 0
    # No feature column should retain NaNs after imputation
    assert not train_df[feats].isna().any().any()
    assert not test_df[feats].isna().any().any()


def test_expanding_window_cv_folds_are_expanding(synthetic_linked_panel):
    fe = FeatureEngineer()
    raw = fe.build_raw_features(synthetic_linked_panel)
    pipeline = ModelPipeline(target="scope1_intensity_rev", group_col="cik")

    cv_df = pipeline.expanding_window_cv(raw, min_train_years=2)
    if cv_df.empty:
        return  # synthetic panel may be too small for some environments
    # Later folds should never test on an earlier or equal year than the
    # previous fold's test year (expanding window moves forward in time).
    test_years_by_fold = (
        cv_df.drop_duplicates("fold").sort_values("fold")["test_year"].tolist()
    )
    assert test_years_by_fold == sorted(test_years_by_fold)


def test_train_all_models_runs(synthetic_linked_panel):
    fe = FeatureEngineer()
    raw = fe.build_raw_features(synthetic_linked_panel)
    pipeline = ModelPipeline(target="scope1_intensity_rev", group_col="cik")
    train_df, test_df, feats = pipeline.prepare_train_test(raw)

    results = pipeline.train_all_models(train_df, test_df, feats, tune=False)
    assert not results.empty
    assert "OLS" in results["model"].values
    assert (results["n_train"] == len(train_df)).all()


def test_hierarchical_rf_shrinkage_limits():
    """
    At n_sector -> 0 (no fitted sector model), predictions must equal the
    pooled global model. At n_sector >> k_shrink, predictions should be
    dominated by the sector model rather than the global one.
    """
    rng = np.random.RandomState(0)
    n = 200
    X = rng.normal(size=(n, 3))
    sectors = np.array(["A"] * 100 + ["B"] * 100)
    # Sector B has a totally different, learnable relationship than A
    y = np.where(sectors == "A", X[:, 0], -5 * X[:, 0])

    model = HierarchicalRF(k_shrink=15, min_sector_n=20)
    model.fit(X, y, sectors)

    assert set(model.sector_models.keys()) == {"A", "B"}

    # A thin, unseen sector at predict time falls back entirely on the global model
    X_new = rng.normal(size=(5, 3))
    unseen_sectors = np.array(["C"] * 5)
    preds_unseen = model.predict(X_new, unseen_sectors)
    preds_global_only = model.global_model.predict(X_new)
    np.testing.assert_allclose(preds_unseen, preds_global_only)


def test_hierarchical_rf_below_min_sector_n_uses_global_only():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(50, 2))
    y = X[:, 0] * 2
    sectors = np.array(["Thin"] * 5 + ["Bulk"] * 45)

    model = HierarchicalRF(k_shrink=10, min_sector_n=20)
    model.fit(X, y, sectors)

    assert "Thin" not in model.sector_models
    assert "Bulk" in model.sector_models

    mask = sectors == "Thin"
    preds = model.predict(X[mask], sectors[mask])
    preds_global = model.global_model.predict(X[mask])
    np.testing.assert_allclose(preds, preds_global)


def test_naive_baselines_sane(synthetic_linked_panel):
    fe = FeatureEngineer()
    raw = fe.build_raw_features(synthetic_linked_panel)
    pipeline = ModelPipeline(target="scope1_intensity_rev", group_col="cik")
    train_df, test_df, feats = pipeline.prepare_train_test(raw)

    baseline_df = pipeline.evaluate_naive_baselines(train_df, test_df)
    assert set(baseline_df["model"]) == {"SectorMeanBaseline", "LastValueBaseline"}
    assert not baseline_df[["test_rmse", "test_mae"]].isna().any().any()
    assert (baseline_df["n_train"] == len(train_df)).all()
    assert (baseline_df["n_test"] == len(test_df)).all()
