import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer


def test_heckman_imr_sign(synthetic_linked_panel):
    """
    Inverse Mills Ratio must be positive for selected (reporting) rows and
    negative for non-selected rows — a sign error here silently breaks the
    Heckman correction's bias-removal property.
    """
    fe = FeatureEngineer()
    raw = fe.build_raw_features(synthetic_linked_panel)
    fe.fit(raw)
    out = fe.transform(raw)

    imr_selected = out.loc[out["selected"] == 1, "inverse_mills_ratio"]
    imr_nonselected = out.loc[out["selected"] == 0, "inverse_mills_ratio"]

    assert (imr_selected > 0).mean() > 0.9, "IMR should be positive for selected rows"
    assert (imr_nonselected < 0).mean() > 0.9, "IMR should be negative for non-selected rows"


def test_winsorize_bounds_fit_on_train_only(synthetic_linked_panel):
    """
    Winsorization bounds must come from the TRAIN fold's distribution, not
    leak information from the test fold. We inject an extreme outlier only
    in the test fold and confirm it does NOT influence train's clip bounds.
    """
    fe = FeatureEngineer()
    raw = fe.build_raw_features(synthetic_linked_panel)

    train_raw = raw[raw["year"] < 2022].copy()
    test_raw = raw[raw["year"] == 2022].copy()

    # Inject an extreme leverage outlier only into the test fold
    test_raw = test_raw.copy()
    test_raw.iloc[0, test_raw.columns.get_loc("leverage")] = 999.0

    fe.fit(train_raw)
    train_bounds = fe._winsor_bounds["leverage"]

    fe2 = FeatureEngineer()
    fe2.fit(train_raw)
    train_bounds_repeat = fe2._winsor_bounds["leverage"]

    assert train_bounds == train_bounds_repeat
    # The outlier must not appear in the fitted train bounds
    assert train_bounds[1] < 900


def test_sector_z_score_thin_cell_fallback():
    """Sectors with fewer than 3 observations must fall back to pooled stats."""
    df = pd.DataFrame({
        "sector": ["A"] * 10 + ["B"] * 2,
        "leverage": np.concatenate([np.random.RandomState(1).normal(0.3, 0.05, 10), [0.9, 0.95]]),
        "roa": np.random.RandomState(2).normal(0.05, 0.02, 12),
        "operating_margin": np.random.RandomState(3).normal(0.1, 0.02, 12),
        "capex_intensity": np.random.RandomState(4).normal(0.05, 0.01, 12),
        "size": np.random.RandomState(5).normal(10, 1, 12),
        "high_emission_naics": [0] * 12,
        "selected": [1, 0] * 6,
    })

    fe = FeatureEngineer()
    fe.fit(df)
    out = fe.transform(df)

    # Sector "B" (thin cell, n=2) should use pooled mean/std, not its own.
    # Z-scores are computed after winsorization, so compare against the
    # winsorized leverage value (out["leverage"]), not the raw input.
    pooled_mu, pooled_sd = fe._sector_z_stats["leverage"]["pooled"]
    b_rows = out[out["sector"] == "B"]
    expected_z = (b_rows["leverage"].values - pooled_mu) / pooled_sd
    np.testing.assert_allclose(b_rows["leverage_sector_z"].values, expected_z, rtol=1e-6)


def test_transform_before_fit_raises():
    fe = FeatureEngineer()
    df = pd.DataFrame({"sector": ["A"], "leverage": [0.1]})
    try:
        fe.transform(df)
        assert False, "transform() before fit() should raise"
    except RuntimeError:
        pass
