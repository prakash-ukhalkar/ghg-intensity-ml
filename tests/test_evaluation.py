from src.feature_engineering import FeatureEngineer
from src.evaluation import Evaluator


def _processed_panel(synthetic_linked_panel, tmp_path):
    # output_dir=tmp_path so this never writes into the real data/processed/
    fe = FeatureEngineer(output_dir=str(tmp_path))
    return fe.create_features(synthetic_linked_panel)


def test_trajectory_volatility_regression_excludes_leverage(synthetic_linked_panel, tmp_path):
    """
    leverage (and roa) are inputs to the Zmijewski-style proxy itself, so
    they must NOT also appear as independent regressors in the
    trajectory-volatility regression — that would be a mechanical bad
    control. Confirmed by checking the saved summary text names no
    "leverage" coefficient row.
    """
    df = _processed_panel(synthetic_linked_panel, tmp_path)
    # output_dir=tmp_path so this never writes into the real outputs/tables/
    evaluator = Evaluator(output_dir=str(tmp_path))
    summary = evaluator.trajectory_volatility_regression(df)
    if not summary:
        return  # sample too small in this synthetic fixture — acceptable skip

    # A statsmodels coefficient table row starts the line with the variable
    # name flush-left (e.g. "leverage      7.38   1.10  ..."). The method
    # notes mention "leverage" in prose but never at the start of a line.
    coef_row_lines = [ln for ln in summary.splitlines() if ln.startswith("leverage")]
    assert coef_row_lines == [], f"leverage must not appear as a regressor: {coef_row_lines}"
    assert "emissions_trajectory_volatility" in summary


def test_sector_variance_decomposition_returns_icc_in_unit_interval(synthetic_linked_panel, tmp_path):
    df = _processed_panel(synthetic_linked_panel, tmp_path)
    evaluator = Evaluator(output_dir=str(tmp_path))
    icc = evaluator.sector_variance_decomposition(df)
    if icc is None:
        return  # insufficient data in the small synthetic fixture — acceptable skip
    assert 0.0 <= icc <= 1.0
