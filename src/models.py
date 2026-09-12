"""
Modelling Module.

Provides a unified ``ModelPipeline`` that trains, tunes, and evaluates
multiple regression models for GHG intensity prediction:

* Linear: OLS, Ridge, Lasso, ElasticNet
* Tree-based: Random Forest, XGBoost, LightGBM
* Deep: PyTorch MLP
* Naive baselines: SectorMeanBaseline, LastValueBaseline (via
  ``evaluate_naive_baselines``) — establishes whether the ML models beat a
  trivial forecaster, not just each other.
* Sector-hierarchical shrinkage model: ``HierarchicalRF`` (via
  ``train_hierarchical_rf``) — blends per-sector and pooled Random Forest
  predictions to address the pooled model's poor fit in thin sectors.

Supports:
* Group-aware train/test splitting by firm (ticker/cik) to prevent
  panel leakage.
* Expanding-window temporal cross-validation.
* Inner-loop hyperparameter tuning via group-aware ``GroupKFold``.
* SHAP feature-importance computation.

Leakage safety
--------------
Fold-dependent feature statistics (winsorization bounds, sector z-scores,
the Heckman Probit, and imputation medians) are fit ONLY on each split's
training rows via ``FeatureEngineer.fit()``/``transform()`` (see
``feature_engineering.py``), then applied to that split's test rows. This
applies to both the final holdout split (``prepare_train_test``) and every
fold of ``expanding_window_cv``.
"""
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Any

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import GroupKFold

from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===================================================================== #
#  Hyperparameter grids for tuning                                       #
# ===================================================================== #
HYPERPARAM_GRIDS: dict[str, list[dict]] = {
    "Ridge": [
        {"alpha": a} for a in [0.01, 0.1, 1.0, 10.0, 100.0]
    ],
    "Lasso": [
        {"alpha": a} for a in [0.001, 0.01, 0.1, 1.0, 10.0]
    ],
    "ElasticNet": [
        {"alpha": a, "l1_ratio": r}
        for a in [0.01, 0.1, 1.0]
        for r in [0.25, 0.5, 0.75]
    ],
    "RandomForest": [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
        for n in [100, 300]
        for d in [5, 10, 20]
        for l in [2, 5]
    ],
    "XGBoost": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss}
        for n in [100, 300]
        for d in [3, 5, 7]
        for lr in [0.01, 0.05, 0.1]
        for ss in [0.8]
    ],
    "LightGBM": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "num_leaves": nl}
        for n in [100, 300]
        for d in [5, 10]
        for lr in [0.01, 0.05, 0.1]
        for nl in [31, 63]
    ],
}


class HierarchicalRF:
    """
    Shrinkage-blended Random Forest for sector-heterogeneous panels.

    Fits one global (pooled) Random Forest plus one per-sector Random
    Forest for every sector with at least ``min_sector_n`` training rows.
    Predictions blend the sector model and the global model with an
    empirical-Bayes-style weight ``w = n_sector / (n_sector + k_shrink)``:
    sectors with abundant training data lean on their own model, thin
    sectors lean on the pooled model, and sectors with no fitted model
    (below ``min_sector_n``) fall back to the pooled model entirely.

    This directly targets the sector-heterogeneity problem observed with a
    single pooled model (strong fit in Manufacturing, catastrophic fit in
    thin sectors like Financials/Mining/Wholesale — see
    ``Evaluator.run_robustness_checks``'s sector-subsample checks).
    """

    def __init__(
        self,
        k_shrink: int = 15,
        min_sector_n: int = 20,
        rf_kwargs: dict | None = None,
        random_state: int = 42,
    ):
        self.k_shrink = k_shrink
        self.min_sector_n = min_sector_n
        self.rf_kwargs = rf_kwargs or {
            "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 3,
            "random_state": random_state, "n_jobs": -1,
        }
        self.global_model = RandomForestRegressor(**self.rf_kwargs)
        self.sector_models: dict[Any, RandomForestRegressor] = {}
        self.sector_n: dict[Any, int] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, sectors: np.ndarray) -> "HierarchicalRF":
        self.global_model.fit(X, y)
        sectors = np.asarray(sectors)
        for sec in pd.unique(sectors):
            mask = sectors == sec
            n = int(mask.sum())
            self.sector_n[sec] = n
            if n >= self.min_sector_n:
                mdl = RandomForestRegressor(**self.rf_kwargs)
                mdl.fit(X[mask], y[mask])
                self.sector_models[sec] = mdl
        return self

    def predict(self, X: np.ndarray, sectors: np.ndarray) -> np.ndarray:
        sectors = np.asarray(sectors)
        pred_global = self.global_model.predict(X)
        pred = pred_global.copy()
        for sec, mdl in self.sector_models.items():
            mask = sectors == sec
            if not mask.any():
                continue
            n = self.sector_n[sec]
            w = n / (n + self.k_shrink)
            pred_sec = mdl.predict(X[mask])
            pred[mask] = w * pred_sec + (1 - w) * pred_global[mask]
        return pred


class ModelPipeline:
    """
    Unified model training, tuning, and evaluation pipeline.

    Operates on the RAW linked panel (output of
    ``FeatureEngineer.build_raw_features()``) rather than a pre-winsorized /
    pre-z-scored table, so that fold-dependent statistics can be fit on
    each split's training rows only.

    Parameters
    ----------
    target : str
        Target column name (default ``scope1_intensity_rev``).
    features : list[str] | None
        Explicit feature list; auto-detected if None.
    group_col : str
        Column for group-aware splitting (``ticker`` or ``cik``).
    """

    DEFAULT_FEATURES = [
        "size", "leverage", "roa", "operating_margin",
        "capex_intensity", "rd_intensity", "revenue_growth",
        "inverse_mills_ratio",
        "us_gdp_growth", "us_co2_per_capita", "us_energy_use_per_capita",
    ]

    def __init__(
        self,
        target: str = "scope1_intensity_rev",
        features: list[str] | None = None,
        group_col: str = "cik",
    ):
        self.target = target
        self.features = features
        self.group_col = group_col
        self.results: dict[str, dict] = {}
        self.fitted_models: dict[str, Any] = {}
        self.fe: FeatureEngineer | None = None  # FeatureEngineer fit on the final train split

    # ================================================================== #
    #  Helpers                                                            #
    # ================================================================== #
    def _filter_reporting(self, df: pd.DataFrame) -> pd.DataFrame:
        reporting = df[df["selected"] == 1].copy()
        reporting = reporting.dropna(subset=[self.target])
        return reporting

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        if self.features is not None:
            feats = [f for f in self.features if f in df.columns]
        else:
            feats = [f for f in self.DEFAULT_FEATURES if f in df.columns]
            feats += [c for c in df.columns if c.startswith("year_")]
            feats += [c for c in df.columns if c.endswith("_sector_z")]
        feats = [f for f in feats if df[f].notna().any()]
        return feats

    def _impute_train_test(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feats: list[str],
    ) -> None:
        """Fills remaining NaNs in-place using TRAIN-only medians."""
        for f in feats:
            med = train_df[f].median()
            if pd.isna(med):
                med = 0.0
            train_df[f] = train_df[f].fillna(med)
            test_df[f] = test_df[f].fillna(med)

    # ================================================================== #
    #  Temporal split on the RAW (unfitted) panel                         #
    # ================================================================== #
    def temporal_split_raw(
        self, raw_df: pd.DataFrame, test_years: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the full raw panel by time: earlier years train, later years test."""
        if test_years is None:
            all_years = sorted(raw_df["year"].unique())
            n_test = max(1, len(all_years) // 3)
            test_years = all_years[-n_test:]

        train_raw = raw_df[~raw_df["year"].isin(test_years)].copy()
        test_raw = raw_df[raw_df["year"].isin(test_years)].copy()
        return train_raw, test_raw

    # ================================================================== #
    #  Final holdout split — leakage-free feature fitting                 #
    # ================================================================== #
    def prepare_train_test(
        self, raw_df: pd.DataFrame, test_years: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Splits the raw panel by year, fits ``FeatureEngineer`` on the
        training rows only, transforms both folds, filters to the
        reporting subset, and imputes remaining NaNs with train medians.

        Returns (train_df, test_df, feature_list).
        """
        train_raw, test_raw = self.temporal_split_raw(raw_df, test_years)
        logger.info(
            "Train years: %s (%d raw rows), Test years: %s (%d raw rows)",
            sorted(train_raw["year"].unique()), len(train_raw),
            sorted(test_raw["year"].unique()), len(test_raw),
        )

        fe = FeatureEngineer()
        fe.fit(train_raw)
        train_full = fe.transform(train_raw)
        test_full = fe.transform(test_raw)
        self.fe = fe

        train_df = self._filter_reporting(train_full)
        test_df = self._filter_reporting(test_full)

        feats = self._select_features(train_df)
        self._impute_train_test(train_df, test_df, feats)

        logger.info(
            "Reporting subset: %d train / %d test firm-years, %d features, target = %s",
            len(train_df), len(test_df), len(feats), self.target,
        )
        return train_df, test_df, feats

    # ================================================================== #
    #  Expanding-window cross-validation (leakage-free per fold)          #
    # ================================================================== #
    def expanding_window_cv(
        self, raw_df: pd.DataFrame, min_train_years: int = 3,
    ) -> pd.DataFrame:
        """
        Expanding-window temporal CV: train on [start, t], test on t+1.
        Refits ``FeatureEngineer`` on each fold's training rows only.

        Returns a DataFrame of per-fold-per-model metrics.
        """
        all_years = sorted(raw_df["year"].unique())
        fold_results: list[dict] = []

        for i in range(min_train_years, len(all_years)):
            train_years = all_years[:i]
            test_year = all_years[i]

            train_raw = raw_df[raw_df["year"].isin(train_years)]
            test_raw = raw_df[raw_df["year"] == test_year]

            fe = FeatureEngineer()
            try:
                fe.fit(train_raw)
            except Exception as exc:
                logger.warning("Fold %d: FeatureEngineer.fit failed: %s", i, exc)
                continue
            train_full = fe.transform(train_raw)
            test_full = fe.transform(test_raw)

            train = self._filter_reporting(train_full)
            test = self._filter_reporting(test_full)

            if len(train) < 10 or len(test) < 3:
                continue

            feats = self._select_features(train)
            self._impute_train_test(train, test, feats)

            X_train = train[feats].values
            y_train = train[self.target].values
            X_test = test[feats].values
            y_test = test[self.target].values

            for name, model_fn in self._model_factories().items():
                try:
                    model = model_fn()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    fold_results.append({
                        "fold": i - min_train_years + 1,
                        "train_years": str(train_years),
                        "test_year": test_year,
                        "model": name,
                        "n_train": len(train),
                        "n_test": len(test),
                        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                        "mae": mean_absolute_error(y_test, preds),
                        "r2": r2_score(y_test, preds) if len(y_test) > 1 else np.nan,
                        "mape": mean_absolute_percentage_error(y_test, preds) * 100,
                    })
                except Exception as exc:
                    logger.warning("Fold %d, model %s failed: %s", i, name, exc)

        cv_df = pd.DataFrame(fold_results)
        if not cv_df.empty:
            logger.info("Expanding-window CV: %d folds × %d models",
                         cv_df["fold"].nunique(), cv_df["model"].nunique())
        return cv_df

    # ================================================================== #
    #  Train with optional hyperparameter tuning                          #
    # ================================================================== #
    def train_all_models(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list[str],
        tune: bool = True,
    ) -> pd.DataFrame:
        """
        Trains all models on the training set and evaluates on the test set.

        Parameters
        ----------
        tune : bool
            If True, performs inner-loop group-aware hyperparameter tuning
            (using GroupKFold on the training fold only — test fold never
            participates in tuning).

        Returns
        -------
        pd.DataFrame  with one row per model and metric columns.
        """
        X_train = train_df[features].values
        y_train = train_df[self.target].values
        X_test = test_df[features].values
        y_test = test_df[self.target].values

        groups_train = train_df[self.group_col].values if self.group_col in train_df.columns else None

        results: list[dict] = []

        for name, model_fn in self._model_factories().items():
            try:
                if tune and name in HYPERPARAM_GRIDS and groups_train is not None:
                    best_params = self._tune_model(
                        name, model_fn, X_train, y_train, groups_train,
                    )
                    model = model_fn(**best_params)
                    logger.info("  %s best params: %s", name, best_params)
                else:
                    model = model_fn()
                    best_params = {}

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)

                preds_train = model.predict(X_train)
                preds_test = model.predict(X_test)

                self.fitted_models[name] = model

                row = self._metrics_row(name, y_train, preds_train, y_test, preds_test)
                row["best_params"] = str(best_params)
                results.append(row)

                self.results[name] = {
                    "y_test": y_test,
                    "preds_test": preds_test,
                    "y_train": y_train,
                    "preds_train": preds_train,
                }

            except Exception as exc:
                logger.error("Model %s failed: %s", name, exc)

        df_results = pd.DataFrame(results)
        logger.info("Trained %d models.", len(df_results))
        return df_results

    # ================================================================== #
    #  Shared metrics-row builder                                         #
    # ================================================================== #
    def _metrics_row(
        self,
        name: str,
        y_train: np.ndarray,
        preds_train: np.ndarray,
        y_test: np.ndarray,
        preds_test: np.ndarray,
    ) -> dict:
        """Builds one results_df-compatible row from train/test predictions."""
        return {
            "model": name,
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, preds_train))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds_test))),
            "train_r2": float(r2_score(y_train, preds_train)),
            "test_r2": float(r2_score(y_test, preds_test)) if len(y_test) > 1 else np.nan,
            "test_mae": float(mean_absolute_error(y_test, preds_test)),
            "test_mape": float(mean_absolute_percentage_error(y_test, preds_test) * 100),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "best_params": "{}",
        }

    # ================================================================== #
    #  Naive baselines (SectorMean / LastValue)                           #
    # ================================================================== #
    def evaluate_naive_baselines(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, sector_col: str = "sector",
    ) -> pd.DataFrame:
        """
        Two trivial forecasters, evaluated on the same split as the ML
        models, so the paper can show the ML models beat a naive floor and
        not just each other.

        * ``SectorMeanBaseline`` — predicts each row with its sector's
          training-period mean intensity. Train-set predictions use a
          leave-one-out sector mean so the in-sample fit isn't trivially
          perfect.
        * ``LastValueBaseline`` — predicts each firm-year with that firm's
          own most recent training-year value, falling back to the sector
          mean for firms with no prior observation in the training window.
        """
        target = self.target
        group_col = self.group_col

        train = train_df.sort_values([group_col, "year"]).reset_index(drop=True)
        test = test_df.reset_index(drop=True)

        global_mean = train[target].mean()
        sector_means = train.groupby(sector_col)[target].mean()

        grp_sum = train.groupby(sector_col)[target].transform("sum")
        grp_cnt = train.groupby(sector_col)[target].transform("count")
        denom = (grp_cnt - 1).replace(0, np.nan)
        loo_sector_mean = ((grp_sum - train[target]) / denom).fillna(global_mean)
        test_sector_mean = test[sector_col].map(sector_means).fillna(global_mean)

        rows = [self._metrics_row(
            "SectorMeanBaseline",
            train[target].values, loo_sector_mean.values,
            test[target].values, test_sector_mean.values,
        )]

        prev_val = train.groupby(group_col)[target].shift(1)
        train_lastval = prev_val.fillna(loo_sector_mean)

        last_obs = train.groupby(group_col)[target].last()
        test_lastval = test[group_col].map(last_obs).fillna(test_sector_mean)

        rows.append(self._metrics_row(
            "LastValueBaseline",
            train[target].values, train_lastval.values,
            test[target].values, test_lastval.values,
        ))

        return pd.DataFrame(rows)

    # ================================================================== #
    #  Sector-hierarchical shrinkage RF                                    #
    # ================================================================== #
    def train_hierarchical_rf(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list[str],
        k_shrink: int = 15,
        min_sector_n: int = 20,
        sector_col: str = "sector",
    ) -> dict:
        """
        Fits ``HierarchicalRF`` (see module-level class) and records it in
        ``self.results``/``self.fitted_models`` alongside the standard
        models, so it participates in bootstrap CIs and Diebold-Mariano
        tests via the same downstream code path.
        """
        X_train = train_df[features].values
        y_train = train_df[self.target].values
        X_test = test_df[features].values
        y_test = test_df[self.target].values
        sectors_train = train_df[sector_col].values if sector_col in train_df.columns else np.zeros(len(train_df))
        sectors_test = test_df[sector_col].values if sector_col in test_df.columns else np.zeros(len(test_df))

        model = HierarchicalRF(k_shrink=k_shrink, min_sector_n=min_sector_n)
        model.fit(X_train, y_train, sectors_train)
        preds_train = model.predict(X_train, sectors_train)
        preds_test = model.predict(X_test, sectors_test)

        self.fitted_models["HierarchicalRF"] = model
        self.results["HierarchicalRF"] = {
            "y_test": y_test, "preds_test": preds_test,
            "y_train": y_train, "preds_train": preds_train,
        }

        row = self._metrics_row("HierarchicalRF", y_train, preds_train, y_test, preds_test)
        row["best_params"] = str({"k_shrink": k_shrink, "min_sector_n": min_sector_n})
        return row

    # ================================================================== #
    #  Hyperparameter tuning (inner loop, train-fold only)                #
    # ================================================================== #
    def _tune_model(
        self,
        name: str,
        model_fn,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> dict:
        """Inner-loop group-aware CV for hyperparameter selection (train fold only)."""
        grid = HYPERPARAM_GRIDS.get(name, [{}])
        unique_groups = np.unique(groups)
        n_splits = min(3, len(unique_groups))

        if n_splits < 2:
            return grid[0] if grid else {}

        gkf = GroupKFold(n_splits=n_splits)
        best_score = np.inf
        best_params: dict = {}

        for params in grid:
            scores: list[float] = []
            try:
                for train_idx, val_idx in gkf.split(X, y, groups):
                    model = model_fn(**params)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X[train_idx], y[train_idx])
                    preds = model.predict(X[val_idx])
                    scores.append(mean_squared_error(y[val_idx], preds))

                avg = np.mean(scores)
                if avg < best_score:
                    best_score = avg
                    best_params = params
            except Exception:
                continue

        return best_params

    # ================================================================== #
    #  SHAP importance                                                    #
    # ================================================================== #
    def compute_shap(
        self, model_name: str, X: np.ndarray, feature_names: list[str],
    ) -> pd.DataFrame | None:
        """Compute SHAP values for the specified fitted model."""
        model = self.fitted_models.get(model_name)
        if model is None:
            return None
        try:
            import shap
            if model_name in ("XGBoost", "LightGBM", "RandomForest"):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X)
            shap_vals = explainer.shap_values(X)
            return pd.DataFrame(shap_vals, columns=feature_names)
        except Exception as exc:
            logger.warning("SHAP failed for %s: %s", model_name, exc)
            return None

    # ================================================================== #
    #  MLP                                                                #
    # ================================================================== #
    def _create_mlp(self, input_dim: int = 10, **kwargs):
        """Creates a PyTorch MLP regressor wrapped in sklearn interface."""
        try:
            import torch
            import torch.nn as nn

            class _MLP(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(d, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.4),
                        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(32, 1),
                    )
                def forward(self, x):
                    return self.net(x).squeeze(-1)

            class MLPRegressor:
                def __init__(self, input_dim=10, epochs=200, lr=1e-3,
                             weight_decay=1e-3, patience=20, **kw):
                    self.input_dim = input_dim
                    self.epochs = epochs
                    self.lr = lr
                    self.weight_decay = weight_decay
                    self.patience = patience
                    self.model = None
                    self.scaler_x = None
                    self.scaler_y = None

                def fit(self, X, y):
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.model_selection import train_test_split as tts

                    self.scaler_x = StandardScaler().fit(X)
                    self.scaler_y = StandardScaler().fit(y.reshape(-1, 1))
                    Xs = self.scaler_x.transform(X)
                    ys = self.scaler_y.transform(y.reshape(-1, 1)).ravel()

                    # Internal early-stopping split (train fold only — no test leakage)
                    if len(Xs) >= 20:
                        Xtr, Xval, ytr, yval = tts(Xs, ys, test_size=0.2, random_state=42)
                    else:
                        Xtr, ytr = Xs, ys
                        Xval, yval = Xs, ys

                    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
                    ytr_t = torch.tensor(ytr, dtype=torch.float32)
                    Xval_t = torch.tensor(Xval, dtype=torch.float32)
                    yval_t = torch.tensor(yval, dtype=torch.float32)

                    self.model = _MLP(X.shape[1])
                    opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                            weight_decay=self.weight_decay)
                    loss_fn = nn.MSELoss()

                    best_val = np.inf
                    best_state = None
                    stale = 0
                    for _ in range(self.epochs):
                        self.model.train()
                        opt.zero_grad()
                        loss_fn(self.model(Xtr_t), ytr_t).backward()
                        opt.step()

                        self.model.eval()
                        with torch.no_grad():
                            val_loss = loss_fn(self.model(Xval_t), yval_t).item()
                        if val_loss < best_val - 1e-6:
                            best_val = val_loss
                            best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                            stale = 0
                        else:
                            stale += 1
                            if stale >= self.patience:
                                break

                    if best_state is not None:
                        self.model.load_state_dict(best_state)
                    return self

                def predict(self, X):
                    self.model.eval()
                    import torch as _t
                    with _t.no_grad():
                        Xs = _t.tensor(self.scaler_x.transform(X), dtype=_t.float32)
                        p = self.model(Xs).numpy()
                    return self.scaler_y.inverse_transform(p.reshape(-1, 1)).ravel()

            return MLPRegressor(input_dim=input_dim, **kwargs)
        except ImportError:
            logger.warning("PyTorch not available — skipping MLP.")
            return None

    # ================================================================== #
    #  Model factory registry                                             #
    # ================================================================== #
    def _model_factories(self) -> dict[str, Any]:
        """Returns callables that produce fresh model instances."""
        factories: dict = {
            "OLS": lambda **kw: LinearRegression(**kw),
            "Ridge": lambda **kw: Ridge(**({"alpha": 1.0} | kw)),
            "Lasso": lambda **kw: Lasso(**({"alpha": 0.1, "max_iter": 5000} | kw)),
            "ElasticNet": lambda **kw: ElasticNet(**({"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 5000} | kw)),
            "RandomForest": lambda **kw: RandomForestRegressor(
                **({"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 3,
                    "random_state": 42, "n_jobs": -1} | kw)
            ),
        }

        # XGBoost
        try:
            from xgboost import XGBRegressor
            factories["XGBoost"] = lambda **kw: XGBRegressor(
                **({"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                    "subsample": 0.8, "reg_lambda": 1.0,
                    "random_state": 42, "verbosity": 0} | kw)
            )
        except ImportError:
            pass

        # LightGBM
        try:
            from lightgbm import LGBMRegressor
            factories["LightGBM"] = lambda **kw: LGBMRegressor(
                **({"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
                    "num_leaves": 15, "min_child_samples": 10,
                    "random_state": 42, "verbose": -1} | kw)
            )
        except ImportError:
            pass

        # MLP
        try:
            import torch  # noqa: F401
            factories["MLP"] = lambda **kw: self._create_mlp(**kw)
        except ImportError:
            pass

        return factories


if __name__ == "__main__":
    print("Model pipeline module — import ModelPipeline and call prepare_train_test()/train_all_models().")
