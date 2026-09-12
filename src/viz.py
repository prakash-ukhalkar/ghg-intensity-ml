"""
Visualization Module.

Publication-quality figures (DPI 300, journal-ready aesthetics):

1. Model performance comparison (bar chart).
2. SHAP feature importance (beeswarm / bar).
3. Actual vs predicted scatter.
4. Expanding-window CV performance over time.
5. Robustness forest plot.
6. Coefficient stability across temporal folds.
7. Residual diagnostics.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Journal-quality style defaults
# -----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (7, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

PALETTE = [
    "#2563EB", "#DC2626", "#059669", "#D97706",
    "#7C3AED", "#DB2777", "#0891B2", "#65A30D",
]


class Visualizer:
    """Generates all publication-quality figures."""

    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig, name):
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info("Saved figure → %s", path)

    # ================================================================== #
    #  1. Model comparison bar chart                                      #
    # ================================================================== #
    def plot_model_comparison(
        self, results_df: pd.DataFrame, metric: str = "test_r2",
    ):
        """Grouped bar chart comparing model performance metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))

        # Re-order results to place RandomForest at the top of the horizontal bar plot
        # by sorting. Let's keep the existing order but style consistently.
        colors = ["#A83232" if m == "RandomForest" else "#2A4B7C" for m in results_df["model"]]

        for ax, col, label in zip(
            axes,
            ["test_rmse", "test_r2", "test_mae"],
            ["RMSE", "R²", "MAE"],
        ):
            bars = ax.barh(
                results_df["model"], results_df[col],
                color=colors, edgecolor="white",
            )
            ax.set_xlabel(label)
            ax.set_title(label, fontsize=10, fontweight="bold")
            max_val = results_df[col].max()
            offset = max_val * 0.015
            for bar, val in zip(bars, results_df[col]):
                ax.text(
                    bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7.5,
                )
        fig.tight_layout()
        self._save(fig, "model_comparison.png")

    # ================================================================== #
    #  2. SHAP feature importance                                         #
    # ================================================================== #
    def plot_shap(
        self, shap_df: pd.DataFrame, model_name: str = "XGBoost",
    ):
        """Mean |SHAP| bar chart."""
        mean_abs = shap_df.abs().mean().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(7, max(4, len(mean_abs) * 0.35)))
        ax.barh(mean_abs.index, mean_abs.values, color=PALETTE[0], edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Feature Importance — {model_name}")
        fig.tight_layout()
        self._save(fig, f"shap_{model_name.lower()}.png")

    # ================================================================== #
    #  3. Actual vs predicted scatter                                     #
    # ================================================================== #
    def plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Best Model",
    ):
        """Scatter with 45° reference line."""
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.6, s=30, color=PALETTE[0], edgecolors="white", linewidth=0.3)

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)

        ax.set_xlabel("Actual GHG Intensity")
        ax.set_ylabel("Predicted GHG Intensity")
        ax.set_title(f"Actual vs Predicted — {model_name}")
        fig.tight_layout()
        self._save(fig, f"actual_vs_predicted_{model_name.lower().replace(' ', '_')}.png")

    # ================================================================== #
    #  4. Expanding-window CV performance                                 #
    # ================================================================== #
    def plot_cv_performance(self, cv_df: pd.DataFrame):
        """Line chart of R² and RMSE per CV fold for each model."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for mi, (ax, metric, label) in enumerate(
            zip(axes, ["r2", "rmse"], ["R²", "RMSE"])
        ):
            for ci, model in enumerate(cv_df["model"].unique()):
                sub = cv_df[cv_df["model"] == model]
                ax.plot(
                    sub["test_year"], sub[metric],
                    marker="o", label=model, color=PALETTE[ci % len(PALETTE)],
                    linewidth=1.5, markersize=5,
                )
            ax.set_xlabel("Test Year")
            ax.set_ylabel(label)
            ax.set_title(f"{label} Across CV Folds")
            ax.legend(fontsize=7, loc="best")

        fig.suptitle("Expanding-Window Cross-Validation", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save(fig, "cv_performance.png")

    # ================================================================== #
    #  5. Robustness forest plot                                          #
    # ================================================================== #
    def plot_robustness(self, rob_df: pd.DataFrame, metric: str = "r2"):
        """Grouped horizontal bar chart of R² across robustness specifications."""
        if rob_df.empty:
            return

        checks = list(rob_df["check"].unique())
        if "Baseline" in checks:
            checks.remove("Baseline")
            checks = ["Baseline"] + sorted(checks)

        y_pos = np.arange(len(checks))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7.5, max(4.5, len(checks) * 0.45)))

        r2_ridge = []
        r2_rf = []
        for chk in checks:
            row_ridge = rob_df[(rob_df["check"] == chk) & (rob_df["model"] == "Ridge")]
            row_rf = rob_df[(rob_df["check"] == chk) & (rob_df["model"] == "RF")]

            r2_ridge.append(row_ridge.iloc[0][metric] if not row_ridge.empty else 0.0)
            r2_rf.append(row_rf.iloc[0][metric] if not row_rf.empty else 0.0)

        rects_ridge = ax.barh(y_pos + width/2, r2_ridge, width, label="Ridge Baseline", color="#2A4B7C", edgecolor="white")
        rects_rf = ax.barh(y_pos - width/2, r2_rf, width, label="Random Forest (IMR)", color="#A83232", edgecolor="white")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(checks, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper(), fontsize=9.5)
        ax.set_title("Model Robustness (R²) across Specifications", fontsize=10.5, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlim(-0.05, 0.75)

        for rect in rects_rf:
            w = rect.get_width()
            if w > 0.01:
                ax.text(
                    w + 0.01, rect.get_y() + rect.get_height() / 2,
                    f"{w:.3f}", va="center", fontsize=7.5, color="#A83232", fontweight="bold"
                )

        fig.tight_layout()
        self._save(fig, "robustness_checks.png")

    # ================================================================== #
    #  6. Residual diagnostics                                            #
    # ================================================================== #
    def plot_residuals(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Best",
    ):
        """Residual plot and histogram."""
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=25, color=PALETTE[1])
        axes[0].axhline(0, color="gray", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted")

        # Histogram
        axes[1].hist(residuals, bins=30, color=PALETTE[0], edgecolor="white", alpha=0.8)
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residual Distribution")

        fig.suptitle(f"Residual Diagnostics — {model_name}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save(fig, f"residuals_{model_name.lower().replace(' ', '_')}.png")

    # ================================================================== #
    #  7. Bootstrap CI comparison                                         #
    # ================================================================== #
    def plot_bootstrap_ci(
        self, boot_results: dict[str, dict[str, dict[str, float]]],
    ):
        """
        Forest plot of bootstrap CIs for each model.

        Parameters
        ----------
        boot_results : {model_name: {metric: {point, ci_lo, ci_hi}}}
        """
        metric = "RMSE"
        models = list(boot_results.keys())
        points = [boot_results[m][metric]["point"] for m in models]
        ci_lo = [boot_results[m][metric]["ci_lo"] for m in models]
        ci_hi = [boot_results[m][metric]["ci_hi"] for m in models]

        fig, ax = plt.subplots(figsize=(7, max(3, len(models) * 0.5)))
        y_pos = range(len(models))
        xerr_lo = [p - lo for p, lo in zip(points, ci_lo)]
        xerr_hi = [hi - p for p, hi in zip(points, ci_hi)]

        ax.errorbar(
            points, y_pos, xerr=[xerr_lo, xerr_hi],
            fmt="o", color=PALETTE[0], ecolor="gray", capsize=4, markersize=8,
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(models)
        ax.set_xlabel("RMSE (95% Bootstrap CI)")
        ax.set_title("Model Comparison with Bootstrap Confidence Intervals")
        fig.tight_layout()
        self._save(fig, "bootstrap_ci_comparison.png")

    # ================================================================== #
    #  8. Emissions intensity distribution                                #
    # ================================================================== #
    def plot_intensity_distribution(
        self, df: pd.DataFrame, target: str = "scope1_intensity_rev",
    ):
        """Histogram + boxplot of the target variable."""
        reporting = df[df["selected"] == 1][target].dropna()
        if reporting.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(reporting, bins=40, color=PALETTE[0], edgecolor="white", alpha=0.8)
        axes[0].set_xlabel("GHG Intensity (scope1 / revenue)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Distribution of GHG Intensity")

        axes[1].boxplot(reporting.values, vert=False, patch_artist=True,
                        boxprops=dict(facecolor=PALETTE[0], alpha=0.6))
        axes[1].set_xlabel("GHG Intensity")
        axes[1].set_title("Box Plot")

        fig.tight_layout()
        self._save(fig, "intensity_distribution.png")


if __name__ == "__main__":
    print("Visualization module — import Visualizer.")
