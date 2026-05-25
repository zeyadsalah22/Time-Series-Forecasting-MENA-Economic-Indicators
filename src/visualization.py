"""
Visualization utilities for the MENA time series forecasting project.
Produces publication-quality plots for time series diagnostics,
forecast comparisons, metric summaries, and DM test heatmaps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

COLORS = {
    "arima": "#2E86AB",
    "ets": "#A23B72",
    "prophet": "#06A77D",
    "lstm": "#E63946",
    "actual": "#1A1A2E",
    "ci": "#CCCCCC",
}


def plot_gdp_growth_panel(
    panel: pd.DataFrame,
    countries: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Multi-panel time series plot of GDP growth for each country.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel with columns [country_code, country_name, year,
        gdp_growth].
    countries : list of str, optional
        Country codes to plot. Defaults to all countries in the panel.
    figsize : tuple
    save_path : str, optional
    """
    if countries is None:
        countries = panel["country_code"].unique().tolist()

    n = len(countries)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for idx, code in enumerate(countries):
        subset = panel[panel["country_code"] == code]
        name = subset["country_name"].iloc[0]
        axes[idx].plot(subset["year"], subset["gdp_growth"],
                       linewidth=2, color=COLORS["actual"])
        axes[idx].axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        axes[idx].set_title(name, fontweight="bold")
        axes[idx].set_xlabel("Year")
        axes[idx].set_ylabel("GDP Growth (%)")
        axes[idx].grid(alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Annual GDP Growth — MENA Panel (1990–2023)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_decomposition(
    series: pd.Series,
    country_name: str,
    period: int = 1,
    model: str = "additive",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot seasonal decomposition (trend, seasonal, residual).

    Parameters
    ----------
    series : pd.Series
        Time series indexed by integer year.
    country_name : str
    period : int
        Seasonal period (1 for annual data with no within-year seasonality).
    model : str
        'additive' or 'multiplicative'.
    figsize : tuple
    save_path : str, optional
    """
    result = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    components = [
        (series.values, "Observed"),
        (result.trend, "Trend"),
        (result.seasonal, "Seasonal"),
        (result.resid, "Residual"),
    ]
    for ax, (data, label) in zip(axes, components):
        ax.plot(series.index, data, linewidth=1.8, color=COLORS["actual"])
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[0].set_title(f"Seasonal Decomposition — {country_name}", fontweight="bold")
    axes[-1].set_xlabel("Year")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_acf_pacf(
    series: pd.Series,
    country_name: str,
    lags: int = 20,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ACF and PACF up to the specified number of lags.

    Parameters
    ----------
    series : pd.Series
    country_name : str
    lags : int
        Maximum lag (default 20 per study specification).
    figsize : tuple
    save_path : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_acf(series.dropna(), lags=lags, ax=axes[0], color=COLORS["actual"])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], color=COLORS["actual"])
    axes[0].set_title(f"ACF — {country_name}")
    axes[1].set_title(f"PACF — {country_name}")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_stationarity_summary(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap of ADF and KPSS stationarity test outcomes per country.

    Parameters
    ----------
    results_df : pd.DataFrame
        Columns: [Country, ADF Stationary, KPSS Stationary] with boolean values.
    figsize : tuple
    save_path : str, optional
    """
    heat_data = results_df.set_index("Country")[["ADF Stationary", "KPSS Stationary"]].astype(int)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heat_data.T, annot=True, fmt="d", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Stationary (1) / Non-Stationary (0)"},
    )
    ax.set_title("Stationarity Test Results — MENA Countries", fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast_df: pd.DataFrame,
    model_name: str,
    country_name: str,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot historical data alongside point forecasts and 95% prediction intervals.

    Parameters
    ----------
    train : pd.Series
        Training time series.
    test : pd.Series
        Test time series (actual values).
    forecast_df : pd.DataFrame
        Columns [forecast, lower, upper], indexed 0..steps-1.
    model_name : str
        Model label for the legend.
    country_name : str
    figsize : tuple
    save_path : str, optional
    """
    color = COLORS.get(model_name.lower().replace(" ", "_"), "#333333")
    test_years = test.index.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train.index, train.values, color=COLORS["actual"],
            linewidth=2, label="Training data")
    ax.plot(test.index, test.values, color=COLORS["actual"],
            linewidth=2, linestyle="--", label="Actual (test)")
    ax.plot(test_years, forecast_df["forecast"].values,
            color=color, linewidth=2.5, label=f"{model_name} forecast")
    ax.fill_between(
        test_years,
        forecast_df["lower"].values,
        forecast_df["upper"].values,
        color=color, alpha=0.15, label="95% PI",
    )
    ax.axvline(x=test.index[0] - 0.5, color="gray",
               linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP Growth (%)")
    ax.set_title(f"{model_name} — {country_name}", fontweight="bold")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side bar plots of MAE, RMSE, MAPE, and MASE across models.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Rows = models, columns = [MAE (%), RMSE (%), MAPE (%), MASE].
    figsize : tuple
    save_path : str, optional
    """
    metrics = ["MAE (%)", "RMSE (%)", "MAPE (%)", "MASE"]
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    palette = list(COLORS.values())[:len(metrics_df)]
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(
            metrics_df.index,
            metrics_df[metric],
            color=palette,
            edgecolor="black",
            alpha=0.8,
        )
        ax.set_title(metric, fontweight="bold")
        ax.set_xticklabels(metrics_df.index, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * ax.get_ylim()[1],
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle("Cross-Country Average Forecast Accuracy — All Models",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_dm_heatmap(
    dm_pvalues: pd.DataFrame,
    figsize: Tuple[int, int] = (7, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap of Diebold-Mariano p-values for all model pairs.

    Cells are coloured by significance: green (p < 0.05) = significant
    difference; red = no significant difference.

    Parameters
    ----------
    dm_pvalues : pd.DataFrame
        Square p-value matrix (models × models).
    figsize : tuple
    save_path : str, optional
    """
    mask = np.eye(len(dm_pvalues), dtype=bool)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        dm_pvalues.astype(float),
        annot=True, fmt=".3f",
        cmap="RdYlGn_r",
        vmin=0, vmax=0.1,
        mask=mask,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "p-value"},
    )
    ax.set_title("Diebold-Mariano Test p-values\n(H\u2080: equal predictive accuracy)",
                 fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_forecast_all_models(
    train: pd.Series,
    test: pd.Series,
    forecasts: Dict[str, pd.DataFrame],
    country_name: str,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Overlay all four model forecasts on a single plot for one country.

    Parameters
    ----------
    train : pd.Series
    test : pd.Series
    forecasts : dict
        {model_name: forecast_df} where forecast_df has [forecast, lower, upper].
    country_name : str
    figsize : tuple
    save_path : str, optional
    """
    model_color_map = {
        "ARIMA": COLORS["arima"],
        "ETS": COLORS["ets"],
        "Prophet": COLORS["prophet"],
        "LSTM": COLORS["lstm"],
    }
    test_years = test.index.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train.index, train.values, color=COLORS["actual"],
            linewidth=2, label="Train")
    ax.plot(test.index, test.values, color=COLORS["actual"],
            linewidth=2.5, linestyle="--", marker="o", markersize=5, label="Actual")

    for model_name, fdf in forecasts.items():
        color = model_color_map.get(model_name, "#888888")
        ax.plot(test_years, fdf["forecast"].values,
                color=color, linewidth=2, label=model_name)
        ax.fill_between(test_years, fdf["lower"].values, fdf["upper"].values,
                        color=color, alpha=0.08)

    ax.axvline(x=test.index[0] - 0.5, color="gray",
               linestyle="--", linewidth=1.5, alpha=0.6, label="Train/Test split")
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP Growth (%)")
    ax.set_title(f"All Model Forecasts — {country_name}", fontweight="bold")
    ax.legend(loc="best", frameon=True, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
