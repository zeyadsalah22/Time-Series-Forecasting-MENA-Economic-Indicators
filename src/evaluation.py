"""
Forecast evaluation utilities.
Implements MAE, RMSE, MAPE, MASE, and the Diebold-Mariano test
(with Harvey-Leybourne-Newbold small-sample correction).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Point accuracy metrics
# ---------------------------------------------------------------------------

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (%).

    Observations where |actual| < eps are excluded to avoid division by zero.
    """
    mask = np.abs(actual) > eps
    return float(100 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    train: np.ndarray,
) -> float:
    """
    Mean Absolute Scaled Error.

    Scales MAE by the in-sample MAE of the naive one-step-ahead forecast
    (random walk benchmark).

    Parameters
    ----------
    actual : np.ndarray
        Test-set observations.
    predicted : np.ndarray
        Test-set forecasts.
    train : np.ndarray
        Full training series (used to compute the naive benchmark scale).

    Returns
    -------
    float
        MASE value (< 1 means the model beats the naive benchmark).
    """
    naive_errors = np.abs(np.diff(train))
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    if scale == 0:
        scale = 1.0
    return float(np.mean(np.abs(actual - predicted)) / scale)


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    train: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, and MASE for a single model.

    Parameters
    ----------
    actual : np.ndarray
        Test-set observations.
    predicted : np.ndarray
        Test-set forecasts.
    train : np.ndarray
        Training series (for MASE scaling).

    Returns
    -------
    dict
        Keys: MAE, RMSE, MAPE, MASE.
    """
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "MASE": mase(actual, predicted, train),
    }


def aggregate_metrics(
    per_country_metrics: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Aggregate per-country per-model metrics into a summary table.

    Parameters
    ----------
    per_country_metrics : dict
        Nested dict: {model_name: {country_code: {metric: value}}}.

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = metrics (cross-country averages).
    """
    rows = []
    for model_name, country_dict in per_country_metrics.items():
        model_metrics: Dict[str, List[float]] = {"MAE": [], "RMSE": [], "MAPE": [], "MASE": []}
        for country, metrics in country_dict.items():
            for k in model_metrics:
                model_metrics[k].append(metrics[k])
        rows.append({
            "Model": model_name,
            "MAE (%)": round(np.mean(model_metrics["MAE"]), 2),
            "RMSE (%)": round(np.mean(model_metrics["RMSE"]), 2),
            "MAPE (%)": round(np.mean(model_metrics["MAPE"]), 2),
            "MASE": round(np.mean(model_metrics["MASE"]), 3),
        })
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> Dict[str, float]:
    """
    Diebold-Mariano test for equal predictive accuracy with
    Harvey-Leybourne-Newbold (1997) small-sample correction.

    H0: E[d_t] = 0 (equal expected loss)
    H1: E[d_t] != 0 (unequal expected loss)

    Parameters
    ----------
    e1, e2 : np.ndarray
        Forecast errors from model 1 and model 2 respectively
        (e = actual - predicted).
    h : int
        Forecast horizon (set to 1 for one-step-ahead comparisons).
    loss : str
        Loss function: 'squared' (MSE-based) or 'absolute' (MAE-based).

    Returns
    -------
    dict
        dm_stat : DM test statistic (HLN-corrected)
        p_value : two-sided p-value
        significant : bool (p < 0.05)
        better_model : 1 if model 1 is better, 2 if model 2 is better
    """
    n = len(e1)

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d_bar = np.mean(d)

    # Newey-West autocovariance estimate up to lag (h-1)
    gamma = [np.mean((d - d_bar) * (np.roll(d, k) - d_bar)) for k in range(h)]
    variance = (gamma[0] + 2 * sum(gamma[1:])) / n

    dm_stat_raw = d_bar / np.sqrt(max(variance, 1e-10))

    # HLN correction factor
    hlc = np.sqrt(
        (n + 1 - 2 * h + h * (h - 1) / n) / n
    )
    dm_stat = dm_stat_raw * hlc

    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "better_model": 1 if d_bar > 0 else 2,
    }


def pairwise_dm_tests(
    forecasts: Dict[str, np.ndarray],
    actual: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> pd.DataFrame:
    """
    Run Diebold-Mariano tests for all pairs of models.

    Parameters
    ----------
    forecasts : dict
        {model_name: forecast_array} for each model.
    actual : np.ndarray
        Observed test values.
    h : int
        Forecast horizon.
    loss : str
        Loss function for DM test.

    Returns
    -------
    pd.DataFrame
        p-value matrix (rows = model 1, columns = model 2).
        A significant p-value (< 0.05) in cell (i, j) means model i and j
        have statistically different accuracy.
    """
    model_names = list(forecasts.keys())
    errors = {name: actual - forecasts[name] for name in model_names}
    n = len(model_names)

    p_matrix = pd.DataFrame(
        np.full((n, n), np.nan),
        index=model_names,
        columns=model_names,
    )

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                continue
            result = diebold_mariano_test(errors[m1], errors[m2], h=h, loss=loss)
            p_matrix.loc[m1, m2] = round(result["p_value"], 4)

    return p_matrix
