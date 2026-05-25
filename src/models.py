"""
Forecasting model implementations for the MENA GDP growth study.
Covers ARIMA (AIC-based order selection), Exponential Smoothing (ETS),
Facebook Prophet, and LSTM with MC-Dropout uncertainty quantification.
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def select_arima_order(
    series: pd.Series,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
) -> Tuple[int, int, int]:
    """
    Select ARIMA(p, d, q) order by minimising AIC over an exhaustive grid.

    Parameters
    ----------
    series : pd.Series
        Training time series.
    max_p, max_d, max_q : int
        Upper bounds for the AR, integration, and MA orders.

    Returns
    -------
    Tuple[int, int, int]
        Best (p, d, q) order.
    """
    best_aic = np.inf
    best_order = (1, 1, 1)

    for p, d, q in itertools.product(
        range(max_p + 1), range(max_d + 1), range(max_q + 1)
    ):
        if p == 0 and q == 0:
            continue
        try:
            model = SARIMAX(series, order=(p, d, q), trend="n")
            result = model.fit(disp=False)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, d, q)
        except Exception:
            continue

    return best_order


def fit_arima(
    series: pd.Series,
    order: Optional[Tuple[int, int, int]] = None,
) -> sm.tsa.statespace.sarimax.SARIMAXResultsWrapper:
    """
    Fit an ARIMA model, optionally with automatic order selection.

    Parameters
    ----------
    series : pd.Series
        Training time series.
    order : tuple, optional
        (p, d, q). If None, order is selected via AIC grid search.

    Returns
    -------
    Fitted SARIMAX results object.
    """
    if order is None:
        order = select_arima_order(series)

    model = SARIMAX(series, order=order, trend="n")
    return model.fit(disp=False)


def forecast_arima(
    result: sm.tsa.statespace.sarimax.SARIMAXResultsWrapper,
    steps: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Generate out-of-sample ARIMA forecasts with prediction intervals.

    Parameters
    ----------
    result : fitted SARIMAX results
    steps : int
        Number of periods to forecast.
    alpha : float
        Significance level for prediction intervals (0.05 → 95% PI).

    Returns
    -------
    pd.DataFrame
        Columns: [forecast, lower, upper].
    """
    pred = result.get_forecast(steps=steps)
    summary = pred.summary_frame(alpha=alpha)
    return pd.DataFrame({
        "forecast": summary["mean"].values,
        "lower": summary["mean_ci_lower"].values,
        "upper": summary["mean_ci_upper"].values,
    })


# ---------------------------------------------------------------------------
# Exponential Smoothing (ETS)
# ---------------------------------------------------------------------------

def fit_ets(series: pd.Series) -> ExponentialSmoothing:
    """
    Fit a Holt-Winters Exponential Smoothing model with automatic
    additive/multiplicative selection based on AIC.

    Parameters
    ----------
    series : pd.Series
        Training time series.

    Returns
    -------
    Fitted ExponentialSmoothing results object.
    """
    best_aic = np.inf
    best_fit = None

    for trend in ["add", "mul", None]:
        for damped in ([True, False] if trend is not None else [False]):
            try:
                model = ExponentialSmoothing(
                    series,
                    trend=trend,
                    damped_trend=damped,
                    seasonal=None,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit
            except Exception:
                continue

    return best_fit


def forecast_ets(
    fit,
    steps: int,
    series: pd.Series,
    alpha: float = 0.05,
    n_simulations: int = 1000,
) -> pd.DataFrame:
    """
    Generate ETS forecasts with bootstrap prediction intervals.

    Parameters
    ----------
    fit : fitted ExponentialSmoothing object
    steps : int
        Number of periods to forecast.
    series : pd.Series
        Original training series (used for residual bootstrap).
    alpha : float
        Significance level.
    n_simulations : int
        Number of bootstrap replications.

    Returns
    -------
    pd.DataFrame
        Columns: [forecast, lower, upper].
    """
    point_forecast = fit.forecast(steps)

    # Residual bootstrap for prediction intervals
    residuals = series.values - fit.fittedvalues.values
    residuals = residuals[~np.isnan(residuals)]

    sim_forecasts = np.zeros((n_simulations, steps))
    rng = np.random.default_rng(42)
    for i in range(n_simulations):
        noise = rng.choice(residuals, size=steps, replace=True)
        sim_forecasts[i] = point_forecast.values + noise

    lower = np.quantile(sim_forecasts, alpha / 2, axis=0)
    upper = np.quantile(sim_forecasts, 1 - alpha / 2, axis=0)

    return pd.DataFrame({
        "forecast": point_forecast.values,
        "lower": lower,
        "upper": upper,
    })


# ---------------------------------------------------------------------------
# Facebook Prophet
# ---------------------------------------------------------------------------

def fit_prophet(
    train_df: pd.DataFrame,
    yearly_seasonality: bool = True,
    changepoint_prior_scale: float = 0.05,
) -> Prophet:
    """
    Fit a Facebook Prophet model.

    Parameters
    ----------
    train_df : pd.DataFrame
        Prophet-format DataFrame with columns [ds, y].
    yearly_seasonality : bool
        Whether to include yearly seasonality.
    changepoint_prior_scale : float
        Flexibility of the trend changepoints.

    Returns
    -------
    Fitted Prophet model.
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.95,
    )
    model.fit(train_df)
    return model


def forecast_prophet(
    model: Prophet,
    steps: int,
    freq: str = "YS",
) -> pd.DataFrame:
    """
    Generate Prophet forecasts with 95% prediction intervals.

    Parameters
    ----------
    model : fitted Prophet model
    steps : int
        Number of future periods.
    freq : str
        Pandas frequency string ('YS' = year start).

    Returns
    -------
    pd.DataFrame
        Columns: [forecast, lower, upper], indexed 0..steps-1.
    """
    future = model.make_future_dataframe(periods=steps, freq=freq)
    forecast = model.predict(future)
    out = forecast[["yhat", "yhat_lower", "yhat_upper"]].tail(steps)
    return pd.DataFrame({
        "forecast": out["yhat"].values,
        "lower": out["yhat_lower"].values,
        "upper": out["yhat_upper"].values,
    })


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def build_lstm(
    lookback: int = 5,
    units_1: int = 64,
    units_2: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build a two-layer LSTM model with MC-Dropout.

    Parameters
    ----------
    lookback : int
        Input sequence length.
    units_1, units_2 : int
        LSTM units in the first and second layers.
    dropout_rate : float
        Dropout fraction applied after each LSTM layer (also active at
        inference for MC-Dropout uncertainty estimation).
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    Compiled Keras Model.
    """
    model = Sequential([
        LSTM(units_1, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout_rate),
        LSTM(units_2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    return model


def train_lstm(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    batch_size: int = 8,
    validation_split: float = 0.15,
) -> tf.keras.callbacks.History:
    """
    Train the LSTM model with early stopping.

    Parameters
    ----------
    model : compiled Keras model
    X_train : np.ndarray
        Shape (n_samples, lookback, 1).
    y_train : np.ndarray
        Shape (n_samples,).
    epochs : int
        Maximum training epochs.
    batch_size : int
    validation_split : float
        Fraction of training data used for validation monitoring.

    Returns
    -------
    Keras History object.
    """
    early_stop = EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0,
    )
    return history


def forecast_lstm(
    model: tf.keras.Model,
    last_sequence: np.ndarray,
    steps: int,
    scaler: MinMaxScaler,
    n_mc_samples: int = 200,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Multi-step LSTM forecast using MC-Dropout for prediction intervals.

    The model is called n_mc_samples times with dropout active (training=True)
    to obtain a distribution of predictions at each future step.

    Parameters
    ----------
    model : trained Keras model
    last_sequence : np.ndarray
        Shape (lookback,) — the final observed values (scaled).
    steps : int
        Number of forecast steps.
    scaler : MinMaxScaler
        Fitted scaler for inverse-transforming predictions.
    n_mc_samples : int
        Number of MC-Dropout forward passes per step.
    alpha : float
        Significance level for prediction intervals.

    Returns
    -------
    pd.DataFrame
        Columns: [forecast, lower, upper] in original (unscaled) units.
    """
    lookback = last_sequence.shape[0]
    all_samples = np.zeros((n_mc_samples, steps))

    for s in range(n_mc_samples):
        seq = last_sequence.copy()
        preds = []
        for _ in range(steps):
            x = seq.reshape(1, lookback, 1)
            pred = model(x, training=True).numpy().flatten()[0]
            preds.append(pred)
            seq = np.append(seq[1:], pred)
        all_samples[s] = preds

    # Inverse-transform each MC sample
    all_original = scaler.inverse_transform(all_samples.T).T

    point = np.mean(all_original, axis=0)
    lower = np.quantile(all_original, alpha / 2, axis=0)
    upper = np.quantile(all_original, 1 - alpha / 2, axis=0)

    return pd.DataFrame({
        "forecast": point,
        "lower": lower,
        "upper": upper,
    })
