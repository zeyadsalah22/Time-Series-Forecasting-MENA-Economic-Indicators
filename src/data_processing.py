"""
Data processing utilities for the MENA time series forecasting project.
Handles World Bank API retrieval, panel construction, train/test splitting,
and sequence preparation for LSTM estimation.
"""

import pandas as pd
import numpy as np
import wbgapi as wb
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler


MENA_COUNTRIES: Dict[str, str] = {
    "DZA": "Algeria",
    "BHR": "Bahrain",
    "EGY": "Egypt",
    "JOR": "Jordan",
    "KWT": "Kuwait",
    "MAR": "Morocco",
    "OMN": "Oman",
    "QAT": "Qatar",
    "SAU": "Saudi Arabia",
    "TUN": "Tunisia",
}

GDP_GROWTH_INDICATOR = "NY.GDP.MKTP.KD.ZG"
START_YEAR = 1990
END_YEAR = 2023
TRAIN_END_YEAR = 2018
TEST_START_YEAR = 2019


def fetch_gdp_growth(
    countries: Optional[List[str]] = None,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> pd.DataFrame:
    """
    Fetch annual GDP growth rates from the World Bank API.

    Parameters
    ----------
    countries : list of str, optional
        ISO-3 country codes. Defaults to the 10 MENA countries defined in
        MENA_COUNTRIES.
    start_year : int
        First year to retrieve.
    end_year : int
        Last year to retrieve.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns [country_code, country_name, year,
        gdp_growth].
    """
    if countries is None:
        countries = list(MENA_COUNTRIES.keys())

    raw = wb.data.DataFrame(
        GDP_GROWTH_INDICATOR,
        economy=countries,
        time=range(start_year, end_year + 1),
    )

    # wb returns wide format: rows = economies, columns = years
    raw = raw.reset_index()
    raw = raw.rename(columns={"economy": "country_code"})

    long = raw.melt(id_vars="country_code", var_name="year", value_name="gdp_growth")
    long["year"] = long["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    long["country_name"] = long["country_code"].map(MENA_COUNTRIES)
    long = long.dropna(subset=["gdp_growth"]).sort_values(["country_code", "year"])
    long = long.reset_index(drop=True)

    return long[["country_code", "country_name", "year", "gdp_growth"]]


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the panel and forward-fill isolated missing values within each
    country series (no more than 2 consecutive gaps).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel as returned by fetch_gdp_growth.

    Returns
    -------
    pd.DataFrame
        Cleaned panel, sorted by country_code and year.
    """
    panel = df.copy().sort_values(["country_code", "year"])

    # Forward-fill within country for isolated gaps
    panel["gdp_growth"] = (
        panel.groupby("country_code")["gdp_growth"]
        .transform(lambda s: s.interpolate(method="linear", limit=2))
    )

    return panel.reset_index(drop=True)


def get_country_series(panel: pd.DataFrame, country_code: str) -> pd.Series:
    """
    Extract a single country's GDP growth series indexed by year.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel as returned by build_panel.
    country_code : str
        ISO-3 code.

    Returns
    -------
    pd.Series
        Annual GDP growth indexed by integer year.
    """
    subset = panel[panel["country_code"] == country_code].set_index("year")
    return subset["gdp_growth"].astype(float)


def train_test_split_series(
    series: pd.Series,
    train_end: int = TRAIN_END_YEAR,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split a time series into training and test portions at a fixed year boundary.

    Parameters
    ----------
    series : pd.Series
        Time series indexed by integer year.
    train_end : int
        Last year included in the training set (inclusive).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (train, test) series.
    """
    train = series[series.index <= train_end]
    test = series[series.index > train_end]
    return train, test


def prepare_prophet_df(series: pd.Series) -> pd.DataFrame:
    """
    Convert a year-indexed series to the Prophet input format (ds, y).

    Parameters
    ----------
    series : pd.Series
        Time series indexed by integer year.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [ds, y] where ds is a datetime (Jan 1 of each
        year).
    """
    df = pd.DataFrame({
        "ds": pd.to_datetime(series.index.astype(str) + "-01-01"),
        "y": series.values,
    })
    return df.reset_index(drop=True)


def normalize_series(
    train: pd.Series,
    test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Min-max scale a series using statistics from the training set only.

    Parameters
    ----------
    train : pd.Series
        Training portion of the series.
    test : pd.Series
        Test portion of the series.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, MinMaxScaler]
        (train_scaled, test_scaled, fitted_scaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test.values.reshape(-1, 1)).flatten()
    return train_scaled, test_scaled, scaler


def create_lstm_sequences(
    data: np.ndarray,
    lookback: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping (X, y) sequence pairs for supervised LSTM training.

    Parameters
    ----------
    data : np.ndarray
        1-D array of scaled values.
    lookback : int
        Number of past time steps used as input features.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X of shape (n_samples, lookback, 1) and y of shape (n_samples,).
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y
