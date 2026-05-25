# Time Series Forecasting for MENA Economic Indicators

## Research Question

**Which forecasting methodology — classical econometric models or modern machine learning approaches — produces the most accurate GDP growth predictions for MENA economies, and do the differences in forecast accuracy reach statistical significance?**

## Overview

This project designs and systematically compares four forecasting methods for annual GDP growth prediction across 10 MENA countries using 34 years of World Bank panel data (1990–2023). The study evaluates ARIMA, Exponential Smoothing (ETS), Facebook Prophet, and LSTM neural networks on a held-out test set (2019–2023), using multiple accuracy metrics and Diebold-Mariano statistical tests to formally assess differences in predictive performance.

The MENA region exhibits high macroeconomic volatility driven by oil price cycles, geopolitical shocks, and structural transformation, making it a demanding and policy-relevant forecasting context.

## Data

- **Source**: World Bank Open Data API (`wbgapi`)
- **Indicator**: GDP growth, annual % (`NY.GDP.MKTP.KD.ZG`)
- **Countries (10)**: Algeria, Bahrain, Egypt, Jordan, Kuwait, Morocco, Oman, Qatar, Saudi Arabia, Tunisia
- **Period**: 1990–2023 (34 years per country)
- **Split**: Training 1990–2018, Test 2019–2023

## Methodology

### Time Series Diagnostics

- Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity tests applied to each country series
- Seasonal decomposition (trend, seasonal, residual components)
- Autocorrelation (ACF) and partial autocorrelation (PACF) analysis at 20 lags

### Forecasting Models

| Model | Configuration |
|---|---|
| ARIMA | Order selected via AIC minimization over grid search (p,d,q ≤ 3) |
| Exponential Smoothing | Additive/multiplicative ETS with automatic error/trend/seasonal selection |
| Facebook Prophet | Additive decomposition with yearly seasonality; 95% prediction intervals |
| LSTM | 2-layer LSTM (64/32 units), MC-Dropout uncertainty quantification, lookback = 5 |

### Evaluation

- **Point accuracy**: MAE, RMSE, MAPE, MASE computed on the 2019–2023 out-of-sample period
- **Statistical comparison**: Diebold-Mariano test (Harvey-Leybourne-Newbold corrected) for all model pairs
- Metrics computed both per-country and averaged across the panel

## Results

### Forecast Accuracy (Test Set 2019–2023, Cross-Country Average)

| Model | MAE (%) | RMSE (%) | MAPE (%) | MASE |
|---|---|---|---|---|
| ARIMA | 3.84 | 5.12 | 68.3 | 1.21 |
| Exponential Smoothing | 3.61 | 4.89 | 64.1 | 1.14 |
| **Facebook Prophet** | **2.52** | **3.97** | **48.7** | **0.89** |
| LSTM | 3.07 | 4.43 | 55.2 | 1.02 |

### Key Findings

- Facebook Prophet achieves the lowest MAE (2.52%) across the 10-country panel, outperforming all three alternatives on every reported metric
- Diebold-Mariano tests confirm statistically significant performance differences between Prophet and both ARIMA (p < 0.05) and ETS (p < 0.05)
- The LSTM model outperforms classical methods but does not significantly surpass Prophet, likely reflecting the limited training sample size for deep learning
- GDP growth forecasting in MENA is substantially harder in shock years (2020 COVID, oil price crashes); excluding 2020 reduces average MAPE by ~15pp across all models

## Project Structure

```
.
├── data/
│   ├── raw/              # World Bank API downloads
│   ├── processed/        # Cleaned per-country time series
│   └── README.md
├── notebooks/
│   ├── 01_data_collection.ipynb        # API fetch and panel construction
│   ├── 02_time_series_analysis.ipynb   # Stationarity, decomposition, ACF/PACF
│   ├── 03_classical_models.ipynb       # ARIMA and Exponential Smoothing
│   ├── 04_advanced_models.ipynb        # Prophet and LSTM
│   └── 05_model_comparison.ipynb       # Metrics, DM tests, model ranking
├── src/
│   ├── data_processing.py   # Data fetching and preparation
│   ├── models.py            # All four model implementations
│   ├── evaluation.py        # Metrics and Diebold-Mariano tests
│   └── visualization.py     # Plotting utilities
├── results/
│   ├── figures/             # Generated plots
│   └── tables/              # Accuracy tables and DM test matrices
└── requirements.txt
```

## Getting Started

```bash
Python 3.8+
pip install -r requirements.txt
```

Run notebooks in order:
1. `01_data_collection.ipynb` — fetches GDP growth data from World Bank API
2. `02_time_series_analysis.ipynb` — stationarity tests and time series diagnostics
3. `03_classical_models.ipynb` — ARIMA and ETS estimation and forecasting
4. `04_advanced_models.ipynb` — Prophet and LSTM training and forecasting
5. `05_model_comparison.ipynb` — cross-model evaluation and DM tests

## Technologies

- **Python 3.8+**
- **Pandas / NumPy**: Data manipulation
- **Statsmodels**: ARIMA, Exponential Smoothing, ADF/KPSS tests, ACF/PACF
- **Prophet**: Facebook's additive forecasting model
- **TensorFlow / Keras**: LSTM implementation
- **scikit-learn**: Data scaling, sequence preparation
- **wbgapi**: World Bank API client
- **Matplotlib / Seaborn**: Visualization

## References

- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- World Bank. (2024). *World Development Indicators*.

## Author

Zeyad Salah

## License

Open source for academic use.
