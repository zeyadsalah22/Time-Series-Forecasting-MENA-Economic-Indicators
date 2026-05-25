# Data

## Source

World Bank Open Data API via `wbgapi`.

**Indicator**: GDP growth, annual % — `NY.GDP.MKTP.KD.ZG`

## Countries

| Code | Country |
|------|---------|
| DZA | Algeria |
| BHR | Bahrain |
| EGY | Egypt |
| JOR | Jordan |
| KWT | Kuwait |
| MAR | Morocco |
| OMN | Oman |
| QAT | Qatar |
| SAU | Saudi Arabia |
| TUN | Tunisia |

## Coverage

1990–2023 (34 annual observations per country, 340 country-year observations total)

## Directory

- `raw/` — unmodified API downloads, one file per fetch
- `processed/` — cleaned panel (`mena_gdp_panel.csv`) and per-country series used by the model notebooks

## Reproducibility

Run `01_data_collection.ipynb` to regenerate all data files from the World Bank API. No manual downloads required.
