# AI-Driven Retail Demand Forecasting & Inventory Optimization

A complete, standalone data science portfolio project for **SKU-level demand forecasting, model explainability, and inventory decision support**.

> **Important:** This repository is an independent project. It does **not** use, depend on, or link to the AdventureWorks Sales & Customer Analytics repository. The dataset here is generated specifically for this project by `src/generate_data.py`.

## Project Goal

Retail teams need to answer two connected questions:

1. **How much of each SKU will each store sell next?**
2. **When should inventory be replenished to reduce stockout risk without carrying unnecessary stock?**

This project builds an end-to-end workflow from raw data generation to machine learning, explainability, inventory recommendations, SQL analysis, automated tests, and an interactive dashboard.

## What the project demonstrates

- Python data engineering with Pandas and NumPy
- Advanced SQL with CTEs and window functions
- Time-series feature engineering
- XGBoost regression
- Chronological train/test validation
- MAE, RMSE and MAPE evaluation
- SHAP model explainability
- Safety-stock and reorder-point calculations
- Streamlit dashboarding
- Unit/data-quality tests and GitHub Actions CI

## Architecture

```text
Synthetic Retail Data
        |
        v
Data Quality + Feature Engineering
(lags, rolling demand, calendar variables)
        |
        v
Chronological Train/Test Split
        |
        v
XGBoost Demand Forecasting
        |
        +-------------------+
        |                   |
        v                   v
Model Evaluation          SHAP Explainability
        |
        v
Inventory Optimization
(safety stock + reorder point)
        |
        v
Streamlit Decision Dashboard
```

## Dataset

The repository contains a reproducible synthetic dataset with:

- **73,100 daily observations**
- **5 stores**
- **20 SKUs**
- **2 years** of data: 2024-01-01 to 2025-12-31
- price, promotion, holiday, region, category, demand and replenishment lead time

See [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for exact definitions.

## Feature Engineering

For each store-SKU time series, the pipeline creates:

- `lag_1`, `lag_7`, `lag_14`, `lag_28`
- 7-day and 28-day prior rolling demand means
- day of week, month, ISO week number and weekend indicator
- price, promotion, holiday and lead time

All lag/rolling features are based only on prior observations to avoid target leakage.

## Model

The forecasting model is an `XGBRegressor`. The split is **chronological**, not random: the first 80% of dates are used for training and the most recent 20% are held out for testing.

Run the pipeline to reproduce the final metrics. The generated values are stored in `outputs/metrics.json`.

## Inventory Optimization

Forecasts are converted into inventory signals using:

```text
Safety Stock = z × Forecast Demand Std. Dev. × sqrt(Lead Time)
Reorder Point = Avg. Daily Forecast × Lead Time + Safety Stock
```

The default `z = 1.65` approximates a 95% one-sided service level under a normal-demand assumption.

## Repository Structure

```text
Tredence-Retail-Demand-Forecasting-Inventory-Optimization/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── raw/retail_sales.csv
│   └── processed/modeling_data.csv
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── train.py
│   ├── inventory.py
│   ├── explain.py
│   └── pipeline.py
├── sql/
│   ├── schema.sql
│   └── analysis.sql
├── app/app.py
├── models/demand_model.joblib
├── outputs/
│   ├── metrics.json
│   ├── forecast_results.csv
│   ├── inventory_recommendations.csv
│   ├── feature_importance.csv
│   └── shap_summary.png
├── tests/
│   ├── test_data_quality.py
│   ├── test_features.py
│   └── test_inventory.py
├── docs/
│   ├── DATA_DICTIONARY.md
│   ├── MODEL_CARD.md
│   └── INTERVIEW_GUIDE.md
├── notebooks/01_eda_and_modeling.ipynb
└── .github/workflows/ci.yml
```

## Run Locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python src/pipeline.py
streamlit run app/app.py
```

## Run Tests

```bash
pytest -q
```

## SQL

`sql/schema.sql` defines the retail table and index. `sql/analysis.sql` includes:

- daily SKU demand
- promotion lift
- rolling 7-day demand using a window function
- top SKUs by region using a CTE and ranking window function

## Resume-Ready Description

**AI-Driven Retail Demand Forecasting & Inventory Optimization**  
*Python | SQL | XGBoost | Time Series | Scikit-learn | SHAP*

- Built an end-to-end SKU-level demand forecasting pipeline using Python, SQL, XGBoost and leakage-safe time-series features, evaluated with MAE, RMSE and MAPE on chronologically held-out data.
- Applied SHAP explainability and inventory optimization using safety-stock and reorder-point logic to translate model predictions into actionable retail decisions.

## Notes for Interviewers

This is intentionally a complete portfolio system rather than a single modeling notebook. It demonstrates the full reasoning chain from business problem and data design to modeling, validation, explainability, operational decision logic, testing and presentation.

For a deeper project walkthrough, see [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md).
