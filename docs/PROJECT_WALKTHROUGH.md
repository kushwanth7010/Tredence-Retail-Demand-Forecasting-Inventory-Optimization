# Project Walkthrough

## 1. Problem Statement

The project models a common retail planning problem: future demand is uncertain, but replenishment decisions must be made before that demand is observed. The solution therefore has two layers: a forecasting layer that estimates future store-SKU demand, and a decision layer that translates those forecasts into inventory planning signals.

The prediction target is `units_sold` at the daily store-SKU level.

## 2. Reproducible Dataset

`src/generate_data.py` creates the raw dataset from a fixed NumPy random seed (`42`). The generator creates 731 dates from 2024-01-01 to 2025-12-31, 5 stores, 20 SKUs, 4 product categories, 5 store regions, and 73,100 total observations.

Demand is generated with interpretable effects rather than pure random noise. These include store-level demand multipliers, product-specific baseline demand, yearly seasonality, an 8% weekend uplift, a gradual time trend, promotions, selected holiday effects, daily price variation, product-specific price sensitivity, and random noise.

## 3. Data Quality Checks

`tests/test_data_quality.py` checks that the dataset exists, contains the required schema, has exactly 73,100 rows, contains no missing required values, has positive prices and non-negative demand, and contains exactly 5 stores and 20 SKUs.

## 4. Feature Engineering

`src/features.py` sorts observations by store, SKU and date before creating features.

### Lag features

- `lag_1`: demand one day earlier
- `lag_7`: demand seven days earlier
- `lag_14`: demand fourteen days earlier
- `lag_28`: demand twenty-eight days earlier

### Rolling features

- `rolling_mean_7`
- `rolling_mean_28`

The rolling calculations use `shift(1)` before the rolling mean is calculated, so the current target is never included in the features used to predict that same day. `tests/test_features.py` verifies this behavior on a known sequential dataset.

### Calendar and business features

The model also receives price, promotion, holiday, lead time, day of week, month, ISO week number, and a weekend indicator.

## 5. Chronological Train/Test Split

`src/train.py` preserves time order instead of randomly splitting observations. Earlier dates form the training set and the latest 20% of dates form the held-out test set.

Current reproducible split:

- split date: **2025-08-13**
- training rows: **56,200**
- test rows: **14,100**

This better represents a forecasting task because future observations are not allowed into model training.

## 6. XGBoost Model

The model is an `XGBRegressor` configured with 350 estimators, maximum depth 6, learning rate 0.05, 90% row subsampling, 90% feature subsampling, squared-error objective, and random seed 42.

XGBoost is suitable for this feature-based forecasting problem because the data is tabular and the relationship between demand history, seasonality, price, promotions and other factors is nonlinear.

The trained model and metadata are saved in `models/demand_model.joblib`.

## 7. Validated Model Results

| Metric | Value |
|---|---:|
| MAE | 4.192 units |
| RMSE | 5.260 units |
| MAPE | 11.555% |

**MAE** is the average absolute forecast error in units. An MAE of 4.192 means the prediction differs from observed daily demand by about 4.2 units on average across the held-out test records.

**RMSE** is 5.260 units and penalizes large forecast misses more strongly than MAE.

**MAPE** is 11.555%. The implementation uses a denominator floor of 1 to prevent division-by-zero behavior when demand is zero or very small.

These metrics are generated automatically and saved in `outputs/metrics.json`.

## 8. Forecast Results

`outputs/forecast_results.csv` contains the held-out observations together with the actual demand, model forecast and absolute error, plus the store, SKU, category, region, price, promotion, holiday and lead-time information needed for downstream analysis.

## 9. Explainability

`src/explain.py` loads the trained model and uses SHAP `TreeExplainer` on a reproducible sample of up to 2,000 observations.

The project saves:

- `outputs/shap_summary.png` for SHAP-based global explanation
- `outputs/feature_importance.csv` for XGBoost feature importance

The goal is to make the model interpretable rather than treating it as a prediction-only black box.

## 10. Inventory Optimization

`src/inventory.py` groups forecast results by store and SKU and calculates average forecast demand, forecast standard deviation, median lead time, safety stock and reorder point.

The formulas are:

```text
Safety Stock = z × Forecast Demand Standard Deviation × sqrt(Lead Time)
```

```text
Reorder Point = Average Daily Forecast × Lead Time + Safety Stock
```

The project uses `z = 1.65`, which approximates a 95% one-sided service level under a normal-demand assumption.

`tests/test_inventory.py` checks that safety stock is non-negative and that reorder point is at least as large as expected lead-time demand.

## 11. SQL Layer

`sql/schema.sql` defines the retail sales table and an index on `(store_id, sku_id, date)`.

`sql/analysis.sql` demonstrates:

1. daily store-SKU demand aggregation;
2. promotion lift using conditional aggregation;
3. seven-day rolling demand using a SQL window function;
4. regional SKU ranking using CTEs and `ROW_NUMBER()`.

## 12. Streamlit Dashboard

`app/app.py` converts the generated outputs into an interactive interface with model-error KPI cards, store and SKU filters, actual-versus-forecast demand, inventory recommendations, global feature importance and recent forecast records.

Run it with:

```bash
streamlit run app/app.py
```

## 13. End-to-End Execution Flow

Running:

```bash
python src/pipeline.py
```

executes:

```text
generate_data.py
      ↓
raw retail_sales.csv
      ↓
train.py
      ├─ feature engineering
      ├─ chronological split
      ├─ XGBoost training
      ├─ model metrics
      ├─ forecast results
      ├─ inventory recommendations
      └─ feature importance
      ↓
explain.py
      ↓
SHAP summary plot
```

The repository is therefore reproducible from source code rather than depending only on pre-generated artifacts.

## 14. GitHub Actions Workflow

The automated workflow checks out the repository, installs dependencies, runs the full pipeline, runs Pytest, and commits the reproducible data, trained model and generated outputs when they change.

The current workflow completed successfully.

## 15. Assumptions and Limitations

This is a complete portfolio data science system, not a production inventory deployment. Important limitations include:

- the dataset is synthetic;
- lead times are simulated;
- the project does not estimate calibrated probabilistic forecast intervals;
- the reorder-point logic does not optimize ordering cost or holding cost;
- supplier constraints, stock on hand and minimum order quantities are not modeled;
- rolling-origin backtesting and formal hyperparameter tuning can be added for a stronger production evaluation.

## 16. Interview Explanation

A concise explanation is:

> I built an end-to-end retail demand forecasting and inventory optimization system. I created a reproducible daily store-SKU dataset, engineered leakage-safe lag and rolling-demand features, preserved temporal order with a chronological split, and trained an XGBoost regressor. On the held-out period the model achieved an MAE of 4.192 units, RMSE of 5.260 units and MAPE of 11.555%. I used SHAP to explain model behavior and converted forecasts into safety-stock and reorder-point recommendations. I also implemented advanced SQL analysis, automated tests, GitHub Actions and a Streamlit dashboard.

## 17. Follow-Up Questions to Prepare

Be ready to explain why chronological validation is preferable to random splitting, how `shift(1)` prevents leakage, why MAE/RMSE/MAPE are reported together, why XGBoost was selected, what SHAP values represent, how safety stock differs from reorder point, why lead time appears in the inventory formulas, and what additions would be required before production deployment.
