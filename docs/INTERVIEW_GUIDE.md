# Interview Guide

## 30-second explanation
I built an end-to-end retail demand forecasting and inventory optimization system. I generated a reproducible retail dataset, engineered leakage-safe lag and rolling features, used a chronological train/test split, trained an XGBoost regressor, evaluated it with MAE, RMSE and MAPE, explained model behavior with SHAP, and converted forecasts into safety-stock and reorder-point recommendations. I also added advanced SQL analysis and a Streamlit dashboard.

## Why XGBoost?
XGBoost handles non-linear interactions, mixed numerical signals, seasonality proxies and promotion effects effectively without requiring the strict assumptions of a classical linear model. It also provides strong tabular-data performance and integrates well with SHAP explainability.

## Why not random train/test split?
A random split can leak future demand patterns into training. A chronological split better simulates the production task: train on past data and forecast unseen future dates.

## How did you prevent leakage?
All lag and rolling features are computed after grouping by store and SKU, and rolling windows use `shift(1)` before aggregation. Therefore the target for the current day never appears in its own features.

## What do the metrics mean?
- MAE is the average absolute error in units and is easy to interpret.
- RMSE gives more weight to large errors.
- MAPE expresses error as a percentage but is unstable around zero, so it is not used alone.

## What business action comes from the model?
Forecasts are converted into reorder points and safety stock. That gives an operations team a decision-oriented output instead of only a prediction score.

## What would you improve for production?
Use real transaction and supplier lead-time data, validate multiple forecasting horizons, add backtesting, forecast intervals, hyperparameter tuning, drift monitoring, automated retraining, and inventory cost optimization.
