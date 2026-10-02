# Model Card

## Objective
Forecast daily SKU-level demand for each store and translate forecasts into inventory planning signals.

## Model
`XGBRegressor` with time-series lag features and calendar/promotion/price variables.

## Evaluation design
A chronological split is used: the first 80% of available dates are training data and the final 20% are held out for testing. Features for each held-out day use *observed* demand from preceding days, including earlier days in the test period. Therefore the reported metrics measure a **rolling one-step-ahead backtest** (actual demand through day t−1 is assumed available for predicting day t), **not** a multi-step forecast made once at the train/test cutoff. A true multi-day deployment needs recursive or direct horizon-specific features and separate multi-horizon evaluation.

## Metrics
The pipeline reports:
- MAE: average absolute forecast error in units
- RMSE: error metric that penalizes large misses more heavily
- MAPE: average absolute percentage error, protected against division by zero with a denominator floor of 1

Actual generated values are saved in `outputs/metrics.json` after training.

## Explainability
`src/explain.py` uses SHAP TreeExplainer to produce `outputs/shap_summary.png`, showing how the model's input features influence predictions across a sample of observations.

## Inventory logic
The project uses:
- `safety_stock = z × demand_std × sqrt(lead_time)`
- `reorder_point = average_daily_forecast × lead_time + safety_stock`

The default `z=1.65` is the approximate 95th percentile of a standard normal variable; it does **not** establish an achieved 95% service level. The implementation uses dispersion of predicted daily demand over the retrospective test window as a variability proxy, not measured forecast-error variance.

## Limitations
- The dataset is synthetic and designed for portfolio demonstration, not deployment.
- Lead times are simulated rather than supplier-derived.
- Recommendations are retrospective illustrations based on backtest predictions, not deployed forward-looking purchase orders.
- The inventory formula assumes approximately stationary forecast uncertainty and does not model order costs, holding costs, minimum order quantities, or supplier capacity.
- MAPE can behave poorly when true demand is near zero; MAE and RMSE are included for that reason.
