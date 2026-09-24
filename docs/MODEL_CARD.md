# Model Card

## Objective
Forecast daily SKU-level demand for each store and translate forecasts into inventory planning signals.

## Model
`XGBRegressor` with time-series lag features and calendar/promotion/price variables.

## Evaluation design
A chronological split is used: the first 80% of available dates are training data and the final 20% are held out for testing. This is more appropriate than a random split for forecasting because it preserves temporal order.

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

The default `z=1.65` approximates a 95% one-sided service level under a normal-demand assumption.

## Limitations
- The dataset is synthetic and designed for portfolio demonstration, not deployment.
- Lead times are simulated rather than supplier-derived.
- The inventory formula assumes approximately stationary forecast uncertainty and does not model order costs, holding costs, minimum order quantities, or supplier capacity.
- MAPE can behave poorly when true demand is near zero; MAE and RMSE are included for that reason.
