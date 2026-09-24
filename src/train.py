from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from features import FEATURE_COLUMNS, build_features
from inventory import build_inventory_recommendations

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "retail_sales.csv"
PROCESSED = ROOT / "data" / "processed" / "modeling_data.csv"
MODEL_PATH = ROOT / "models" / "demand_model.joblib"
OUTPUTS = ROOT / "outputs"
TARGET = "units_sold"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA}. Run src/generate_data.py first.")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_DATA, parse_dates=["date"])
    df = build_features(raw)
    df.to_csv(PROCESSED, index=False)

    unique_dates = np.sort(df["date"].unique())
    split_index = int(len(unique_dates) * 0.80)
    split_date = pd.Timestamp(unique_dates[split_index])
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()

    model = XGBRegressor(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    pred = np.clip(model.predict(test[FEATURE_COLUMNS]), 0, None)

    metrics = {
        "MAE": round(float(mean_absolute_error(test[TARGET], pred)), 3),
        "RMSE": round(float(mean_squared_error(test[TARGET], pred) ** 0.5), 3),
        "MAPE_percent": round(mape(test[TARGET].to_numpy(), pred), 3),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split_date": split_date.date().isoformat(),
    }

    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS, "target": TARGET, "split_date": str(split_date.date())},
        MODEL_PATH,
    )

    results = test[[
        "date", "store_id", "sku_id", "category", "region", "price", "promotion",
        "holiday", "units_sold", "lead_time_days",
    ]].copy()
    results["forecast_units"] = np.round(pred, 2)
    results["absolute_error"] = np.round(np.abs(results[TARGET] - pred), 2)
    results.to_csv(OUTPUTS / "forecast_results.csv", index=False)

    inventory = build_inventory_recommendations(results)
    inventory.to_csv(OUTPUTS / "inventory_recommendations.csv", index=False)

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUTS / "feature_importance.csv", index=False)

    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
