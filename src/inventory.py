from __future__ import annotations

import numpy as np
import pandas as pd


def build_inventory_recommendations(
    forecast_results: pd.DataFrame,
    service_level_z: float = 1.65,
) -> pd.DataFrame:
    """Calculate safety stock and reorder point from forecast demand.

    Formula:
      safety_stock = z * demand_std * sqrt(lead_time)
      reorder_point = avg_daily_forecast * lead_time + safety_stock

    z=1.65 approximates a 95% one-sided service level under a normal-demand assumption.
    """
    required = {"store_id", "sku_id", "forecast_units", "lead_time_days"}
    missing = required.difference(forecast_results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    inv = forecast_results.groupby(["store_id", "sku_id"], as_index=False).agg(
        avg_daily_forecast=("forecast_units", "mean"),
        forecast_std=("forecast_units", "std"),
        lead_time_days=("lead_time_days", "median"),
    )
    inv["forecast_std"] = inv["forecast_std"].fillna(0)
    inv["safety_stock"] = (
        service_level_z * inv["forecast_std"] * np.sqrt(inv["lead_time_days"])
    )
    inv["reorder_point"] = (
        inv["avg_daily_forecast"] * inv["lead_time_days"] + inv["safety_stock"]
    )
    numeric = ["avg_daily_forecast", "forecast_std", "safety_stock", "reorder_point"]
    inv[numeric] = inv[numeric].round(2)
    return inv
