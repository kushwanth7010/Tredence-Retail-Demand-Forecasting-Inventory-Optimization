from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "price", "promotion", "holiday", "lead_time_days", "day_of_week", "month",
    "week_of_year", "weekend", "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe time-series features using only prior demand values."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)

    group = work.groupby(["store_id", "sku_id"], group_keys=False)["units_sold"]
    for lag in (1, 7, 14, 28):
        work[f"lag_{lag}"] = group.shift(lag)

    work["rolling_mean_7"] = group.transform(lambda s: s.shift(1).rolling(7).mean())
    work["rolling_mean_28"] = group.transform(lambda s: s.shift(1).rolling(28).mean())
    work["day_of_week"] = work["date"].dt.dayofweek
    work["month"] = work["date"].dt.month
    work["week_of_year"] = work["date"].dt.isocalendar().week.astype(int)
    work["weekend"] = (work["day_of_week"] >= 5).astype(int)
    return work.dropna().reset_index(drop=True)
