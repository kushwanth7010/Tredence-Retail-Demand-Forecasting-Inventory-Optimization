from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "retail_sales.csv"


def generate_retail_data(seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible, realistic synthetic retail demand dataset.

    The dataset is intentionally synthetic so the repository is fully self-contained
    and can be shared publicly without licensing or privacy concerns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    stores = [f"S{i:02d}" for i in range(1, 6)]
    skus = [f"SKU{i:03d}" for i in range(1, 21)]
    categories = ["Beverages", "Snacks", "Personal Care", "Household"]
    regions = ["South", "West", "North", "East", "Central"]

    rows: list[list[object]] = []
    for store_idx, store in enumerate(stores):
        store_multiplier = [1.05, 0.95, 1.10, 0.90, 1.00][store_idx]
        for sku_idx, sku in enumerate(skus):
            category = categories[sku_idx % len(categories)]
            base_demand = rng.uniform(18, 65) * store_multiplier
            base_price = rng.uniform(40, 450)
            price_sensitivity = rng.uniform(0.15, 0.40)

            for date in dates:
                day_of_week = date.dayofweek
                weekend = int(day_of_week >= 5)
                promotion = int(rng.random() < 0.12)
                holiday = int((date.month, date.day) in {(1, 1), (8, 15), (10, 2), (12, 25)})
                yearly = 1 + 0.18 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                weekly = 1 + 0.08 * weekend
                trend = 1 + 0.00025 * (date - dates[0]).days
                price = base_price * (1 - 0.08 * promotion) * (1 + rng.normal(0, 0.015))
                relative_price = price / base_price
                price_effect = max(0.75, 1 - price_sensitivity * (relative_price - 1))
                demand = (
                    base_demand
                    * yearly
                    * weekly
                    * trend
                    * price_effect
                    * (1 + 0.28 * promotion + 0.15 * holiday)
                    + rng.normal(0, 5)
                )
                units_sold = max(0, int(round(demand)))
                lead_time_days = int(rng.integers(2, 10))

                rows.append([
                    date.date(), store, sku, category, regions[store_idx],
                    round(float(price), 2), promotion, holiday,
                    units_sold, lead_time_days,
                ])

    return pd.DataFrame(
        rows,
        columns=[
            "date", "store_id", "sku_id", "category", "region", "price",
            "promotion", "holiday", "units_sold", "lead_time_days",
        ],
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate_retail_data()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")


if __name__ == "__main__":
    main()
