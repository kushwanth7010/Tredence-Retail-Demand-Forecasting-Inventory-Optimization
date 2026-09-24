import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inventory import build_inventory_recommendations


def test_reorder_point_exceeds_lead_time_demand():
    df = pd.DataFrame({
        "store_id": ["S01"] * 4,
        "sku_id": ["SKU001"] * 4,
        "forecast_units": [10.0, 12.0, 11.0, 13.0],
        "lead_time_days": [5, 5, 5, 5],
    })
    out = build_inventory_recommendations(df)
    row = out.iloc[0]
    assert row["safety_stock"] >= 0
    assert row["reorder_point"] >= row["avg_daily_forecast"] * row["lead_time_days"]
