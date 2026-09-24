import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import build_features


def test_lag_features_use_prior_values():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=35, freq="D"),
        "store_id": ["S01"] * 35,
        "sku_id": ["SKU001"] * 35,
        "category": ["Snacks"] * 35,
        "region": ["South"] * 35,
        "price": [100.0] * 35,
        "promotion": [0] * 35,
        "holiday": [0] * 35,
        "units_sold": list(range(35)),
        "lead_time_days": [5] * 35,
    })
    out = build_features(df)
    first = out.iloc[0]
    assert first["units_sold"] == 28
    assert first["lag_1"] == 27
    assert first["lag_7"] == 21
    assert first["lag_28"] == 0
