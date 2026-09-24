from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_raw_data_quality():
    path = ROOT / "data" / "raw" / "retail_sales.csv"
    assert path.exists()
    df = pd.read_csv(path)
    expected = {
        "date", "store_id", "sku_id", "category", "region", "price",
        "promotion", "holiday", "units_sold", "lead_time_days",
    }
    assert expected.issubset(df.columns)
    assert len(df) == 73100
    assert df[list(expected)].isna().sum().sum() == 0
    assert (df["price"] > 0).all()
    assert (df["units_sold"] >= 0).all()
    assert df["store_id"].nunique() == 5
    assert df["sku_id"].nunique() == 20
