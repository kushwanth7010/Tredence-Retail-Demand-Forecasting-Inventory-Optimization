from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics.json"
FORECASTS = ROOT / "outputs" / "forecast_results.csv"
INVENTORY = ROOT / "outputs" / "inventory_recommendations.csv"
IMPORTANCE = ROOT / "outputs" / "feature_importance.csv"

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("AI-Driven Retail Demand Forecasting & Inventory Optimization")
st.caption("Standalone data science portfolio project using reproducible synthetic retail data.")

if not all(p.exists() for p in [METRICS, FORECASTS, INVENTORY, IMPORTANCE]):
    st.warning("Run `python src/pipeline.py` first to generate model outputs.")
    st.stop()

metrics = json.loads(METRICS.read_text(encoding="utf-8"))
forecasts = pd.read_csv(FORECASTS, parse_dates=["date"])
inventory = pd.read_csv(INVENTORY)
importance = pd.read_csv(IMPORTANCE)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", metrics["MAE"])
c2.metric("RMSE", metrics["RMSE"])
c3.metric("MAPE", f'{metrics["MAPE_percent"]}%')

store = st.selectbox("Store", sorted(forecasts["store_id"].unique()))
sku = st.selectbox("SKU", sorted(forecasts.loc[forecasts["store_id"] == store, "sku_id"].unique()))
view = forecasts[(forecasts["store_id"] == store) & (forecasts["sku_id"] == sku)].copy()

st.subheader("Actual vs Forecast Demand")
st.line_chart(view.set_index("date")[["units_sold", "forecast_units"]])

left, right = st.columns(2)
with left:
    st.subheader("Inventory Recommendation")
    inv = inventory[(inventory["store_id"] == store) & (inventory["sku_id"] == sku)]
    st.dataframe(inv, use_container_width=True, hide_index=True)
with right:
    st.subheader("Global Feature Importance")
    st.bar_chart(importance.head(10).set_index("feature")["importance"])

st.subheader("Forecast Records")
st.dataframe(view.tail(100), use_container_width=True, hide_index=True)
