from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "demand_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "modeling_data.csv"
OUT_PATH = ROOT / "outputs" / "shap_summary.png"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
    df = pd.read_csv(DATA_PATH)
    sample = df[features].sample(min(2000, len(df)), random_state=42)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    shap.summary_plot(values, sample, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
