# src/trend_strength_dashboard.py v1
"""
Visualize predicted vs actual trend and strength for recent months.

This dashboard:
- Reads walkforward or contextual predictions from artifacts/
- Plots actual vs predicted trend (Up/Down)
- Shows strength (Weak/Medium/Strong) as color bands
- Highlights confidence and misclassifications
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
OUT_PATH = os.path.join(ARTIFACTS_DIR, "trend_strength_dashboard.png")

# -------------------------------------------------------------------
# Load latest walkforward or context predictions
# -------------------------------------------------------------------
def load_predictions():
    candidates = [
        "walkforward_predictions_context.csv",
        "walkforward_predictions_balanced.csv",
        "walkforward_predictions.csv"
    ]
    for f in candidates:
        path = os.path.join(ARTIFACTS_DIR, f)
        if os.path.exists(path):
            print(f"Loading predictions from {path}")
            df = pd.read_csv(path)
            return df
    raise FileNotFoundError("No walkforward prediction CSV found in artifacts/")

df = load_predictions()

# Normalize timestamps
for c in ["timestamp", "date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

if "timestamp" in df.columns:
    df["date"] = df["timestamp"]
elif "date" not in df.columns:
    raise RuntimeError("No timestamp/date column found in predictions file.")

df = df.sort_values("date").dropna(subset=["date"])

# -------------------------------------------------------------------
# Keep only last 6 months
# -------------------------------------------------------------------
cutoff = df["date"].max() - timedelta(days=180)
df_recent = df[df["date"] >= cutoff].copy()

# -------------------------------------------------------------------
# Map trend labels for visual clarity
# -------------------------------------------------------------------
trend_map = {0: "Down", 1: "Up"}
if "true_trend_bin" in df_recent.columns:
    df_recent["true_trend"] = df_recent["true_trend_bin"].map(trend_map)
if "pred_trend_bin" in df_recent.columns:
    df_recent["pred_trend"] = df_recent["pred_trend_bin"].map(trend_map)

# -------------------------------------------------------------------
# Plot setup
# -------------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.title("NIFTY Trend & Strength Dashboard (Last 6 Months)", fontsize=14, fontweight="bold")

# Plot predicted and actual trends
if "true_trend_bin" in df_recent.columns and "pred_trend_bin" in df_recent.columns:
    plt.plot(df_recent["date"], df_recent["true_trend_bin"], "o-", label="Actual Trend", alpha=0.7)
    plt.plot(df_recent["date"], df_recent["pred_trend_bin"], "x--", label="Predicted Trend", alpha=0.7)
else:
    plt.plot(df_recent["date"], df_recent["pred_trend_bin"], "x--", label="Predicted Trend Only", alpha=0.8)

# Strength as shaded color band
if "pred_strength_bin" in df_recent.columns:
    for i, sname in enumerate(["Weak", "Medium", "Strong"]):
        plt.fill_between(
            df_recent["date"],
            i * 0.5, (i + 1) * 0.5,
            where=(df_recent["pred_strength_bin"] == i),
            color=["#a6cee3", "#1f78b4", "#b2df8a"][i],
            alpha=0.2, label=f"Pred Strength: {sname}" if i == 0 else None
        )

# Axis & style
plt.yticks([0, 1], ["Down", "Up"])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
plt.xlabel("Date")
plt.ylabel("Trend")
plt.legend(loc="upper left", frameon=True)
plt.grid(True, alpha=0.3)

# Save and show
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"\n✅ Dashboard saved to {OUT_PATH}")

try:
    plt.show()
except Exception:
    print("Plot displayed (headless mode).")
