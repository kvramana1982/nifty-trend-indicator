# src/walkforward_eval_context.py v2
"""
Walk-forward evaluation using contextual LightGBM models.

Changelog v2:
- Fixed PeriodArray sorting issue (builds a sorted list of periods)
- Aligns test indices to rows that actually have labels (avoids shape mismatches)
- Intersects feature columns from saved model with available dataframe columns
- Saves predictions and summary to artifacts/
"""

import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models_context")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Load Data
# -------------------------------------------------------------------
print("Loading contextual features...")
df = pd.read_parquet(os.path.join(DATA_DIR, "features_context.parquet"))
labels = pd.read_parquet(os.path.join(DATA_DIR, "labeled_daily.parquet"))

# Align timestamp column names
if "timestamp" not in labels.columns and "date" in labels.columns:
    labels.rename(columns={"date": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
labels["timestamp"] = pd.to_datetime(labels["timestamp"])

# Merge features + labels
df = df.merge(labels[["timestamp", "label_class_bin", "strength_bin"]], on="timestamp", how="inner")
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Aligned dataset: {len(df)} rows, {df['timestamp'].min()} → {df['timestamp'].max()}")

# -------------------------------------------------------------------
# Load Models & feature columns
# -------------------------------------------------------------------
model_trend = joblib.load(os.path.join(MODEL_DIR, "model_trend_context.pkl"))
model_strength = joblib.load(os.path.join(MODEL_DIR, "model_strength_context.pkl"))
saved_feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns_context.pkl"))

# Intersect saved features with those present in df
feature_cols = [f for f in saved_feature_cols if f in df.columns]
dropped = [f for f in saved_feature_cols if f not in df.columns]
if dropped:
    print(f"Warning: {len(dropped)} saved feature columns are missing in the dataset and will be ignored. Examples: {dropped[:6]}")

if len(feature_cols) == 0:
    raise RuntimeError("No feature columns found in dataframe after intersecting with saved feature list.")

# -------------------------------------------------------------------
# Walk-forward setup
# -------------------------------------------------------------------
# Build a sorted list of monthly periods
months = df["timestamp"].dt.to_period("M")
# sorted() works for Period objects and returns a list
dates = sorted(list(months.unique()))
print(f"Running walk-forward over {len(dates)} months ({dates[0]} -> {dates[-1]})")

trend_results = []
strength_results = []
pred_records = []

# Use at least 12 months as initial 'history' for walk-forward
start_index = 12 if len(dates) > 12 else 0

for i in tqdm(range(start_index, len(dates))):
    train_end = dates[i - 1]
    test_period = dates[i]

    train_mask = df["timestamp"].dt.to_period("M") <= train_end
    test_mask = df["timestamp"].dt.to_period("M") == test_period

    train_data = df.loc[train_mask].copy()
    test_data = df.loc[test_mask].copy()

    if len(test_data) == 0:
        continue

    # Align test rows to those that actually have labels (drop rows with NaN labels)
    test_idx_trend = test_data[test_data["label_class_bin"].notna()].index
    test_idx_strength = test_data[test_data["strength_bin"].notna()].index

    # If no labeled rows in test_data, skip this month for the corresponding metric
    if len(test_idx_trend) == 0 and len(test_idx_strength) == 0:
        continue

    # Prepare X_test for rows that have labels (use same X for both if both exist)
    X_test_all = test_data.loc[:, feature_cols].fillna(0)

    # Trend evaluation (only for rows with label_class_bin)
    if len(test_idx_trend) > 0:
        X_test_trend = X_test_all.loc[test_idx_trend]
        y_trend_test = test_data.loc[test_idx_trend, "label_class_bin"].astype(int).values
        y_pred_trend = model_trend.predict(X_test_trend)
        trend_acc = accuracy_score(y_trend_test, y_pred_trend)
    else:
        y_trend_test = np.array([], dtype=int)
        y_pred_trend = np.array([], dtype=int)
        trend_acc = np.nan

    # Strength evaluation (only for rows with strength_bin)
    if len(test_idx_strength) > 0:
        X_test_strength = X_test_all.loc[test_idx_strength]
        y_strength_test = test_data.loc[test_idx_strength, "strength_bin"].astype(int).values
        y_pred_strength = model_strength.predict(X_test_strength)
        strength_acc = accuracy_score(y_strength_test, y_pred_strength)
    else:
        y_strength_test = np.array([], dtype=int)
        y_pred_strength = np.array([], dtype=int)
        strength_acc = np.nan

    trend_results.append((str(test_period), float(np.nan_to_num(trend_acc, nan=np.nan))))
    strength_results.append((str(test_period), float(np.nan_to_num(strength_acc, nan=np.nan))))

    # Store predictions in a common DataFrame (keep timestamp + align to row index)
    preds_list = []
    if len(test_idx_trend) > 0:
        preds_list.append(pd.DataFrame({
            "timestamp": test_data.loc[test_idx_trend, "timestamp"],
            "true_trend_bin": test_data.loc[test_idx_trend, "label_class_bin"].astype(int).values,
            "pred_trend_bin": y_pred_trend
        }).reset_index(drop=True))

    if len(test_idx_strength) > 0:
        preds_list.append(pd.DataFrame({
            "timestamp": test_data.loc[test_idx_strength, "timestamp"],
            "true_strength_bin": test_data.loc[test_idx_strength, "strength_bin"].astype(int).values,
            "pred_strength_bin": y_pred_strength
        }).reset_index(drop=True))

    if preds_list:
        # merge side-by-side on timestamp if both present, else keep single
        if len(preds_list) == 2:
            merged_preds = pd.merge(preds_list[0], preds_list[1], on="timestamp", how="outer")
        else:
            merged_preds = preds_list[0]
        pred_records.append(merged_preds)

# -------------------------------------------------------------------
# Aggregate results
# -------------------------------------------------------------------
trend_df = pd.DataFrame(trend_results, columns=["period", "trend_accuracy"])
strength_df = pd.DataFrame(strength_results, columns=["period", "strength_accuracy"])
summary_df = pd.merge(trend_df, strength_df, on="period", how="outer").sort_values("period").reset_index(drop=True)

mean_trend = summary_df["trend_accuracy"].dropna().mean()
mean_strength = summary_df["strength_accuracy"].dropna().mean()

print("\n=== Walk-Forward Evaluation Summary (last 12 rows) ===")
print(summary_df.tail(12))
print(f"\nMean Trend Accuracy (over months with data): {mean_trend:.3f}")
print(f"Mean Strength Accuracy (over months with data): {mean_strength:.3f}")

# -------------------------------------------------------------------
# Save Artifacts
# -------------------------------------------------------------------
if pred_records:
    all_preds = pd.concat(pred_records, sort=False).sort_values("timestamp").reset_index(drop=True)
else:
    all_preds = pd.DataFrame()

preds_path = os.path.join(ARTIFACTS_DIR, "walkforward_predictions_context.csv")
summary_path = os.path.join(ARTIFACTS_DIR, "walkforward_summary_context.csv")

all_preds.to_csv(preds_path, index=False)
summary_df.to_csv(summary_path, index=False)

print(f"\nSaved detailed predictions to: {preds_path}")
print(f"Saved monthly summary to: {summary_path}")

# -------------------------------------------------------------------
# Final report (if we have predictions)
# -------------------------------------------------------------------
if not all_preds.empty and "true_trend_bin" in all_preds.columns and "pred_trend_bin" in all_preds.columns:
    print("\n=== Final Classification Report (Full Period: Trend) ===")
    y_true_all = all_preds["true_trend_bin"].dropna().astype(int)
    y_pred_all = all_preds["pred_trend_bin"].dropna().astype(int)
    if len(y_true_all) > 0:
        print(classification_report(y_true_all, y_pred_all, digits=3))
        print(f"Overall Trend Accuracy: {accuracy_score(y_true_all, y_pred_all):.3f}")
    else:
        print("No trend-labeled rows in predictions; cannot produce final trend report.")
else:
    print("No predictions produced; check data / labeling alignment.")
