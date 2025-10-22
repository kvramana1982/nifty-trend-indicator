# src/walkforward_eval_balanced.py v4
"""
Walk-forward evaluation with balanced sampling.

Changelog v4:
- FIXED ValueError: "Cannot use median strategy with non-numeric data"
  by automatically dropping all non-numeric columns (e.g., 'label_name', 'date').
- Added clearer diagnostic prints to confirm numeric feature count before training.

Purpose:
- Runs rolling window training & testing.
- Evaluates both Trend (binary/multiclass) and Strength (3-bin).
- Saves predictions to artifacts/walkforward_predictions_balanced.csv
"""

import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "labeled_daily.parquet")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "walkforward_predictions_balanced.csv")

def run_walkforward(start_date, min_train_date, end_date, retrain_every=1):
    df = pd.read_parquet(DATA_PATH)
    if "date" not in df.columns and "timestamp" in df.columns:
        df.rename(columns={"timestamp": "date"}, inplace=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Drop rows without trend labels
    if "label_class_bin" in df.columns:
        mask = df["label_class_bin"].notna()
        df = df[mask]

    # --- Feature selection: keep only numeric columns
    exclude_cols = [
        "date", "label_raw", "label_name", "label_class", "label_class_bin",
        "strength_01", "strength_bin"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Ensure numeric dtype only
    X = df[feature_cols].select_dtypes(include=[np.number])
    print(f"Using {X.shape[1]} numeric features out of {len(feature_cols)} total.")

    y_trend = df["label_class_bin"].astype(int)
    y_strength = df["strength_bin"].astype(int)

    imp_trend = SimpleImputer(strategy="median")
    imp_strength = SimpleImputer(strategy="median")

    preds = []

    dates = df["date"].unique()
    n_steps = len(dates)
    for i in tqdm(range(n_steps), desc="Walk-forward progress"):
        current_date = dates[i]
        if current_date < np.datetime64(start_date) or current_date > np.datetime64(end_date):
            continue

        train_mask = df["date"] < current_date
        test_mask = df["date"] == current_date

        if train_mask.sum() < 100:
            continue

        X_train = X[train_mask]
        y_trend_train = y_trend[train_mask]
        y_strength_train = y_strength[train_mask]

        X_test = X[test_mask]
        y_trend_test = y_trend[test_mask]
        y_strength_test = y_strength[test_mask]

        # --- Impute numeric features
        X_train_imp = pd.DataFrame(imp_trend.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_imp = pd.DataFrame(imp_trend.transform(X_test), columns=X_test.columns, index=X_test.index)

        # --- Balance classes with SMOTE
        sm = SMOTE(random_state=42)
        X_train_bal, y_trend_bal = sm.fit_resample(X_train_imp, y_trend_train)

        # --- Train Trend model
        trend_model = LGBMClassifier(objective="binary", n_estimators=100, learning_rate=0.05)
        trend_model.fit(X_train_bal, y_trend_bal)

        # --- Predict Trend
        prob_up = trend_model.predict_proba(X_test_imp)[:, 1]
        pred_trend = (prob_up >= 0.5).astype(int)

        # --- Strength model
        X_train_imp_s = pd.DataFrame(imp_strength.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_imp_s = pd.DataFrame(imp_strength.transform(X_test), columns=X_test.columns, index=X_test.index)
        X_train_bal_s, y_strength_bal = sm.fit_resample(X_train_imp_s, y_strength_train)

        strength_model = LGBMClassifier(objective="multiclass", num_class=3, n_estimators=100, learning_rate=0.05)
        strength_model.fit(X_train_bal_s, y_strength_bal)

        pred_strength = strength_model.predict(X_test_imp_s)

        # --- Store predictions
        tmp = pd.DataFrame({
            "date": df.loc[test_mask, "date"],
            "true_trend": y_trend_test,
            "pred_trend": pred_trend,
            "true_strength": y_strength_test,
            "pred_strength": pred_strength,
            "prob_up": prob_up,
        })
        preds.append(tmp)

    preds_df = pd.concat(preds, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    preds_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved walkforward predictions to {OUT_PATH}")

    # --- Evaluation
    acc_trend = accuracy_score(preds_df["true_trend"], preds_df["pred_trend"])
    acc_strength = accuracy_score(preds_df["true_strength"], preds_df["pred_strength"])
    print(f"\nAccuracy (Trend): {acc_trend:.3f}")
    print("Classification report (Trend):")
    print(classification_report(preds_df["true_trend"], preds_df["pred_trend"]))

    print(f"Accuracy (Strength): {acc_strength:.3f}")
    print("Classification report (Strength):")
    print(classification_report(preds_df["true_strength"], preds_df["pred_strength"]))


if __name__ == "__main__":
    run_walkforward("2025-05-31", "2025-06-01", "2025-09-25", retrain_every=1)
