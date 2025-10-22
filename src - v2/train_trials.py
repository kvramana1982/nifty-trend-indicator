# src/train_trials.py v1
"""
Experiment runner for Nifty Trend and Strength Indicator

- Runs multiple LightGBM hyperparameter configurations
- Evaluates them using walk-forward evaluation
- Reports accuracy, precision/recall/F1 for trend + strength

Use this to identify best configs before locking them into train.py
"""

import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

DATA_PATH = os.path.join("data", "processed", "labeled_daily.parquet")

# Candidate configurations
CONFIGS = [
    {
        "name": "baseline",
        "params": dict(
            objective="multiclass",
            num_class=3,
            learning_rate=0.05,
            n_estimators=2000,
            num_leaves=64,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        )
    },
    {
        "name": "deeper_trees",
        "params": dict(
            objective="multiclass",
            num_class=3,
            learning_rate=0.05,
            n_estimators=3000,
            num_leaves=128,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        )
    },
    {
        "name": "slow_lr",
        "params": dict(
            objective="multiclass",
            num_class=3,
            learning_rate=0.01,
            n_estimators=5000,
            num_leaves=128,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        )
    },
    {
        "name": "sideways_boost",
        "params": dict(
            objective="multiclass",
            num_class=3,
            learning_rate=0.05,
            n_estimators=2000,
            num_leaves=64,
            class_weight={0: 1.0, 1: 1.5, 2: 1.0},  # Upweight Sideways
            random_state=42,
            verbose=-1
        )
    }
]


def safe_smote_impute(X_train, y_strength_train, random_state=42):
    all_nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    X_reduced = X_train.drop(columns=all_nan_cols)

    if X_reduced.shape[1] == 0:
        return X_train.fillna(0).astype(np.float32), y_strength_train.to_numpy(), None, all_nan_cols

    imp = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imp.fit_transform(X_reduced),
                             columns=X_reduced.columns,
                             index=X_reduced.index)

    y_vals = y_strength_train.to_numpy()
    unique, counts = np.unique(y_vals, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0

    if min_count <= 1:
        return X_imputed, y_vals, imp, all_nan_cols

    k_neighbors = min(5, max(1, min_count - 1))
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res_arr, y_res = sm.fit_resample(X_imputed.values, y_vals)
    X_res = pd.DataFrame(X_res_arr, columns=X_imputed.columns)

    for col in all_nan_cols:
        X_res[col] = 0.0
    X_res = X_res.reindex(columns=X_train.columns, fill_value=0.0).astype(np.float32)

    return X_res, y_res, imp, all_nan_cols


def run_walkforward(config, start_date="2024-01-01", end_date="2025-09-25", retrain_every=5):
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    non_features = ["date", "timestamp", "label_class", "label_raw", "label_name",
                    "strength_01", "strength_bin", "adj_close",
                    "open", "high", "low", "close", "volume",
                    "rsi_14", "adx_14", "macd_diff", "bollinger_band_width", "bb_pct", "stoch_oscillator"]
    feature_cols = [c for c in df.columns if c not in non_features]

    preds = []
    unique_dates = df[df["date"] >= pd.Timestamp(start_date)]["date"].unique()
    unique_dates = [d for d in unique_dates if d <= pd.Timestamp(end_date)]

    clf_trend, clf_strength = None, None
    imp_strength, nan_cols_strength = None, []

    for i, test_date in enumerate(tqdm(unique_dates, desc=f"Config {config['name']}")):
        train_df = df[df["date"] < test_date]
        test_df = df[df["date"] == test_date]

        X_train = train_df[feature_cols]
        y_trend_train = train_df["label_class"]
        y_strength_train = train_df["strength_bin"]

        X_test = test_df[feature_cols]
        y_trend_test = test_df["label_class"]
        y_strength_test = test_df["strength_bin"]

        if i % retrain_every == 0 or clf_trend is None:
            # Train trend
            clf_trend = LGBMClassifier(**config["params"])
            clf_trend.fit(X_train, y_trend_train)

            # Train strength
            X_strength_res, y_strength_res, imp_strength, nan_cols_strength = safe_smote_impute(X_train, y_strength_train)
            clf_strength = LGBMClassifier(
                objective="multiclass",
                num_class=3,
                learning_rate=0.05,
                n_estimators=1000,
                num_leaves=64,
                class_weight="balanced",
                random_state=42,
                verbose=-1
            )
            clf_strength.fit(X_strength_res, y_strength_res)

        # Predictions
        y_trend_pred = clf_trend.predict(X_test)
        y_strength_pred = clf_strength.predict(X_test.fillna(0))

        for j in range(len(test_df)):
            preds.append({
                "date": str(test_df["date"].iloc[j]),
                "true_trend": int(y_trend_test.iloc[j]),
                "pred_trend": int(y_trend_pred[j]),
                "true_strength_bin": int(y_strength_test.iloc[j]),
                "pred_strength_bin": int(y_strength_pred[j]),
            })

    return pd.DataFrame(preds)


def main():
    results = []
    for cfg in CONFIGS:
        df_preds = run_walkforward(cfg)
        trend_acc = accuracy_score(df_preds["true_trend"], df_preds["pred_trend"])
        strength_acc = accuracy_score(df_preds["true_strength_bin"], df_preds["pred_strength_bin"])
        print(f"\n=== Config: {cfg['name']} ===")
        print("Trend accuracy:", trend_acc)
        print("Strength accuracy:", strength_acc)
        print("Trend report:")
        print(classification_report(df_preds["true_trend"], df_preds["pred_trend"], digits=3))
        print("Strength report:")
        print(classification_report(df_preds["true_strength_bin"], df_preds["pred_strength_bin"], digits=3))
        results.append({"config": cfg["name"], "trend_acc": trend_acc, "strength_acc": strength_acc})

    print("\n=== Summary ===")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
