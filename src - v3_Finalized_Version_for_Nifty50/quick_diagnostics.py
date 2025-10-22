# src/quick_diagnostics.py v2
"""
Quick diagnostics before heavy experiments.

- Loads data/processed/labeled_daily.parquet
- Prints dataset shape / date range / label distributions
- Computes mutual information (feature -> label_class and -> strength_bin)
- Trains a small LightGBM (n_estimators=150) on time split (70/15/15) and prints classification reports
- Saves a small prediction CSV to artifacts/quick_diag_preds.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier, log_evaluation

# Paths
PROCESSED = os.path.join("data", "processed", "labeled_daily.parquet")
ARTIFACTS = os.path.join("artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

# Columns to drop (same as pipeline)
NON_FEATURES = [
    "label_raw", "label_class", "label_name",
    "strength_01", "strength_bin",
    "date", "timestamp", "adj_close",
    "open", "high", "low", "close", "volume"
]

def load_df():
    df = pd.read_parquet(PROCESSED)
    if "date" not in df.columns:
        raise RuntimeError("No 'date' column found in labeled dataset.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def mutual_info_ranking(X, y, topn=30):
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    mi = mutual_info_classif(X_imp, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi_series.head(topn), imp

def time_split_indices(n, train_frac=0.7, val_frac=0.15):
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return i_train, i_val

def run_quick():
    df = load_df()
    print("Rows:", len(df))
    print("Date range:", df['date'].min(), "to", df['date'].max())
    print("Label counts (trend label_class):")
    print(df["label_class"].value_counts().sort_index().to_string())
    print("Label counts (strength_bin):")
    print(df["strength_bin"].value_counts().sort_index().to_string())

    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    print(f"\nNumber of candidate features: {len(feature_cols)}")
    print("Sample features:", feature_cols[:12])

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    y_trend = df["label_class"].astype(int)
    y_strength = df["strength_bin"].astype(int)

    print("\nComputing mutual information (this is fast)...")
    mi_trend, _ = mutual_info_ranking(X, y_trend, topn=30)
    mi_strength, _ = mutual_info_ranking(X, y_strength, topn=30)
    print("\nTop features by mutual information (trend):")
    print(mi_trend.to_string())
    print("\nTop features by mutual information (strength):")
    print(mi_strength.to_string())

    n = len(df)
    i_train, i_val = time_split_indices(n)
    X_train, X_val, X_test = X.iloc[:i_train], X.iloc[i_train:i_val], X.iloc[i_val:]
    y_train, y_val, y_test = y_trend.iloc[:i_train], y_trend.iloc[i_train:i_val], y_trend.iloc[i_val:]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_train.columns, index=X_val.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns, index=X_test.index)

    print("\nTraining quick baseline LightGBM (n_estimators=150)...")
    clf = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        n_estimators=150,
        num_leaves=64,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    clf.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)],
        callbacks=[log_evaluation(period=50)]  # instead of verbose arg
    )

    y_pred = clf.predict(X_test_imp)
    print("\nQuick Trend classification report (test set):")
    print(classification_report(y_test, y_pred, digits=4))
    print("Trend accuracy (test):", accuracy_score(y_test, y_pred))

    fi = pd.DataFrame({"feature": X_train_imp.columns, "importance": clf.feature_importances_})
    fi = fi.sort_values("importance", ascending=False).head(20)
    fi.to_csv(os.path.join(ARTIFACTS, "quick_feature_importances.csv"), index=False)
    print("Saved top feature importances to artifacts/quick_feature_importances.csv")

    preds_df = pd.DataFrame({
        "date": df['date'].iloc[i_val:].astype(str).values,
        "true_trend": y_test.values,
        "pred_trend": y_pred
    })
    preds_df.to_csv(os.path.join(ARTIFACTS, "quick_diag_preds.csv"), index=False)
    print("Saved quick predictions to artifacts/quick_diag_preds.csv")

    return {
        "mi_trend_top": mi_trend,
        "mi_strength_top": mi_strength,
        "quick_test_report": classification_report(y_test, y_pred, output_dict=True)
    }

if __name__ == "__main__":
    run_quick()
    print("\nQuick diagnostics finished.")
