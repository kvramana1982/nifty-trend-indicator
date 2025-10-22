# src/train.py v6
"""
Final training script for Nifty Trend and Strength Indicator (v6)

- Binary trend classification (Up vs Down), Sideways excluded.
- Trend model uses up_more_weighted config (Up weighted 3×).
- Strength model unchanged (3-bin with SMOTE).
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "labeled_daily.parquet")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

NON_FEATURES = [
    "label_raw", "label_class", "label_name", "label_class_bin",
    "strength_01", "strength_bin",
    "date", "timestamp", "adj_close",
    "open", "high", "low", "close", "volume"
]

def load_data():
    df = pd.read_parquet(DATA_PATH).sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Binary trend labels (exclude Sideways NaN)
    mask_trend = df["label_class_bin"].notna()
    X_trend = X.loc[mask_trend].astype(np.float32)
    y_trend = df.loc[mask_trend, "label_class_bin"].astype(int)

    # Strength labels
    mask_strength = df["strength_bin"].notna()
    X_strength = X.loc[mask_strength].astype(np.float32)
    y_strength = df.loc[mask_strength, "strength_bin"].astype(int)

    print(f"Loaded dataset: {len(df)} rows total")
    print(f"Trend rows (binary, no Sideways): {X_trend.shape[0]}")
    print(f"Strength rows: {X_strength.shape[0]}")
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Feature sample:", feature_cols[:10])
    return df, feature_cols, X_trend, y_trend, X_strength, y_strength

def time_split(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    i_train, i_val = int(n*train_frac), int(n*(train_frac+val_frac))
    return (X.iloc[:i_train], y.iloc[:i_train],
            X.iloc[i_train:i_val], y.iloc[i_train:i_val],
            X.iloc[i_val:], y.iloc[i_val:])

def smote_impute(X_train, y_train, X_val, X_test):
    imp = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_imp = pd.DataFrame(imp.transform(X_val), columns=X_train.columns, index=X_val.index)
    X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_train.columns, index=X_test.index)

    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train_imp, y_train)
    return X_res, y_res, X_val_imp, X_test_imp

def main():
    df, feature_cols, X_trend, y_trend, X_strength, y_strength = load_data()

    # --- Binary trend split
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X_trend, y_trend)

    print("\nTrend binary class distribution (train):", y_train.value_counts().to_dict())
    print("Trend binary class distribution (val):", y_val.value_counts().to_dict())
    print("Trend binary class distribution (test):", y_test.value_counts().to_dict())

    clf_trend = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=64,
        class_weight={0: 1.0, 1: 3.0},  # up_more_weighted
        random_state=42,
        verbose=-1
    )
    clf_trend.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)]
    )

    # --- Strength split
    Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test = time_split(X_strength, y_strength)
    Xs_res, ys_res, Xs_val_imp, Xs_test_imp = smote_impute(Xs_train, ys_train, Xs_val, Xs_test)

    print("\nStrength distribution (train before SMOTE):", ys_train.value_counts().to_dict())
    print("Strength distribution (train after SMOTE):", pd.Series(ys_res).value_counts().to_dict())

    clf_strength = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=64,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    clf_strength.fit(
        Xs_res, ys_res,
        eval_set=[(Xs_val_imp, ys_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)]
    )

    # Save models
    joblib.dump(clf_trend, os.path.join(MODELS_DIR, "lgbm_trend_binary.joblib"))
    joblib.dump(clf_strength, os.path.join(MODELS_DIR, "lgbm_strength.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    print("\nSaved models and feature columns to", MODELS_DIR)

    # --- Evaluation
    print("\nClassification report (Trend Up vs Down):")
    print(classification_report(y_test, clf_trend.predict(X_test), digits=4))

    print("\nClassification report (Strength Weak/Medium/Strong):")
    print(classification_report(ys_test, clf_strength.predict(Xs_test_imp), digits=4))

if __name__ == "__main__":
    main()
