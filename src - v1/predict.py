# src/predict.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "labeled_daily.parquet")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    df = pd.read_parquet(DATA_PATH)

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in dataset. Please re-run features.py and labeling.py.")

    df = df.sort_values("date").reset_index(drop=True)

    if "label_class" not in df.columns or "strength_bin" not in df.columns:
        raise ValueError("Missing required columns 'label_class' or 'strength_bin' in labeled dataset")

    feature_path = os.path.join(MODELS_DIR, "feature_columns.joblib")
    saved_features = joblib.load(feature_path) if os.path.exists(feature_path) else None

    non_features = ["label_class", "strength_01", "strength_bin", "date", "timestamp", "adj_close"]
    feature_cols = [c for c in df.columns if c not in non_features]

    if saved_features is not None:
        missing = [c for c in saved_features if c not in feature_cols]
        if missing:
            raise ValueError(f"Missing expected features in dataset: {missing}")
        feature_cols = saved_features

    X = df[feature_cols].copy().apply(pd.to_numeric, errors="coerce")

    mask = df["label_class"].notna() & df["strength_bin"].notna()
    df = df.loc[mask].reset_index(drop=True)
    X = X.loc[mask].astype(np.float32)

    y_trend = df["label_class"].astype(int)
    y_strength = df["strength_bin"].astype(int)

    return df, X, y_trend, y_strength


def time_split(df, X, y_trend, y_strength, train_frac=0.7, val_frac=0.15):
    n = len(df)
    if n == 0:
        raise ValueError("No data available after cleaning — cannot split dataset")

    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    if i_train < 1:
        i_train = max(1, int(n * 0.6))
    if i_val <= i_train:
        i_val = min(n - 1, i_train + max(1, int(n * 0.1)))

    X_train, X_val, X_test = X.iloc[:i_train], X.iloc[i_train:i_val], X.iloc[i_val:]
    y_trend_train, y_trend_val, y_trend_test = y_trend.iloc[:i_train], y_trend.iloc[i_train:i_val], y_trend.iloc[i_val:]
    y_strength_train, y_strength_val, y_strength_test = y_strength.iloc[:i_train], y_strength.iloc[i_train:i_val], y_strength.iloc[i_val:]
    return (X_train, y_trend_train, y_strength_train), (X_val, y_trend_val, y_strength_val), (X_test, y_trend_test, y_strength_test)


def safe_smote_impute(X_train, X_val, X_test, y_strength_train, random_state=42):
    all_nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    X_reduced_train = X_train.drop(columns=all_nan_cols)

    if X_reduced_train.shape[1] == 0:
        print("All features NaN → fallback to zeros.")
        return X_train.fillna(0), y_strength_train.to_numpy(), X_val.fillna(0), X_test.fillna(0), None, all_nan_cols

    imp = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imp.fit_transform(X_reduced_train), columns=X_reduced_train.columns, index=X_reduced_train.index)

    X_val_imp = pd.DataFrame(imp.transform(X_val.drop(columns=all_nan_cols)), columns=X_reduced_train.columns, index=X_val.index)
    X_test_imp = pd.DataFrame(imp.transform(X_test.drop(columns=all_nan_cols)), columns=X_reduced_train.columns, index=X_test.index)

    # Safe SMOTE
    y_vals = y_strength_train.to_numpy()
    unique, counts = np.unique(y_vals, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0

    if min_count <= 1:
        print("SMOTE skipped (tiny class).")
        X_res, y_res = X_train_imp.copy(), y_vals.copy()
    else:
        k_neighbors = min(5, max(1, min_count - 1))
        sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_arr, y_res = sm.fit_resample(X_train_imp.values, y_vals)
        X_res = pd.DataFrame(X_arr, columns=X_train_imp.columns)

    for df_ in [X_res, X_val_imp, X_test_imp]:
        for col in all_nan_cols:
            df_[col] = 0.0
    X_res = X_res.reindex(columns=X_train.columns, fill_value=0.0)
    X_val_imp = X_val_imp.reindex(columns=X_train.columns, fill_value=0.0)
    X_test_imp = X_test_imp.reindex(columns=X_train.columns, fill_value=0.0)

    return X_res.astype(np.float32), y_res, X_val_imp.astype(np.float32), X_test_imp.astype(np.float32), imp, all_nan_cols


def main():
    df, X, y_trend, y_strength = load_data()
    (X_train, y_trend_train, y_strength_train), (X_val, y_trend_val, y_strength_val), (X_test, y_trend_test, y_strength_test) = time_split(df, X, y_trend, y_strength)
    print("Train/val/test sizes:", len(X_train), len(X_val), len(X_test))

    # --- Train Trend model ---
    clf_trend = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=64,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    clf_trend.fit(
        X_train, y_trend_train,
        eval_set=[(X_val, y_trend_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)]
    )

    # --- Train Strength model ---
    try:
        X_strength_res, y_strength_res, X_val_imp, X_test_imp, imp, all_nan_cols = safe_smote_impute(
            X_train, X_val, X_test, y_strength_train
        )
        print("Applied SMOTE to strength training; before:", np.bincount(y_strength_train.to_numpy()),
              "after:", np.bincount(y_strength_res))
    except Exception as e:
        print("SMOTE/impute failed:", e)
        X_strength_res, y_strength_res = X_train.fillna(0), y_strength_train.to_numpy()
        X_val_imp, X_test_imp = X_val.fillna(0), X_test.fillna(0)
        imp, all_nan_cols = None, []

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
        X_strength_res, y_strength_res,
        eval_set=[(X_val_imp, y_strength_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)]
    )

    # Save
    joblib.dump(clf_trend, os.path.join(MODELS_DIR, "lgbm_trend.joblib"))
    joblib.dump(clf_strength, os.path.join(MODELS_DIR, "lgbm_strength.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_columns.joblib"))
    print("Saved models and feature columns to", MODELS_DIR)

    # --- Evaluation ---
    print("\nClassification report (Trend Direction):")
    print(classification_report(y_trend_test, clf_trend.predict(X_test), digits=4))

    print("\nClassification report (Strength Weak/Medium/Strong):")
    print(classification_report(y_strength_test, clf_strength.predict(X_test_imp), digits=4))

    # --- Latest Prediction ---
    latest = X.iloc[[-1]]

    if imp is not None:
        latest_reduced = latest.drop(columns=all_nan_cols)
        latest_imputed = pd.DataFrame(imp.transform(latest_reduced), columns=latest_reduced.columns, index=latest.index)
        for col in all_nan_cols:
            latest_imputed[col] = 0.0
        latest_imputed = latest_imputed[X_train.columns]
    else:
        latest_imputed = latest.fillna(0)

    pred_trend = clf_trend.predict(latest)[0]
    pred_strength = clf_strength.predict(latest_imputed)[0]

    strength_map = {0: "Weak", 1: "Medium", 2: "Strong"}
    strength_label = strength_map.get(pred_strength, "Unknown")

    if pred_trend == 0:
        blended = "Sideways"
    elif pred_trend == 1:
        blended = f"{strength_label} Up"
    else:
        blended = f"{strength_label} Down"

    print(f"\nLatest prediction (blended): {blended}")


if __name__ == "__main__":
    main()
