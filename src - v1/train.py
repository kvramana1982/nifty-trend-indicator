# src/train.py
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

    # standardized 'date' should be present (created by features.py)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in labeled dataset. Re-run features.py and labeling.py.")

    # sort chronologically (protect against accidentally unsorted files)
    df = df.sort_values("date").reset_index(drop=True)

    if 'label_class' not in df.columns or 'strength_bin' not in df.columns:
        raise ValueError("Missing required columns 'label_class' or 'strength_bin' in labeled dataset")

    # define feature columns (drop known non-feature cols)
    non_features = ['label_class', 'strength_01', 'strength_bin', 'date', 'timestamp', 'adj_close']
    feature_cols = [c for c in df.columns if c not in non_features]
    X = df[feature_cols].copy()

    # coerce numerics
    X = X.apply(pd.to_numeric, errors='coerce')

    # drop rows where labels are missing (shouldn't happen normally)
    mask = df['label_class'].notna() & df['strength_bin'].notna()
    df = df.loc[mask].reset_index(drop=True)
    X = X.loc[mask].astype(np.float32)

    y_class = df['label_class'].astype(int)
    y_strength = df['strength_bin'].astype(int)

    return df, X, y_class, y_strength


def time_split(df, X, y_class, y_strength, train_frac=0.7, val_frac=0.15):
    n = len(df)
    if n == 0:
        raise ValueError("No data available after cleaning — cannot split dataset")

    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    # safety fallbacks for tiny datasets
    if i_train < 1:
        i_train = max(1, int(n * 0.6))
    if i_val <= i_train:
        i_val = min(n - 1, i_train + max(1, int(n * 0.1)))

    X_train, X_val, X_test = X.iloc[:i_train], X.iloc[i_train:i_val], X.iloc[i_val:]
    y_class_train, y_class_val, y_class_test = y_class.iloc[:i_train], y_class.iloc[i_train:i_val], y_class.iloc[i_val:]
    y_strength_train, y_strength_val, y_strength_test = y_strength.iloc[:i_train], y_strength.iloc[i_train:i_val], y_strength.iloc[i_val:]
    return (X_train, y_class_train, y_strength_train), (X_val, y_class_val, y_strength_val), (X_test, y_class_test, y_strength_test)


def safe_smote_impute(X_train, X_val, X_test, y_strength_train, random_state=42):
    """
    Impute (median) on the reduced set (drop all-NaN cols), run SMOTE on the imputed reduced training set,
    then re-add all-NaN columns as zeros and return full-shape DataFrames for train/val/test.

    Returns:
      X_train_res (DataFrame) - possibly larger due to SMOTE
      y_strength_res (ndarray)
      X_val_imputed_full (DataFrame)
      X_test_imputed_full (DataFrame)
    """
    # detect all-NaN columns
    all_nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    X_reduced_train = X_train.drop(columns=all_nan_cols)

    # If reduced has zero columns, we cannot SMOTE — fallback to fillna(0)
    if X_reduced_train.shape[1] == 0:
        print("All features are NaN. Skipping SMOTE/imputation and using zeros fallback.")
        return X_train.fillna(0).astype(np.float32), y_strength_train.to_numpy(), X_val.fillna(0).astype(np.float32), X_test.fillna(0).astype(np.float32)

    # Fit imputer on reduced train
    imp = SimpleImputer(strategy="median")
    X_reduced_train_imputed = pd.DataFrame(
        imp.fit_transform(X_reduced_train),
        columns=X_reduced_train.columns,
        index=X_reduced_train.index
    )

    # Apply imputer to val/test reduced sets (handle case where val/test might be empty)
    X_reduced_val = X_val.drop(columns=all_nan_cols) if X_val.shape[0] > 0 else pd.DataFrame(columns=X_reduced_train.columns, index=[])
    X_reduced_test = X_test.drop(columns=all_nan_cols) if X_test.shape[0] > 0 else pd.DataFrame(columns=X_reduced_train.columns, index=[])

    X_reduced_val_imputed = pd.DataFrame(
        imp.transform(X_reduced_val) if X_reduced_val.shape[0] > 0 else np.empty((0, X_reduced_train_imputed.shape[1])),
        columns=X_reduced_train.columns,
        index=X_reduced_val.index if X_reduced_val.shape[0] > 0 else []
    )

    X_reduced_test_imputed = pd.DataFrame(
        imp.transform(X_reduced_test) if X_reduced_test.shape[0] > 0 else np.empty((0, X_reduced_train_imputed.shape[1])),
        columns=X_reduced_train.columns,
        index=X_reduced_test.index if X_reduced_test.shape[0] > 0 else []
    )

    # Decide SMOTE k_neighbors safely
    y_vals = y_strength_train.to_numpy()
    unique, counts = np.unique(y_vals, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0

    if min_count <= 1:
        # SMOTE cannot run with classes that have only 1 sample
        print("SMOTE skipped: at least one class has <= 1 sample. Using imputed training data without resampling.")
        X_train_res = X_reduced_train_imputed.copy()
        y_strength_res = y_vals.copy()
    else:
        # set k_neighbors <= min_count - 1 and at most 5
        k_neighbors = min(5, max(1, min_count - 1))
        sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        try:
            X_res_arr, y_res = sm.fit_resample(X_reduced_train_imputed.values, y_vals)
            X_train_res = pd.DataFrame(X_res_arr, columns=X_reduced_train_imputed.columns)
            y_strength_res = y_res
        except Exception as e:
            print("SMOTE failed, falling back to imputed training set. Error:", e)
            X_train_res = X_reduced_train_imputed.copy()
            y_strength_res = y_vals.copy()

    # Re-add all-NaN cols as zeros to train_res, val_imputed, test_imputed and reorder to original
    for df_ in [X_train_res, X_reduced_val_imputed, X_reduced_test_imputed]:
        for col in all_nan_cols:
            df_[col] = 0.0

    # Ensure columns order consistent with original X_train.columns
    X_train_res = X_train_res.reindex(columns=X_train.columns, fill_value=0.0)
    X_val_imputed_full = X_reduced_val_imputed.reindex(columns=X_train.columns, fill_value=0.0)
    X_test_imputed_full = X_reduced_test_imputed.reindex(columns=X_train.columns, fill_value=0.0)

    # Convert to float32
    X_train_res = X_train_res.astype(np.float32)
    X_val_imputed_full = X_val_imputed_full.astype(np.float32)
    X_test_imputed_full = X_test_imputed_full.astype(np.float32)

    return X_train_res, y_strength_res, X_val_imputed_full, X_test_imputed_full


def main():
    df, X, y_class, y_strength = load_data()
    (X_train, y_class_train, y_strength_train), (X_val, y_class_val, y_strength_val), (X_test, y_class_test, y_strength_test) = time_split(
        df, X, y_class, y_strength
    )
    print("Train/val/test sizes:", len(X_train), len(X_val), len(X_test))

    # --- Trend classifier (train on raw X_train; LightGBM handles NaN) ---
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
        X_train, y_class_train,
        eval_set=[(X_val, y_class_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)]
    )

    # --- Strength classifier: impute + SMOTE on reduced set, re-add missing cols ---
    try:
        X_strength_res, y_strength_res, X_val_imputed, X_test_imputed = safe_smote_impute(
            X_train, X_val, X_test, y_strength_train, random_state=42
        )
        print("Applied SMOTE to strength training; before:", np.bincount(y_strength_train.to_numpy()),
              "after:", np.bincount(y_strength_res))
    except Exception as e:
        print("SMOTE/impute pipeline failed:", e)
        X_strength_res, y_strength_res = X_train.fillna(0), y_strength_train.to_numpy()
        X_val_imputed, X_test_imputed = X_val.fillna(0), X_test.fillna(0)

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
        eval_set=[(X_val_imputed, y_strength_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)]
    )

    # Save models + feature column order
    joblib.dump(clf_trend, os.path.join(MODELS_DIR, "lgbm_trend.joblib"))
    joblib.dump(clf_strength, os.path.join(MODELS_DIR, "lgbm_strength.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_columns.joblib"))
    print("Saved models and feature columns to", MODELS_DIR)

    # --- Evaluation ---
    print("\nClassification report (Trend Direction):")
    print(classification_report(y_class_test, clf_trend.predict(X_test), digits=4))

    print("\nClassification report (Strength Weak/Medium/Strong):")
    # use X_test_imputed (same transformation used during strength training)
    print(classification_report(y_strength_test, clf_strength.predict(X_test_imputed), digits=4))


if __name__ == "__main__":
    main()
