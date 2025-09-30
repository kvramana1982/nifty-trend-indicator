# src/walkforward_eval_balanced.py
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import os


def safe_smote_impute(X_train, y_strength_train, random_state=42):
    """
    Impute (median) on the reduced set (drop all-NaN cols), run SMOTE on the imputed reduced training set,
    then re-add all-NaN columns as zeros and return full-shape DataFrame + labels + imputer + all_nan_cols.
    """
    all_nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    X_reduced = X_train.drop(columns=all_nan_cols)

    # If no usable features, fallback
    if X_reduced.shape[1] == 0:
        print("All features NaN, skipping SMOTE.")
        return X_train.fillna(0), y_strength_train.to_numpy(), None, all_nan_cols

    imp = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imp.fit_transform(X_reduced),
                             columns=X_reduced.columns,
                             index=X_reduced.index)

    # Safe SMOTE
    y_vals = y_strength_train.to_numpy()
    unique, counts = np.unique(y_vals, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0
    if min_count <= 1:
        print("SMOTE skipped: class with <=1 sample.")
        return X_imputed, y_vals, imp, all_nan_cols

    k_neighbors = min(5, max(1, min_count - 1))
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res_arr, y_res = sm.fit_resample(X_imputed.values, y_vals)

    X_res = pd.DataFrame(X_res_arr, columns=X_imputed.columns)
    for col in all_nan_cols:
        X_res[col] = 0.0
    X_res = X_res.reindex(columns=X_train.columns, fill_value=0.0).astype(np.float32)

    return X_res, y_res, imp, all_nan_cols


def run_walkforward(train_end, start_date, end_date, retrain_every=5):
    processed_path = os.path.join("data", "processed", "labeled_daily.parquet")
    df = pd.read_parquet(processed_path)

    # Ensure date column
    if "date" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "date"})
        else:
            raise KeyError("No 'date' or 'timestamp' column in dataset")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    non_features = ["date", "timestamp", "label_class", "strength_01", "strength_bin", "adj_close"]
    feature_cols = [c for c in df.columns if c not in non_features]

    preds = []
    unique_dates = df[df["date"] > pd.Timestamp(train_end)]["date"].unique()
    unique_dates = [d for d in unique_dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

    clf_trend, clf_strength = None, None
    imp_trend, imp_strength, nan_cols_trend, nan_cols_strength = None, None, [], []
    feature_importances_saved = False

    for i, test_date in enumerate(tqdm(unique_dates, desc="Walk-forward progress")):
        train_df = df[df["date"] < test_date]
        test_df = df[df["date"] == test_date]

        X_train = train_df[feature_cols]
        y_trend_train = train_df["label_class"]
        y_strength_train = train_df["strength_bin"]

        X_test = test_df[feature_cols]
        y_trend_test = test_df["label_class"]
        y_strength_test = test_df["strength_bin"]

        if i % retrain_every == 0 or clf_trend is None:
            # --- Trend pipeline ---
            nan_cols_trend = [c for c in X_train.columns if X_train[c].isna().all()]
            X_reduced = X_train.drop(columns=nan_cols_trend)
            imp_trend = SimpleImputer(strategy="median")
            X_trend_imputed = pd.DataFrame(imp_trend.fit_transform(X_reduced),
                                           columns=X_reduced.columns,
                                           index=X_reduced.index)
            for col in nan_cols_trend:
                X_trend_imputed[col] = 0.0
            X_trend_final = X_trend_imputed[X_train.columns].astype(np.float32)

            # --- Strength pipeline ---
            X_strength_res, y_strength_res, imp_strength, nan_cols_strength = safe_smote_impute(
                X_train, y_strength_train
            )

            # Train models
            clf_trend = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                class_weight="balanced",
                verbose=-1,
                objective="multiclass",
                num_class=3,
            )
            clf_strength = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                class_weight="balanced",
                verbose=-1,
                objective="multiclass",
                num_class=3,
            )

            clf_trend.fit(X_trend_final, y_trend_train)
            clf_strength.fit(X_strength_res, y_strength_res)

            if not feature_importances_saved:
                os.makedirs("artifacts", exist_ok=True)
                pd.DataFrame(
                    {"feature": X_train.columns, "importance": clf_trend.feature_importances_}
                ).sort_values("importance", ascending=False).to_csv(
                    os.path.join("artifacts", "feature_importances_trend.csv"), index=False
                )
                pd.DataFrame(
                    {"feature": X_train.columns, "importance": clf_strength.feature_importances_}
                ).sort_values("importance", ascending=False).to_csv(
                    os.path.join("artifacts", "feature_importances_strength.csv"), index=False
                )
                print("Saved feature importances for Trend and Strength to artifacts/")
                feature_importances_saved = True

        # --- Prepare test data ---
        # Trend
        X_test_reduced = X_test.drop(columns=nan_cols_trend, errors="ignore")
        X_test_imputed = pd.DataFrame(imp_trend.transform(X_test_reduced),
                                      columns=X_test_reduced.columns,
                                      index=X_test.index)
        for col in nan_cols_trend:
            X_test_imputed[col] = 0.0
        X_test_final = X_test_imputed[X_train.columns].astype(np.float32)

        # Strength
        X_test_reduced_s = X_test.drop(columns=nan_cols_strength, errors="ignore")
        X_test_imputed_s = pd.DataFrame(imp_strength.transform(X_test_reduced_s),
                                        columns=X_test_reduced_s.columns,
                                        index=X_test.index)
        for col in nan_cols_strength:
            X_test_imputed_s[col] = 0.0
        X_test_final_s = X_test_imputed_s[X_train.columns].astype(np.float32)

        # --- Predictions ---
        y_trend_pred = clf_trend.predict(X_test_final)
        y_trend_proba = clf_trend.predict_proba(X_test_final)
        y_strength_pred = clf_strength.predict(X_test_final_s)

        strength_map = {0: "Weak", 1: "Medium", 2: "Strong"}

        for j in range(len(test_df)):
            trend = int(y_trend_pred[j])
            strength_bin = int(y_strength_pred[j])
            strength_label = strength_map.get(strength_bin, "Weak")

            if trend == 0:
                blended = "Sideways"
            elif trend == 1:
                blended = f"{strength_label} Up"
            else:
                blended = f"{strength_label} Down"

            preds.append({
                "date": str(test_df["date"].iloc[j]),
                "true_trend": int(y_trend_test.iloc[j]),
                "pred_trend": trend,
                "true_strength_bin": int(y_strength_test.iloc[j]),
                "pred_strength_bin": strength_bin,
                "prob_down": float(y_trend_proba[j][0]),
                "prob_side": float(y_trend_proba[j][1]),
                "prob_up": float(y_trend_proba[j][2]),
                "blended": blended,
            })

    out_path = os.path.join("artifacts", "walkforward_predictions_balanced.csv")
    pd.DataFrame(preds).to_csv(out_path, index=False)
    print(f"\nSaved walkforward predictions to {out_path}")

    print(f"Accuracy (Trend): {accuracy_score([p['true_trend'] for p in preds], [p['pred_trend'] for p in preds])}")
    print("Classification report (Trend):")
    print(classification_report([p["true_trend"] for p in preds], [p["pred_trend"] for p in preds]))

    print(f"Accuracy (Strength, 3 bins): {accuracy_score([p['true_strength_bin'] for p in preds], [p['pred_strength_bin'] for p in preds])}")
    print("Classification report (Strength, 3 bins):")
    print(classification_report([p["true_strength_bin"] for p in preds], [p["pred_strength_bin"] for p in preds]))


if __name__ == "__main__":
    run_walkforward("2025-05-31", "2025-06-01", "2025-09-25", retrain_every=1)
