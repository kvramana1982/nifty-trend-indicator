# src/compare_feature_importances.py v1
"""
Compare feature importances between baseline and contextual feature sets.

- Loads baseline (features_daily.parquet)
- Loads extended (features_context.parquet)
- Trains LightGBM classifiers (quick runs)
- Prints top 15 features before vs after
- Saves comparison to artifacts/feature_importance_comparison.csv
"""

import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "feature_importance_comparison.csv")

def train_lightgbm(X, y):
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return model, imp

def compare_feature_importances():
    base_path = os.path.join(DATA_DIR, "features_daily.parquet")
    ext_path = os.path.join(DATA_DIR, "features_context.parquet")

    print(f"Loading baseline: {base_path}")
    df_base = pd.read_parquet(base_path)
    print(f"Loading contextual: {ext_path}")
    df_ext = pd.read_parquet(ext_path)

    y = (df_base["close"].pct_change().shift(-1) > 0).astype(int)  # simple next-day up/down proxy
    y = y.loc[df_base.index]

    X_base = df_base.select_dtypes("number").dropna(axis=1)
    X_ext = df_ext.select_dtypes("number").dropna(axis=1)

    # Align labels
    X_base, y_base = X_base.align(y, axis=0, join="inner")
    X_ext, y_ext = X_ext.align(y, axis=0, join="inner")

    # Train/test split
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_base, y_base, test_size=0.2, shuffle=False)
    Xe_train, Xe_test, ye_train, ye_test = train_test_split(X_ext, y_ext, test_size=0.2, shuffle=False)

    # Train models
    print("Training baseline LightGBM...")
    _, imp_base = train_lightgbm(Xb_train, yb_train)
    print("Training contextual LightGBM...")
    _, imp_ext = train_lightgbm(Xe_train, ye_train)

    # Merge importances
    imp_merge = pd.merge(imp_base, imp_ext, on="feature", how="outer", suffixes=("_base", "_context"))
    imp_merge = imp_merge.fillna(0).sort_values("importance_context", ascending=False)

    # Save and print
    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    imp_merge.to_csv(ARTIFACT_PATH, index=False)
    print(f"Saved importance comparison to {ARTIFACT_PATH}")

    print("\n=== Top 15 Features (Contextual Model) ===")
    print(imp_merge[["feature", "importance_context"]].head(15).to_string(index=False))

if __name__ == "__main__":
    compare_feature_importances()
