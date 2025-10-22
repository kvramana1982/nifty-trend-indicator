# src/train_context.py v2
"""
Train contextual LightGBM models for Nifty Trend and Strength prediction.

Changelog v2:
- Automatically aligns timestamp/date columns between features and labels
- Handles NaN-safe merging and consistent schema
- Uses contextual features (features_context.parquet)
- Binary Trend + 3-class Strength
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import joblib
__version__ = "2.1"

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models_context")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Load Data
# -------------------------------------------------------------------
print("Loading contextual features...")
df = pd.read_parquet(os.path.join(DATA_DIR, "features_context.parquet"))
print(f"Loaded dataset: {len(df)} rows")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# -------------------------------------------------------------------
# Load Labels
# -------------------------------------------------------------------
label_path = os.path.join(DATA_DIR, "labeled_daily.parquet")
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Missing labeling output: {label_path}")

labels = pd.read_parquet(label_path)
print(f"Loaded labels: {len(labels)} rows")

# -------------------------------------------------------------------
# Align Columns (timestamp vs date)
# -------------------------------------------------------------------
# Normalize label time column name
if "timestamp" not in labels.columns and "date" in labels.columns:
    labels.rename(columns={"date": "timestamp"}, inplace=True)

# Normalize feature time column name
if "timestamp" not in df.columns and "date" in df.columns:
    df.rename(columns={"date": "timestamp"}, inplace=True)

# Ensure both have datetime timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
labels["timestamp"] = pd.to_datetime(labels["timestamp"], errors="coerce")

# -------------------------------------------------------------------
# Merge
# -------------------------------------------------------------------
df = df.merge(
    labels[["timestamp", "label_class_bin", "strength_bin"]],
    on="timestamp",
    how="inner"
)
print(f"After merge: {len(df)} rows aligned")

# -------------------------------------------------------------------
# Prepare Features
# -------------------------------------------------------------------
feature_cols = df.select_dtypes(include=[np.number]).columns.drop(["label_class_bin", "strength_bin"], errors="ignore")
print(f"Feature sample: {list(feature_cols[:10])}")

mask_trend = df["label_class_bin"].notna()
mask_strength = df["strength_bin"].notna()

X_trend = df.loc[mask_trend, feature_cols]
y_trend = df.loc[mask_trend, "label_class_bin"].astype(int)

X_strength = df.loc[mask_strength, feature_cols]
y_strength = df.loc[mask_strength, "strength_bin"].astype(int)

# -------------------------------------------------------------------
# Split into train/val/test
# -------------------------------------------------------------------
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_tr_train, X_tr_val, X_tr_test, y_tr_train, y_tr_val, y_tr_test = split_data(X_trend, y_trend)
Xs_train, Xs_val, Xs_test, ys_train, ys_val, ys_test = split_data(X_strength, y_strength)

print(f"Trend train/val/test sizes: {len(X_tr_train)} / {len(X_tr_val)} / {len(X_tr_test)}")
print(f"Strength train/val/test sizes: {len(Xs_train)} / {len(Xs_val)} / {len(Xs_test)}")

# -------------------------------------------------------------------
# Imputation + SMOTE (for Strength)
# -------------------------------------------------------------------
imp = SimpleImputer(strategy="median")

X_tr_train_imp = pd.DataFrame(imp.fit_transform(X_tr_train), columns=X_tr_train.columns)
X_tr_val_imp = pd.DataFrame(imp.transform(X_tr_val), columns=X_tr_val.columns)
X_tr_test_imp = pd.DataFrame(imp.transform(X_tr_test), columns=X_tr_test.columns)

Xs_train_imp = pd.DataFrame(imp.fit_transform(Xs_train), columns=Xs_train.columns)
Xs_val_imp = pd.DataFrame(imp.transform(Xs_val), columns=Xs_val.columns)
Xs_test_imp = pd.DataFrame(imp.transform(Xs_test), columns=Xs_test.columns)

print(f"Applied median imputation to all numeric features.")

sm = SMOTE(random_state=42)
Xs_train_bal, ys_train_bal = sm.fit_resample(Xs_train_imp, ys_train)
print(f"Applied SMOTE balancing: before={ys_train.value_counts().to_dict()}, after={ys_train_bal.value_counts().to_dict()}")

# -------------------------------------------------------------------
# Train LightGBM Models
# -------------------------------------------------------------------
def train_lightgbm(X_train, y_train, X_val, y_val, params, objective):
    model = lgb.LGBMClassifier(**params, objective=objective, random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    return model

# Trend Model (binary)
params_trend = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=25,
)

print("\nTraining Trend Model (binary)...")
model_trend = train_lightgbm(X_tr_train_imp, y_tr_train, X_tr_val_imp, y_tr_val, params_trend, "binary")

# Strength Model (3-class)
params_strength = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=30,
)

print("\nTraining Strength Model (3-class)...")
model_strength = train_lightgbm(Xs_train_bal, ys_train_bal, Xs_val_imp, ys_val, params_strength, "multiclass")

# -------------------------------------------------------------------
# Evaluate
# -------------------------------------------------------------------
print("\n=== Evaluation (Trend Binary) ===")
y_pred_tr = model_trend.predict(X_tr_test_imp)
print(classification_report(y_tr_test, y_pred_tr, digits=4))

print("\n=== Evaluation (Strength 3-Class) ===")
y_pred_st = model_strength.predict(Xs_test_imp)
print(classification_report(ys_test, y_pred_st, digits=4))

# -------------------------------------------------------------------
# Save Models + Metadata
# -------------------------------------------------------------------
joblib.dump(model_trend, os.path.join(MODEL_DIR, "model_trend_context.pkl"))
joblib.dump(model_strength, os.path.join(MODEL_DIR, "model_strength_context.pkl"))
joblib.dump(feature_cols.tolist(), os.path.join(MODEL_DIR, "feature_columns_context.pkl"))

print(f"\n✅ Saved contextual models and feature columns to: {MODEL_DIR}")
