# src/train_trials_binary.py v3
"""
Walk-forward experiments for binary trend (Up vs Down).

Fixes:
- Skip early test dates until enough training history (MIN_TRAIN).
- Drop all-NaN columns in train windows before imputation, then re-add.
"""

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score
from lightgbm import LGBMClassifier
from tqdm import tqdm
import json
import datetime

DATA_PATH = os.path.join("data", "processed", "labeled_daily.parquet")
ARTIFACTS = os.path.join("artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

NON_FEATURES = [
    "label_raw", "label_class", "label_name", "label_class_bin",
    "strength_01", "strength_bin",
    "date", "timestamp", "adj_close",
    "open", "high", "low", "close", "volume"
]

CONFIGS = [
    {"name": "baseline", "params": {"objective": "binary", "learning_rate": 0.05, "n_estimators": 1000, "num_leaves": 64, "class_weight": "balanced", "random_state": 42}},
    {"name": "up_weighted", "params": {"objective": "binary", "learning_rate": 0.05, "n_estimators": 1000, "num_leaves": 64, "class_weight": {0:1.0, 1:2.0}, "random_state": 42}},
    {"name": "up_more_weighted", "params": {"objective": "binary", "learning_rate": 0.05, "n_estimators": 1000, "num_leaves": 64, "class_weight": {0:1.0, 1:3.0}, "random_state": 42}},
]

RETRAIN_EVERY = 20
MIN_TRAIN = 100   # skip test dates until at least 100 training samples exist

def load_data():
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def features_and_masks(df):
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    mask_trend = df['label_class_bin'].notna()
    return df, X, feature_cols, mask_trend

def train_and_predict_walkforward(config, df, X, feature_cols, mask_trend):
    test_dates = df.loc[mask_trend, 'date'].unique()
    preds = []
    clf, imputer = None, None

    for i, test_date in enumerate(tqdm(test_dates, desc=f"cfg:{config['name']}")):
        train_mask = (df['date'] < test_date) & (df['label_class_bin'].notna())
        test_mask = (df['date'] == test_date) & (df['label_class_bin'].notna())

        X_train = X.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, 'label_class_bin'].astype(int)

        X_test = X.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, 'label_class_bin'].astype(int)

        # Skip if not enough training history
        if len(X_train) < MIN_TRAIN:
            continue

        if (i % RETRAIN_EVERY == 0) or (clf is None):
            # Drop all-NaN columns in train
            all_nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
            X_reduced = X_train.drop(columns=all_nan_cols)

            imputer = SimpleImputer(strategy='median')
            X_train_imp = pd.DataFrame(imputer.fit_transform(X_reduced),
                                       columns=X_reduced.columns,
                                       index=X_reduced.index)

            print(f"Retraining {config['name']} at {test_date.date()} with {len(X_train_imp)} samples")

            clf = LGBMClassifier(**config['params'])
            clf.fit(X_train_imp, y_train)

            active_cols = X_reduced.columns.tolist()

        if len(X_test) == 0:
            continue

        X_test_reduced = X_test[active_cols]
        X_test_imp = pd.DataFrame(imputer.transform(X_test_reduced),
                                  columns=active_cols,
                                  index=X_test.index)

        probs = clf.predict_proba(X_test_imp)[:, 1]
        preds_local = clf.predict(X_test_imp)

        for idx, row_idx in enumerate(X_test_imp.index):
            preds.append({
                'date': str(df.loc[row_idx, 'date']),
                'true_trend': int(df.loc[row_idx, 'label_class_bin']),
                'pred_trend': int(preds_local[idx]),
                'prob_up': float(probs[idx]),
                'config': config['name']
            })

    return pd.DataFrame(preds)

def evaluate_preds(df_preds):
    if df_preds.empty:
        return {}
    acc = accuracy_score(df_preds['true_trend'], df_preds['pred_trend'])
    rec_down = recall_score(df_preds['true_trend'], df_preds['pred_trend'], pos_label=0)
    rec_up = recall_score(df_preds['true_trend'], df_preds['pred_trend'], pos_label=1)
    prec_down = precision_score(df_preds['true_trend'], df_preds['pred_trend'], pos_label=0)
    prec_up = precision_score(df_preds['true_trend'], df_preds['pred_trend'], pos_label=1)
    return {
        'accuracy': acc,
        'recall_down': rec_down,
        'recall_up': rec_up,
        'precision_down': prec_down,
        'precision_up': prec_up
    }

def main():
    df = load_data()
    df, X, feature_cols, mask_trend = features_and_masks(df)
    summary = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for cfg in CONFIGS:
        preds_df = train_and_predict_walkforward(cfg, df, X, feature_cols, mask_trend)
        out_csv = os.path.join(ARTIFACTS, f"wf_preds_{cfg['name']}_{timestamp}.csv")
        preds_df.to_csv(out_csv, index=False)
        metrics = evaluate_preds(preds_df)
        summary.append({'config': cfg['name'], **metrics})
        print(f"\nConfig {cfg['name']} metrics:")
        print(json.dumps(metrics, indent=2))

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(ARTIFACTS, f"wf_summary_{timestamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\nSaved summary to", summary_csv)

if __name__ == "__main__":
    main()
