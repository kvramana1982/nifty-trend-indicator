"""
Load models and run predictions for the latest available row.
Also shows how to map classifier probabilities + regressor strength + ADX into final 1-100 score.
"""
import os
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "labeled_daily.parquet")

FEATURES_TO_USE = [
    "ret_1", "ret_5", "logret_1",
    "ema_10", "ema_20", "ema_50", "sma_50",
    "macd", "macd_diff", "rsi_14", "atr_14", "adx_14",
    "bb_pct",
    "slope_10", "slope_20", "dist_ema10", "dist_ema50",
    "volume"
]

def load_models():
    clf = joblib.load(os.path.join(MODELS_DIR, "lgbm_class.joblib"))
    reg = joblib.load(os.path.join(MODELS_DIR, "lgbm_reg.joblib"))
    return clf, reg

def map_to_strength(probs, reg_s, adx, slope_norm):
    # probs: array-like [p_class0 (down), p_class1 (sideways), p_class2 (up)]
    p_down, p_side, p_up = probs
    adx_norm = adx / 100.0
    # default blend weights
    alpha, beta, gamma = 0.6, 0.25, 0.15
    # up raw score
    score_up = p_up * (alpha * reg_s + beta * adx_norm + gamma * slope_norm)
    score_down = p_down * (alpha * reg_s + beta * adx_norm + gamma * slope_norm)
    score_side = p_side * (1 - adx_norm) * 0.8 + p_side * reg_s * 0.2
    scores = np.array([score_down, score_side, score_up])
    # normalize relative to historical max (simple min-max across these scores)
    maxv = np.max(scores) if np.max(scores) > 0 else 1.0
    norm = scores / maxv
    # get class index
    cls_idx = int(np.argmax(scores))
    # map that class score to 1..100
    mapped = int(round(norm[cls_idx] * 99 + 1))
    cls_map = {0:"Down", 1:"Sideways", 2:"Up"}
    return cls_map[cls_idx], mapped, scores

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH)
    clf, reg = load_models()
    latest = df.iloc[-1]
    X_latest = latest[FEATURES_TO_USE].values.reshape(1, -1)
    probs = clf.predict_proba(X_latest)[0]  # order: class_map used in train.py (0=down,1=sideways,2=up)
    reg_s = reg.predict(X_latest)[0]  # in [0,1]
    adx = latest['adx_14']
    slope_norm = abs(latest['slope_10'])
    cls, strength100, raw_scores = map_to_strength(probs, reg_s, adx, slope_norm)
    print("Predicted class:", cls)
    print("Predicted strength (1..100):", strength100)
    print("Classifier probs (down,side,up):", probs)
    print("Regressor strength [0..1]:", reg_s)
    print("Raw blended scores (down,side,up):", raw_scores)
