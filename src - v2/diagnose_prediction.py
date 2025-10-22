import os
import pandas as pd
import joblib
import numpy as np

MODELS_DIR = os.path.join("models")
FEATURES_PATH = os.path.join("data", "processed", "features_daily.parquet")
FEATURES_TO_USE = [
    "ret_1", "ret_5", "logret_1",
    "ema_10", "ema_20", "ema_50", "sma_50",
    "macd", "macd_diff", "rsi_14", "atr_14", "adx_14",
    "bb_pct", "bollinger_band_width", "slope_10", "slope_20",
    "dist_ema10", "dist_ema50", "ema_cross_diff",
    "volume", "volume_sma_10", "volume_ratio", "roc_5", "stoch_oscillator"
]
CLASS_MAP_REV = {0: -1, 1: 0, 2: 1}


def main():
    clf = joblib.load(os.path.join(MODELS_DIR, "lgbm_class.joblib"))
    reg = joblib.load(os.path.join(MODELS_DIR, "lgbm_reg.joblib"))
    df = pd.read_parquet(FEATURES_PATH).sort_values('timestamp').reset_index(drop=True)
    row = df.iloc[-1]
    X = row[FEATURES_TO_USE].astype(float).to_frame().T

    probs = clf.predict_proba(X)[0]
    pred_internal = int(clf.predict(X)[0])
    pred_class = CLASS_MAP_REV[pred_internal]
    reg_s = float(reg.predict(X)[0])

    # diagnostics
    adx = float(row['adx_14'])
    slope10 = float(row['slope_10'])
    slope99 = df['slope_10'].abs().quantile(0.99)
    slope_norm = min(abs(slope10)/(slope99+1e-9), 1.0)

    print("Date:", row['timestamp'])
    print(f"Price O/H/L/C: {row['open']} / {row['high']} / {row['low']} / {row['close']}")
    print(f"Intraday change (close vs open): {((row['close']/row['open']-1)*100):.3f}%")
    print("
Model internals:")
    print("Classifier probs (down, sideways, up):", probs)
    print("Regressor strength (0..1):", round(reg_s, 4))
    print(f"ADX (adx_14): {adx}")
    print(f"slope_10: {slope10} slope_norm (vs 99pct): {slope_norm:.4f}")

    # blending logic (simple)
    up_raw = probs[2] * (reg_s + adx/100.0 + slope_norm)
    down_raw = probs[0] * (reg_s + adx/100.0 + slope_norm)
    side_raw = probs[1] * (1 - adx/100.0)
    raw = np.array([down_raw, side_raw, up_raw])
    norm = raw / (raw.max() if raw.max()>0 else 1.0)
    idx = int(np.argmax(raw))
    mapped_strength = int(round(norm[idx]*99 + 1))

    print("
Blended raw scores (down, side, up):", np.round(raw,6))
    print("Normalized scores (0..1):", np.round(norm,4))
    print("Blended predicted class index:", idx, "Mapped strength 1..100:", mapped_strength)

if __name__ == '__main__':
    main()