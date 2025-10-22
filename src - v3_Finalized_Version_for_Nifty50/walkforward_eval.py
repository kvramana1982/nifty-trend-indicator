import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

DATA_PATH = os.path.join('data','processed','labeled_daily.parquet')
ARTIFACTS_DIR = os.path.join('artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

FEATURES = [
    "ret_1", "ret_5", "logret_1",
    "ema_10", "ema_20", "ema_50", "sma_50",
    "macd", "macd_diff", "rsi_14", "atr_14", "adx_14",
    "bb_pct", "bollinger_band_width", "slope_10", "slope_20",
    "dist_ema10", "dist_ema50", "ema_cross_diff",
    "volume", "volume_sma_10", "volume_ratio", "roc_5", "stoch_oscillator"
]

CLASS_MAP_REV = {0:-1,1:0,2:1}


def run_walkforward(train_end, start_date, end_date, retrain_every=5):
    df = pd.read_parquet(DATA_PATH).sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=FEATURES + ['label_class','strength_01'])

    preds = []
    unique_dates = df[(df['timestamp']>train_end) & (df['timestamp']>=start_date) & (df['timestamp']<=end_date)]['timestamp'].unique()

    clf=None; reg=None
    for i, date in enumerate(tqdm(unique_dates, desc='Walk-forward')):
        train_df = df[df['timestamp']<=date]
        X_train = train_df[FEATURES].astype(float)
        y_train = train_df['label_class'].astype(int).map({-1:0,0:1,1:2})
        y_reg = train_df['strength_01'].astype(float)

        if i % retrain_every == 0 or clf is None:
            clf = lgb.LGBMClassifier(objective='multiclass', num_class=3, learning_rate=0.05, n_estimators=1000)
            reg = lgb.LGBMRegressor(objective='regression', learning_rate=0.03, n_estimators=1000)
            clf.fit(X_train,y_train, eval_set=[(X_train,y_train)], callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)])
            reg.fit(X_train,y_reg, eval_set=[(X_train,y_reg)], callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)])

        test_df = df[df['timestamp']==date]
        X_test = test_df[FEATURES].astype(float)
        probs = clf.predict_proba(X_test)[0]
        pred = clf.predict(X_test)[0]
        reg_s = float(reg.predict(X_test)[0])

        preds.append({'date':str(date)[:10],'timestamp':date,'true_label':int(test_df['label_class'].iloc[0]),'pred_class':CLASS_MAP_REV[pred],'prob_down':probs[0],'prob_side':probs[1],'prob_up':probs[2],'reg_s':reg_s,'blended_strength':int(np.clip(reg_s*100,1,100))})

    out = pd.DataFrame(preds)
    out.to_csv(os.path.join(ARTIFACTS_DIR,'walkforward_predictions.csv'), index=False)
    print('Saved walkforward predictions to', os.path.join(ARTIFACTS_DIR,'walkforward_predictions.csv'))
    print('Accuracy:', accuracy_score(out['true_label'], out['pred_class']))
    print(classification_report(out['true_label'], out['pred_class']))

if __name__ == '__main__':
    run_walkforward('2025-05-31','2025-06-01','2025-09-25', retrain_every=1)