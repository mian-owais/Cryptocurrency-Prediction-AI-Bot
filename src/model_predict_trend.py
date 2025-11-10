"""model_predict_trend.py
Wrapper to produce a prediction using the RL agent (or fallback rule-based model).
Exports: predict_and_log(agent, coin_id, timeframe, log_path)
"""
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from src.data_fetcher import fetch_price_series, simple_features_from_series, fetch_latest_price

ACTION_TO_LABEL = {0: 'Increase', 1: 'Decrease', 2: 'Constant'}


def predict_and_log(agent, coin_id: str, timeframe: str, log_path: str = 'data/logs/prediction_log.csv'):
    """Generate a prediction from the agent and append to CSV.

    state is a vector constructed from simple_features_from_series.
    """
    Path(os.path.dirname(log_path) or '.').mkdir(parents=True, exist_ok=True)

    # gather features
    df = fetch_price_series(coin_id, days=2.0)
    feats = simple_features_from_series(df)
    state = np.array([feats['rsi'], feats['ema_diff'],
                     feats['momentum'], feats['volatility']], dtype=np.float32)

    action = agent.select_action(state, epsilon=0.05)
    label = ACTION_TO_LABEL.get(action, 'Constant')

    price = fetch_latest_price(coin_id)

    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'coin': coin_id,
        'timeframe': timeframe,
        'model_name': getattr(agent, 'name', 'DQNAgent'),
        'predicted_label': label,
        'predicted_action': int(action),
        'predicted_prob': 0.0,
        'pred_features_rsi': feats['rsi'],
        'pred_features_ema_diff': feats['ema_diff'],
        'pred_features_momentum': feats['momentum'],
        'pred_features_volatility': feats['volatility'],
        'predicted_price': price
    }

    # Append to CSV
    df_out = pd.DataFrame([row])
    if not os.path.exists(log_path):
        df_out.to_csv(log_path, index=False)
    else:
        df_out.to_csv(log_path, index=False, mode='a', header=False)

    return row
