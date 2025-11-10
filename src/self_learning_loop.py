"""self_learning_loop.py
Orchestration for verifying predictions and updating the RL agent.

This module provides functions to:
- verify pending predictions (compare predicted vs actual after timeframe)
- compute reward and append verification_log.csv
- push transitions to the agent replay buffer
- optionally trigger periodic retraining

This is a lightweight implementation intended for local use and experimentation.
"""
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from src.data_fetcher import fetch_price_series
import numpy as np

# Reward mapping
LABELS = ['Increase', 'Decrease', 'Constant']


def compute_movement_label(price_before: float, price_after: float, thresh: float = 0.001) -> str:
    if price_before == 0 or pd.isna(price_before) or pd.isna(price_after):
        return 'Constant'
    pct = (price_after - price_before) / (price_before + 1e-9)
    if pct > thresh:
        return 'Increase'
    elif pct < -thresh:
        return 'Decrease'
    else:
        return 'Constant'


def verify_pending_predictions(agent, prediction_log_path: str = 'data/logs/prediction_log.csv', verification_log_path: str = 'data/logs/verification_log.csv') -> int:
    """Load predictions and verify those without a corresponding verification entry. Returns number processed."""
    Path(os.path.dirname(prediction_log_path) or '.').mkdir(
        parents=True, exist_ok=True)
    Path(os.path.dirname(verification_log_path)
         or '.').mkdir(parents=True, exist_ok=True)

    if not os.path.exists(prediction_log_path):
        return 0

    preds = pd.read_csv(prediction_log_path)
    if preds.empty:
        return 0

    # load existing verifications
    existing = pd.read_csv(verification_log_path) if os.path.exists(
        verification_log_path) else pd.DataFrame()

    processed = 0

    for _, row in preds.iterrows():
        ts = pd.to_datetime(row['timestamp'])
        coin = row['coin']
        timeframe = row['timeframe']
        action = int(row.get('predicted_action', 0))
        predicted_label = row.get('predicted_label', 'Constant')

        # See if already verified by matching timestamp+coin+timeframe
        if not existing.empty:
            matched = existing[(existing['timestamp'] == row['timestamp']) & (
                existing['coin'] == coin) & (existing['timeframe'] == timeframe)]
            if not matched.empty:
                continue

        # Determine verification time window
        if timeframe == 'Next 5 Minutes':
            delta = timedelta(minutes=5)
        elif timeframe == 'Next Day':
            delta = timedelta(days=1)
        else:
            delta = timedelta(weeks=1)

        verify_time = ts + delta

        # Fetch price at verify_time by requesting historic series and selecting nearest
        df = fetch_price_series(coin, days=max(
            1, delta.total_seconds() / (24*3600) * 1.5))
        if df.empty:
            continue

        # find nearest index to verify_time
        try:
            nearest_idx = df.index.get_indexer(
                [verify_time], method='nearest')[0]
            price_after = float(df['price'].iloc[nearest_idx])
            # price at prediction time - we have predicted_price column optionally
            price_before = float(
                row.get('predicted_price', df['price'].iloc[0]))
        except Exception:
            continue

        actual_label = compute_movement_label(price_before, price_after)

        # reward assignment
        if actual_label == predicted_label:
            reward = 1.0
        elif actual_label == 'Constant':
            reward = 0.0
        else:
            reward = -1.0

        # Log verification
        vrow = {
            'timestamp': row['timestamp'],
            'coin': coin,
            'timeframe': timeframe,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'action': action,
            'reward': reward,
            'predicted_price': price_before,
            'actual_price': price_after,
            'verified_at': datetime.utcnow().isoformat()
        }

        dfv = pd.DataFrame([vrow])
        if not os.path.exists(verification_log_path):
            dfv.to_csv(verification_log_path, index=False)
        else:
            dfv.to_csv(verification_log_path, index=False,
                       mode='a', header=False)

        # Push a simple transition into agent replay buffer if agent supports it
        # Construct a minimal numeric state from features stored in prediction log (if available)
        try:
            # expect pred_features_* columns
            s = np.array([
                float(row.get('pred_features_rsi', 50.0)),
                float(row.get('pred_features_ema_diff', 0.0)),
                float(row.get('pred_features_momentum', 0.0)),
                float(row.get('pred_features_volatility', 0.0))
            ], dtype=np.float32)
            ns = s.copy()  # next_state placeholder (could fetch new features)
            done = False
            if hasattr(agent, 'store_transition'):
                agent.store_transition(s, action, reward, ns, done)
        except Exception:
            pass

        processed += 1

    # Optionally perform an online training burst
    if processed > 0 and hasattr(agent, 'train_step'):
        for _ in range(50):
            agent.train_step()

    return processed
