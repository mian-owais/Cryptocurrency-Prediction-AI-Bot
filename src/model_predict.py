"""
model_predict.py
----------------
Purpose:
 - Load a saved model and make next-interval predictions.
 - Provide convenience wrappers to produce human-friendly outputs.

Key functions:
 - load_model(path): load a persisted model (joblib or tf.keras)
 - predict_next(model, features): return prediction and optional confidence
 - batch_predict(model, X): vectorized predictions
"""

from typing import Any, Tuple
import numpy as np
import joblib
import os
import pandas as pd


def load_processed_for_coin(path: str = "data/processed/crypto_data.csv", coin: str = "bitcoin") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")
    df = pd.read_csv(path, index_col=[0, 1], parse_dates=[1])
    try:
        return df.loc[coin].sort_index()
    except KeyError:
        raise


def load_model(path: str) -> Any:
    """Load a model persisted with joblib or other serializer.
    Extend to auto-detect Keras/TensorFlow saved models in the future.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_next(model: Any, features: np.ndarray) -> float:
    """Predict a single next-step value given a 1D array of features.

    Args:
        model: trained model
        features: 1D numpy array of features

    Returns:
        float: predicted next-step price (or value)
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)
    preds = model.predict(features)
    return float(preds[0])


def predict_next_from_data(model: Any, df: pd.DataFrame) -> dict:
    """Given a time-indexed dataframe for one coin, prepare the latest feature vector
    and predict the next-step price. Returns predicted price and pct change vs last close.
    """
    if df.empty:
        raise ValueError("Empty dataframe provided for prediction")
    # Use last row's features
    last = df.select_dtypes(include=["number"]).iloc[-1]
    X = last.values.reshape(1, -1)
    pred = predict_next(model, X)
    last_price = float(last.get("close", last.get("price", np.nan)))
    pct_change = (pred - last_price) / (last_price + 1e-9)
    return {"predicted_price": float(pred), "predicted_pct_change": float(pct_change), "last_price": float(last_price)}


def batch_predict(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


if __name__ == "__main__":
    print("model_predict module loaded — implement load/predict flow in app")
