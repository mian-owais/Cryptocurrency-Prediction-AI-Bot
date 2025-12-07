"""
model_train.py
--------------
Implements Stage 1 training pipeline with feature selection, Optuna tuning hooks,
walk-forward (time-series) validation, metric logging, and model persistence.

Key functions:
 - load_processed(path, coin): load processed CSV for a coin
 - prepare_training_data(df, target_col, horizon): build X, y and feature names
 - evaluate_preds(y_true, y_pred, last_prices): compute RMSE, MAE, R2, MAPE, directional accuracy
 - walk_forward_train(df, model_name): perform time-series CV and return final model + metrics
 - train_models_for_coin(coin): high-level function to run selection, training, tuning, logging, and saving

Notes:
 - This module uses `src/optimize.py` for Optuna tuning when available.
 - Metrics are appended to `data/logs/training_logs.csv`.
"""

import os
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Tuple, Dict, Any, List
import pandas as pd
from .optimize import tune_random_forest, tune_xgboost
from datetime import datetime


def load_processed(path: str = "data/processed/crypto_data.csv", coin: str = "bitcoin") -> pd.DataFrame:
    """Load processed CSV and return DataFrame for the requested coin as a time-indexed DF."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")
    df = pd.read_csv(path, index_col=[0, 1], parse_dates=[1])
    # Select coin
    try:
        coin_df = df.loc[coin]
    except KeyError:
        # If single-index CSV or different shape, try to recover
        df2 = pd.read_csv(path, parse_dates=[1])
        if "coin" in df2.columns:
            df2 = df2.set_index(["coin", df2.columns[1]])
            coin_df = df2.loc[coin]
        else:
            raise
    return coin_df.sort_index()


def prepare_training_data(df: pd.DataFrame, target_col: str = None, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Return X, y, feature_names, last_prices used for directional accuracy.

    If target_col is None, use 'close' then 'price'.
    """
    df = df.copy()
    if target_col is None:
        if "close" in df.columns:
            target_col = "close"
        elif "price" in df.columns:
            target_col = "price"
        else:
            # fallback: use first numeric column as target
            target_col = df.select_dtypes(include=["number"]).columns[0]

    df["target"] = df[target_col].shift(-horizon)
    
    # Only drop rows where target or close price is NaN, fill other NaN values
    df = df.dropna(subset=["target", target_col])
    df = df.ffill(limit=5).bfill(limit=5)
    
    # Check if we have any data left after cleaning
    if len(df) == 0:
        raise ValueError("No valid training data available after cleaning (all rows were NaN)")
    
    feature_names = list(df.select_dtypes(include=["number"]).columns.drop(
        "target")) if "target" in df.columns else list(df.select_dtypes(include=["number"]).columns)
    X = df[feature_names].values
    y = df["target"].values
    # last_prices aligned with y (price at t for target at t+1)
    last_prices = df[target_col].values
    return X, y, feature_names, last_prices


def evaluate_preds(y_true: np.ndarray, y_pred: np.ndarray, last_prices: np.ndarray = None) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))))

    # Directional accuracy: how often sign(pred - last_price) == sign(true - last_price)
    dir_acc = None
    if last_prices is not None and len(last_prices) == len(y_true):
        pred_dir = np.sign(y_pred - last_prices)
        true_dir = np.sign(y_true - last_prices)
        dir_acc = float((pred_dir == true_dir).mean())

    out = {"mse": float(mse), "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
    if dir_acc is not None:
        out["directional_accuracy"] = dir_acc
    return out


def ensure_logs_dir(path: str = "data/logs"):
    os.makedirs(path, exist_ok=True)


def log_metrics(metrics: Dict[str, float], coin: str, model_name: str, out_path: str = "data/logs/training_logs.csv") -> None:
    """Append metrics to a CSV log file with timestamp, coin, model name, and metrics."""
    ensure_logs_dir(os.path.dirname(out_path) or "data/logs")
    row = {"timestamp": datetime.utcnow().isoformat(), "coin": coin,
           "model": model_name}
    row.update(metrics)
    df_row = pd.DataFrame([row])
    if not os.path.exists(out_path):
        df_row.to_csv(out_path, index=False)
    else:
        df_row.to_csv(out_path, mode="a", header=False, index=False)


def walk_forward_train(df: pd.DataFrame, model_name: str = "random_forest", n_splits: int = 5) -> Dict[str, Any]:
    """Perform walk-forward CV and train final model with optional tuning.

    Returns dict with keys: model (fitted on full data) and metrics (aggregated from CV folds).
    """
    X, y, feature_names, last_prices = prepare_training_data(df)
    
    # Check if we have enough data for cross-validation
    if len(X) == 0:
        raise ValueError("No training data available (0 samples)")
    if len(X) < n_splits:
        # If not enough data for n_splits folds, reduce n_splits dynamically
        n_splits = max(2, len(X) // 2)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        last_test = last_prices[test_idx]

        if model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=300, max_depth=15, max_features="sqrt", 
                min_samples_split=5, min_samples_leaf=2,
                random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        elif model_name == "xgboost":
            model = xgb.XGBRegressor(
                n_estimators=200, objective="reg:squarederror", random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            raise ValueError("Unsupported model_name")

        m = evaluate_preds(y_test, preds, last_test)
        fold_metrics.append(m)

    # Aggregate metrics (mean across folds)
    if fold_metrics:
        agg = {k: float(np.mean([m.get(k, np.nan)
                        for m in fold_metrics])) for k in fold_metrics[0]}
    else:
        # Fallback metrics if no folds were created
        agg = {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "mape": 0.0}

    # Attempt to tune hyperparameters (Optuna) — increased trials for better accuracy
    try:
        if model_name == "random_forest":
            best_params = tune_random_forest(X, y, n_trials=50, cv_splits=3)
            # Safety check: ensure max_features is never 'auto'
            if 'max_features' in best_params and best_params['max_features'] == 'auto':
                best_params['max_features'] = 'sqrt'
            final_model = RandomForestRegressor(
                **best_params, n_jobs=-1, random_state=42)
        else:
            best_params = tune_xgboost(X, y, n_trials=50, cv_splits=3)
            final_model = xgb.XGBRegressor(
                **best_params, objective="reg:squarederror", random_state=42)
    except Exception:
        # fallback defaults
        if model_name == "random_forest":
            final_model = RandomForestRegressor(
                n_estimators=300, max_depth=15, max_features="sqrt",
                min_samples_split=5, min_samples_leaf=2,
                random_state=42, n_jobs=-1)
        else:
            final_model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=42)

    final_model.fit(X, y)
    return {"model": final_model, "metrics": agg, "feature_names": feature_names}


def train_models_for_coin(coin: str = "bitcoin", processed_path: str = "data/processed/crypto_data.csv", out_model_path: str = "models/best_model.pkl") -> Dict[str, Any]:
    """High-level training flow: load data, optional feature selection, train models, choose best, save, and log metrics."""
    df = load_processed(processed_path, coin=coin)
    df_numeric = df.select_dtypes(include=["number"]).copy()

    if df_numeric.empty:
        raise ValueError("No numeric data available for training")

    print(f"Training models for {coin} — data rows: {len(df_numeric)}")

    # Feature selection via RFE (lightweight)
    try:
        X_full = df_numeric.values[:-1, :]
        y_full = df_numeric.iloc[:, 0].shift(-1).dropna().values
        selector = RFE(LinearRegression(), n_features_to_select=max(
            3, min(10, X_full.shape[1] // 2)))
        selector = selector.fit(X_full, y_full)
        mask = selector.support_
        if mask.shape[0] == df_numeric.shape[1]:
            df_selected = df_numeric.iloc[:, mask]
        else:
            df_selected = df_numeric
    except Exception:
        df_selected = df_numeric

    # Train RF and XGB and collect metrics
    rf_res = walk_forward_train(df_selected, model_name="random_forest")
    xgb_res = walk_forward_train(df_selected, model_name="xgboost")

    print("RandomForest metrics:", rf_res["metrics"])
    print("XGBoost metrics:", xgb_res["metrics"])

    # pick best by RMSE
    rf_rmse = rf_res["metrics"].get("rmse", np.inf)
    xgb_rmse = xgb_res["metrics"].get("rmse", np.inf)
    best_res = rf_res if rf_rmse <= xgb_rmse else xgb_res
    best_name = "random_forest" if best_res is rf_res else "xgboost"

    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    joblib.dump(best_res["model"], out_model_path)
    print(f"Saved best model to {out_model_path}")

    # Log metrics
    try:
        log_metrics(best_res["metrics"], coin, best_name)
    except Exception:
        pass

    return {"best_metrics": best_res["metrics"], "model_path": out_model_path, "model_name": best_name}


if __name__ == "__main__":
    # Quick local run (safe defaults)
    try:
        res = train_models_for_coin("bitcoin")
        print(res)
    except Exception as e:
        print("Training failed:", e)
