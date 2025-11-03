"""
optimize.py
-----------
Optuna-based hyperparameter tuning helpers for RandomForest and XGBoost.

Functions:
 - tune_random_forest(X, y, n_trials=20): returns best_params
 - tune_xgboost(X, y, n_trials=20): returns best_params

The module falls back gracefully if Optuna is not installed.
"""
from typing import Dict, Any
import numpy as np

try:
    import optuna
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
except Exception:
    optuna = None
    cross_val_score = None
    TimeSeriesSplit = None
    RandomForestRegressor = None
    xgb = None


def tune_random_forest(X, y, n_trials: int = 20, cv_splits: int = 3) -> Dict[str, Any]:
    """Tune RandomForest hyperparameters using Optuna. Returns best params dict.

    If Optuna is not available, returns a sensible default config.
    """
    if optuna is None:
        return {"n_estimators": 200, "max_depth": None, "max_features": "auto"}

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 400)
        max_depth = trial.suggest_int("max_depth", 3, 30)
        max_features = trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"])
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=-1, random_state=42)
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = cross_val_score(
            model, X, y, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_xgboost(X, y, n_trials: int = 20, cv_splits: int = 3) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters using Optuna. Returns best params dict.

    If Optuna is not available, returns default params.
    """
    if optuna is None:
        return {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}

    def objective(trial):
        params = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        }
        model = xgb.XGBRegressor(**params, random_state=42)
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = cross_val_score(
            model, X, y, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
