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
        return {
            "n_estimators": 300, 
            "max_depth": 15, 
            "max_features": "sqrt",
            "min_samples_split": 5,
            "min_samples_leaf": 2
        }

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 5, 30, log=False)
        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "log2"])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1, 
            random_state=42)
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = cross_val_score(
            model, X, y, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    # Safety check: ensure max_features is never 'auto' (for compatibility with newer sklearn)
    if 'max_features' in best_params and best_params['max_features'] == 'auto':
        best_params['max_features'] = 'sqrt'
    return best_params


def tune_xgboost(X, y, n_trials: int = 20, cv_splits: int = 3) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters using Optuna. Returns best params dict.

    If Optuna is not available, returns default params.
    """
    if optuna is None:
        return {
            "n_estimators": 300, 
            "max_depth": 6, 
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0
        }

    def objective(trial):
        params = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_loguniform("gamma", 0.01, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 0.01, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 0.01, 10.0),
        }
        model = xgb.XGBRegressor(**params, random_state=42)
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = cross_val_score(
            model, X, y, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
