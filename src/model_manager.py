"""
model_manager.py
----------------
Manages model training and prediction with automatic data fetching and retraining.
Handles:
- Fetching fresh data from CoinGecko API
- Loading and combining with historical data
- Training models on combined dataset
- Making predictions with trained models
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import joblib

from src.data_fetcher import fetch_price_series
from src.model_train import (
    train_models_for_coin,
    walk_forward_train,
    prepare_training_data,
    evaluate_preds
)
from src.model_predict import predict_next_from_data, load_model
from src.model_predict_trend import predict_and_log
from src.rl_agent import DQNAgent
from src.feature_engineering import add_advanced_features, normalize_features, create_target_features


MODELS_DIR = Path("models")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "crypto_data.csv"


def ensure_directories():
    """Ensure required directories exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)


def fetch_and_prepare_data(coin_id: str, days: float = 30.0) -> pd.DataFrame:
    """
    Fetch fresh data from CoinGecko and prepare it with indicators.
    
    Args:
        coin_id: Coin identifier (e.g., 'bitcoin')
        days: Number of days of history to fetch
        
    Returns:
        DataFrame with price data and technical indicators (indexed by timestamp)
    """
    try:
        # Fetch fresh price series
        df_price = fetch_price_series(coin_id, days=days)
        
        if df_price.empty:
            raise ValueError(f"No data fetched for {coin_id}")
        
        # Ensure index is datetime
        if not isinstance(df_price.index, pd.DatetimeIndex):
            if 'ts' in df_price.columns:
                df_price = df_price.set_index('ts')
            elif 'timestamp' in df_price.columns:
                df_price = df_price.set_index('timestamp')
        
        # Rename 'price' to 'close' for compatibility
        if 'price' in df_price.columns:
            df_price = df_price.rename(columns={'price': 'close'})
        
        # Add volume column if missing (set to 0 as placeholder)
        if 'volume' not in df_price.columns:
            df_price['volume'] = 0.0
        
        # Ensure 'close' column exists
        if 'close' not in df_price.columns:
            raise ValueError("No price/close column in fetched data")
        
        # Add advanced technical indicators and features
        df_with_indicators = add_advanced_features(df_price)
        
        return df_with_indicators
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Return minimal dataframe with price
        try:
            df_price = fetch_price_series(coin_id, days=days)
            if not df_price.empty:
                if not isinstance(df_price.index, pd.DatetimeIndex):
                    if 'ts' in df_price.columns:
                        df_price = df_price.set_index('ts')
                    elif 'timestamp' in df_price.columns:
                        df_price = df_price.set_index('timestamp')
                if 'price' in df_price.columns:
                    df_price = df_price.rename(columns={'price': 'close'})
                if 'volume' not in df_price.columns:
                    df_price['volume'] = 0.0
                return df_price
        except:
            pass
        return pd.DataFrame()


def add_indicators_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical indicators to price data."""
    if df.empty or 'close' not in df.columns:
        return df
    
    df = df.copy()
    close = df['close']
    
    # Simple Moving Averages
    df['sma_20'] = close.rolling(window=20, min_periods=1).mean()
    df['sma_50'] = close.rolling(window=50, min_periods=1).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['ema_26'] = close.ewm(span=26, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()
    
    # RSI approximation
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_mid'] = close.rolling(window=20, min_periods=1).mean()
    df['bb_std'] = close.rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # Momentum
    df['momentum'] = close.pct_change(periods=5)
    
    # Volatility
    df['volatility'] = close.pct_change().rolling(window=12, min_periods=1).std()
    
    # Fill NaN values (using new pandas API)
    df = df.ffill().bfill()
    
    return df


def load_historical_data(coin_id: str) -> Optional[pd.DataFrame]:
    """Load previously processed historical data if it exists."""
    if not PROCESSED_DATA_PATH.exists():
        return None
    
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=[0, 1], parse_dates=[1])
        if coin_id in df.index.get_level_values(0):
            coin_df = df.loc[coin_id].sort_index()
            # Ensure it has the expected structure
            if 'close' not in coin_df.columns and 'price' in coin_df.columns:
                coin_df = coin_df.rename(columns={'price': 'close'})
            return coin_df
    except Exception as e:
        print(f"Error loading historical data: {e}")
        # Try alternative loading method
        try:
            df = pd.read_csv(PROCESSED_DATA_PATH)
            if 'coin' in df.columns:
                coin_df = df[df['coin'] == coin_id].copy()
                if not coin_df.empty:
                    # Set timestamp as index if available
                    if 'timestamp' in coin_df.columns:
                        coin_df['timestamp'] = pd.to_datetime(coin_df['timestamp'])
                        coin_df = coin_df.set_index('timestamp').sort_index()
                    elif 'ts' in coin_df.columns:
                        coin_df['ts'] = pd.to_datetime(coin_df['ts'])
                        coin_df = coin_df.set_index('ts').sort_index()
                    # Remove coin column if present
                    if 'coin' in coin_df.columns:
                        coin_df = coin_df.drop('coin', axis=1)
                    if 'close' not in coin_df.columns and 'price' in coin_df.columns:
                        coin_df = coin_df.rename(columns={'price': 'close'})
                    return coin_df
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
    
    return None


def combine_data(old_data: Optional[pd.DataFrame], new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine old and new data, removing duplicates and sorting by timestamp.
    
    Args:
        old_data: Historical data (can be None)
        new_data: Freshly fetched data
        
    Returns:
        Combined DataFrame sorted by timestamp
    """
    if old_data is None or old_data.empty:
        return new_data.sort_index()
    
    if new_data.empty:
        return old_data.sort_index()
    
    # Combine dataframes
    combined = pd.concat([old_data, new_data])
    
    # Remove duplicates (keep last occurrence)
    combined = combined[~combined.index.duplicated(keep='last')]
    
    # Sort by timestamp
    combined = combined.sort_index()
    
    return combined


def train_model_for_prediction(
    coin_id: str,
    model_name: str,
    days: float = 30.0,
    use_cached: bool = True
) -> Tuple[Any, pd.DataFrame, Dict[str, float]]:
    """
    Train a model for making predictions. Fetches fresh data and combines with historical.
    
    Args:
        coin_id: Coin identifier
        model_name: Model to train ('random_forest', 'xgboost', 'rl_agent')
        days: Days of data to fetch
        use_cached: Whether to use cached historical data
        
    Returns:
        Tuple of (trained_model, training_data, metrics)
    """
    ensure_directories()
    
    # Fetch fresh data
    print(f"Fetching fresh data for {coin_id}...")
    new_data = fetch_and_prepare_data(coin_id, days=days)
    
    if new_data.empty:
        raise ValueError(f"Failed to fetch data for {coin_id}")
    
    # Load historical data if available
    old_data = None
    if use_cached:
        old_data = load_historical_data(coin_id)
    
    # Combine old and new data
    combined_data = combine_data(old_data, new_data)
    
    # Ensure we have enough data
    if len(combined_data) < 50:
        raise ValueError(f"Insufficient data: {len(combined_data)} rows. Need at least 50 rows.")
    
    print(f"Training {model_name} on {len(combined_data)} data points...")
    
    # Normalize features for better model performance
    # Exclude target and identifier columns from normalization
    exclude_cols = ['target', 'coin', 'timestamp', 'ts', 'close', 'price']
    feature_cols = [col for col in combined_data.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    if feature_cols:
        combined_data = normalize_features(combined_data, feature_cols)
    
    # Train model based on type
    if model_name.lower() in ['random_forest', 'rf', 'randomforest']:
        result = walk_forward_train(combined_data, model_name="random_forest", n_splits=3)
        model = result['model']
        metrics = result['metrics']
        
    elif model_name.lower() in ['xgboost', 'xgb']:
        result = walk_forward_train(combined_data, model_name="xgboost", n_splits=3)
        model = result['model']
        metrics = result['metrics']
        
    elif model_name.lower() in ['rl_agent', 'rl', 'dqn']:
        # For RL agent, we don't train here - it uses the predict_and_log function
        # Return a dummy model object
        model = DQNAgent(state_size=4)
        metrics = {'accuracy': 0.0, 'rmse': 0.0}
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Save the model
    model_path = MODELS_DIR / f"{model_name}_{coin_id}.pkl"
    if model_name.lower() not in ['rl_agent', 'rl', 'dqn']:
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    # Save combined data for future use
    try:
        # Update processed data file
        if PROCESSED_DATA_PATH.exists():
            try:
                existing_df = pd.read_csv(PROCESSED_DATA_PATH, index_col=[0, 1], parse_dates=[1])
                # Update or add coin data
                if coin_id in existing_df.index.get_level_values(0):
                    # Remove old data for this coin
                    existing_df = existing_df.drop(coin_id, level=0)
                # Add new combined data
                combined_data_with_coin = combined_data.copy()
                combined_data_with_coin['coin'] = coin_id
                combined_data_with_coin = combined_data_with_coin.reset_index()
                # Get index name (could be 'ts' or 'timestamp' or index name)
                index_name = combined_data_with_coin.columns[0] if len(combined_data_with_coin.columns) > 0 else 'timestamp'
                combined_data_with_coin = combined_data_with_coin.set_index(['coin', index_name])
                existing_df = pd.concat([existing_df, combined_data_with_coin])
                existing_df.to_csv(PROCESSED_DATA_PATH)
            except Exception as e2:
                # If reading fails, create new file
                combined_data_with_coin = combined_data.copy()
                combined_data_with_coin['coin'] = coin_id
                combined_data_with_coin = combined_data_with_coin.reset_index()
                index_name = combined_data_with_coin.columns[0] if len(combined_data_with_coin.columns) > 0 else 'timestamp'
                combined_data_with_coin = combined_data_with_coin.set_index(['coin', index_name])
                combined_data_with_coin.to_csv(PROCESSED_DATA_PATH)
        else:
            # Create new file
            combined_data_with_coin = combined_data.copy()
            combined_data_with_coin['coin'] = coin_id
            combined_data_with_coin = combined_data_with_coin.reset_index()
            index_name = combined_data_with_coin.columns[0] if len(combined_data_with_coin.columns) > 0 else 'timestamp'
            combined_data_with_coin = combined_data_with_coin.set_index(['coin', index_name])
            combined_data_with_coin.to_csv(PROCESSED_DATA_PATH)
    except Exception as e:
        print(f"Warning: Could not save processed data: {e}")
    
    return model, combined_data, metrics


def predict_with_model(
    coin_id: str,
    model_name: str,
    timeframe: str,
    days: float = 30.0,
    retrain: bool = True
) -> Dict[str, Any]:
    """
    Make a prediction using the specified model. Optionally retrains with fresh data.
    
    Args:
        coin_id: Coin identifier
        model_name: Model to use ('RandomForest', 'XGBoost', 'RL_Agent', etc.)
        timeframe: Prediction timeframe ('Next 5 Minutes', 'Next Day', 'Next Week')
        days: Days of data to use for training
        retrain: Whether to retrain the model with fresh data
        
    Returns:
        Dictionary with prediction results
    """
    ensure_directories()
    
    model_name_lower = model_name.lower()
    
    # Handle RL Agent separately
    if model_name_lower in ['rl_agent', 'rl', 'dqn']:
        agent = DQNAgent(state_size=4)
        # Try to load existing agent
        agent_path = MODELS_DIR / f"rl_agent_{coin_id}.pkl"
        if agent_path.exists():
            try:
                agent.load(str(agent_path))
            except:
                pass
        
        # Make prediction using predict_and_log
        prediction = predict_and_log(agent, coin_id, timeframe)
        
        # Save agent
        agent.save(str(agent_path))
        
        return {
            'predicted_label': prediction.get('predicted_label', 'Unknown'),
            'predicted_price': prediction.get('predicted_price', 0.0),
            'confidence': 0.0,
            'model_name': model_name,
            'timestamp': prediction.get('timestamp', datetime.utcnow().isoformat())
        }
    
    # For ML models (RandomForest, XGBoost)
    model_path = MODELS_DIR / f"{model_name_lower}_{coin_id}.pkl"
    
    # Train or load model
    if retrain or not model_path.exists():
        print(f"Training {model_name} with fresh data...")
        model, training_data, metrics = train_model_for_prediction(
            coin_id, model_name_lower, days=days
        )
    else:
        try:
            model = load_model(str(model_path))
            # Still fetch fresh data for prediction
            training_data = fetch_and_prepare_data(coin_id, days=days)
            metrics = {}
        except Exception as e:
            print(f"Error loading model, retraining: {e}")
            model, training_data, metrics = train_model_for_prediction(
                coin_id, model_name_lower, days=days
            )
    
    if training_data.empty:
        raise ValueError(f"No data available for prediction")
    
    # Make prediction
    try:
        prediction = predict_next_from_data(model, training_data)
        predicted_price = prediction['predicted_price']
        last_price = prediction['last_price']
        pct_change = prediction['predicted_pct_change']
        
        # Determine trend label
        thresh = 0.003  # 0.3% threshold
        if pct_change > thresh:
            label = 'Increase'
        elif pct_change < -thresh:
            label = 'Decrease'
        else:
            label = 'Constant'
        
        return {
            'predicted_label': label,
            'predicted_price': predicted_price,
            'last_price': last_price,
            'predicted_pct_change': pct_change,
            'confidence': min(abs(pct_change) * 100, 100.0),  # Simple confidence metric
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise


if __name__ == "__main__":
    # Test the module
    print("Testing model_manager...")
    try:
        result = predict_with_model('bitcoin', 'random_forest', 'Next Day', retrain=True)
        print("Prediction result:", result)
    except Exception as e:
        print(f"Test failed: {e}")

