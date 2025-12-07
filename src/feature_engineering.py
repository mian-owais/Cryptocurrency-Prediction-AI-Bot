"""
feature_engineering.py
----------------------
Enhanced feature engineering for better model accuracy.
Adds advanced technical indicators, lag features, and statistical features.
"""

import pandas as pd
import numpy as np
from typing import Optional


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced technical indicators and features to improve model accuracy.
    
    Args:
        df: DataFrame with 'close' column and datetime index
        
    Returns:
        DataFrame with additional features
    """
    if df.empty or 'close' not in df.columns:
        return df
    
    df = df.copy()
    close = df['close']
    
    # ===== BASIC INDICATORS (already exist, but ensure they're there) =====
    # Moving Averages
    df['sma_5'] = close.rolling(window=5, min_periods=1).mean()
    df['sma_10'] = close.rolling(window=10, min_periods=1).mean()
    df['sma_20'] = close.rolling(window=20, min_periods=1).mean()
    df['sma_50'] = close.rolling(window=50, min_periods=1).mean()
    df['sma_200'] = close.rolling(window=200, min_periods=1).mean()
    
    # Exponential Moving Averages
    df['ema_5'] = close.ewm(span=5, adjust=False).mean()
    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['ema_26'] = close.ewm(span=26, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()
    
    # ===== PRICE-BASED FEATURES =====
    # Price ratios
    df['price_sma5_ratio'] = close / (df['sma_5'] + 1e-10)
    df['price_sma20_ratio'] = close / (df['sma_20'] + 1e-10)
    df['price_sma50_ratio'] = close / (df['sma_50'] + 1e-10)
    df['sma5_sma20_ratio'] = df['sma_5'] / (df['sma_20'] + 1e-10)
    df['sma20_sma50_ratio'] = df['sma_20'] / (df['sma_50'] + 1e-10)
    
    # Price position within range
    high_20 = close.rolling(window=20, min_periods=1).max()
    low_20 = close.rolling(window=20, min_periods=1).min()
    df['price_position'] = (close - low_20) / (high_20 - low_20 + 1e-10)
    
    # ===== MOMENTUM INDICATORS =====
    # RSI (multiple periods)
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # Rate of Change (ROC)
    df['roc_5'] = close.pct_change(periods=5) * 100
    df['roc_10'] = close.pct_change(periods=10) * 100
    df['roc_20'] = close.pct_change(periods=20) * 100
    
    # Momentum
    df['momentum_5'] = close.pct_change(periods=5)
    df['momentum_10'] = close.pct_change(periods=10)
    df['momentum_20'] = close.pct_change(periods=20)
    
    # ===== TREND INDICATORS =====
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ADX (Average Directional Index) approximation
    if 'high' in df.columns and 'low' in df.columns:
        high = df['high']
        low = df['low']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14, min_periods=1).mean()
    else:
        df['atr'] = close.rolling(window=14, min_periods=1).std()
    
    # ===== VOLATILITY FEATURES =====
    # Rolling volatility (multiple periods)
    returns = close.pct_change()
    for period in [5, 10, 20, 30]:
        df[f'volatility_{period}'] = returns.rolling(window=period, min_periods=1).std()
    
    # Bollinger Bands
    df['bb_mid'] = close.rolling(window=20, min_periods=1).mean()
    df['bb_std'] = close.rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-10)
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # ===== LAG FEATURES =====
    # Lagged prices (important for time series)
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = close.shift(lag)
        df[f'return_lag_{lag}'] = returns.shift(lag)
    
    # ===== ROLLING STATISTICS =====
    # Rolling mean, std, min, max
    for window in [5, 10, 20]:
        df[f'rolling_mean_{window}'] = close.rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = close.rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = close.rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = close.rolling(window=window, min_periods=1).max()
        df[f'rolling_range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
    
    # ===== VOLUME FEATURES (if available) =====
    if 'volume' in df.columns and df['volume'].sum() > 0:
        # Volume moving averages
        df['volume_sma_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        
        # Price-Volume features
        df['price_volume'] = close * df['volume']
        df['vwap'] = df['price_volume'].rolling(window=20, min_periods=1).sum() / df['volume'].rolling(window=20, min_periods=1).sum()
        df['price_vwap_ratio'] = close / (df['vwap'] + 1e-10)
    else:
        # Fill with zeros if no volume data
        df['volume_ratio'] = 0.0
        df['price_vwap_ratio'] = 1.0
    
    # ===== TIME-BASED FEATURES =====
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # ===== FEATURE INTERACTIONS =====
    # RSI and MACD interactions
    df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
    df['rsi_volatility_interaction'] = df['rsi_14'] * df['volatility_20']
    
    # ===== CLEANUP =====
    # Fill NaN values
    df = df.ffill().bfill()
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    
    # Fill any remaining NaN with 0
    df = df.fillna(0)
    
    return df


def normalize_features(df: pd.DataFrame, feature_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Normalize features for better model performance.
    
    Args:
        df: DataFrame with features
        feature_cols: List of columns to normalize (if None, normalize all numeric)
        
    Returns:
        DataFrame with normalized features
    """
    df = df.copy()
    
    if feature_cols is None:
        # Exclude target and identifier columns
        exclude_cols = ['target', 'coin', 'timestamp', 'ts']
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
    
    for col in feature_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 1e-10:  # Avoid division by zero
                df[col] = (df[col] - mean) / std
    
    return df


def create_target_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Create target variable and related features.
    
    Args:
        df: DataFrame with 'close' column
        horizon: Prediction horizon (1 = next period)
        
    Returns:
        DataFrame with target and target-related features
    """
    df = df.copy()
    
    if 'close' not in df.columns:
        return df
    
    # Create target (future price)
    df['target'] = df['close'].shift(-horizon)
    
    # Create target direction (for classification)
    df['target_direction'] = (df['target'] > df['close']).astype(int)
    
    # Create target percentage change
    df['target_pct_change'] = (df['target'] - df['close']) / (df['close'] + 1e-10)
    
    return df

