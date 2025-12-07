"""
data_pipeline.py
----------------
Full data pipeline implementation:
 - fetch_data(coin_ids, vs_currency, days): Uses CoinGecko to fetch price & volume history
 - clean_data(df): Clean and forward/back-fill per coin
 - add_indicators(df): Compute SMA, EMA, RSI, MACD, Bollinger Bands
 - save_data(df, path): Persist processed data to CSV
 - prepare_and_save(...): convenience runner that executes the pipeline

Notes:
 - Designed for daily/hourly data from CoinGecko. Handles basic rate-limit retry
   and fills missing values.
"""

import requests
from typing import Optional
from typing import List
import pandas as pd
import numpy as np
import os
import time
from pycoingecko import CoinGeckoAPI
import ta

cg = CoinGeckoAPI()


def fetch_data(coin_ids: List[str], vs_currency: str = "usd", days: int = 30) -> pd.DataFrame:
    """Fetch price history for each coin id and return a concatenated DataFrame.

    Returns a DataFrame indexed by MultiIndex (coin, timestamp) with columns ['price','volume'].
    """
    rows = []
    for coin in coin_ids:
        try:
            data = cg.get_coin_market_chart_by_id(
                id=coin, vs_currency=vs_currency, days=days)
        except Exception:
            time.sleep(2)
            data = cg.get_coin_market_chart_by_id(
                id=coin, vs_currency=vs_currency, days=days)

        prices = data.get("prices", [])
        volumes = {int(t): v for t, v in data.get("total_volumes", [])}
        for ts, price in prices:
            ts_ms = int(ts)
            rows.append({
                "coin": coin,
                "timestamp": pd.to_datetime(ts_ms, unit="ms"),
                "price": float(price),
                "volume": float(volumes.get(ts_ms, np.nan)),
            })

        time.sleep(1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.set_index(["coin", "timestamp"]).sort_index()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean combined DataFrame: remove duplicates and fill missing per coin."""
    if df.empty:
        return df

    df = df[~df.index.duplicated(keep="first")]
    # Forward/back fill per coin
    cleaned = []
    for coin, group in df.groupby(level=0):
        g = group.droplevel(0).sort_index()
        g = g.astype(float)
        g = g.ffill().bfill()
        g["coin"] = coin
        cleaned.append(g)

    if not cleaned:
        return pd.DataFrame()

    frames = []
    for g in cleaned:
        coin = g["coin"].iloc[0]
        tmp = g.drop(columns=["coin"]).copy()
        tmp.index = pd.MultiIndex.from_product([[coin], tmp.index])
        frames.append(tmp)

    out = pd.concat(frames)
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators per coin and return combined DataFrame.

    Adds: sma_20, ema_50, rsi_14, macd, macd_signal, bb_upper, bb_mid, bb_lower
    """
    if df.empty:
        return df

    results = []
    for coin, group in df.groupby(level=0):
        g = group.droplevel(0).sort_index()
        # ensure column names
        if "price" not in g.columns:
            raise ValueError("Input dataframe must contain 'price' column")
        g = g.rename(columns={"price": "close"})
        g["sma_20"] = g["close"].rolling(window=20, min_periods=1).mean()
        g["ema_50"] = g["close"].ewm(span=50, adjust=False).mean()
        try:
            g["rsi_14"] = ta.momentum.rsi(g["close"], window=14)
        except Exception:
            g["rsi_14"] = np.nan
        try:
            g["macd"] = ta.trend.macd(g["close"])
            g["macd_signal"] = ta.trend.macd_signal(g["close"])
        except Exception:
            g["macd"] = np.nan
            g["macd_signal"] = np.nan

        rolling20 = g["close"].rolling(window=20, min_periods=1)
        g["bb_mid"] = rolling20.mean()
        g["bb_std"] = rolling20.std()
        g["bb_upper"] = g["bb_mid"] + 2 * g["bb_std"]
        g["bb_lower"] = g["bb_mid"] - 2 * g["bb_std"]

        g = g.fillna(method="ffill").fillna(method="bfill")
        g["coin"] = coin
        results.append(g)

    frames = []
    for g in results:
        coin = g["coin"].iloc[0]
        tmp = g.drop(columns=["coin"]).copy()
        tmp.index = pd.MultiIndex.from_product([[coin], tmp.index])
        frames.append(tmp)

    out = pd.concat(frames)
    return out


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def prepare_and_save(coin_ids: List[str] = ["bitcoin", "ethereum"], days: int = 30, out_path: str = "data/processed/crypto_data.csv") -> pd.DataFrame:
    df = fetch_data(coin_ids, days=days)
    df = clean_data(df)
    df = add_indicators(df)
    save_data(df, out_path)
    return df


if __name__ == "__main__":
    print("Preparing data for bitcoin and ethereum...")
    df = prepare_and_save(["bitcoin", "ethereum"], days=30)
    print(df.tail())
"""
data_pipeline.py
----------------
Purpose:
 - Fetch cryptocurrency price data (CoinGecko API)
 - Clean and normalize data
 - Compute common technical indicators (RSI, MACD, SMA, EMA)
 - Persist processed datasets for training/prediction

Key functions (placeholders):
 - fetch_price_data(symbol, vs_currency, days): Fetch OHLCV or price history
 - clean_data(df): Clean and basic-impute missing values
 - compute_indicators(df): Compute RSI, MACD, moving averages, and other features
 - save_processed_data(df, path): Save processed data locally

Notes:
 - This module should be kept dependency-light and deterministic. Heavy feature
   engineering or experimental transformations should live in notebooks/.
"""


cg = CoinGeckoAPI()


def fetch_price_data(symbol: str = "bitcoin", vs_currency: str = "usd", days: int = 90) -> pd.DataFrame:
    """Fetch historical market data from CoinGecko.

    Args:
        symbol: coin id used by CoinGecko (e.g., 'bitcoin')
        vs_currency: quote currency (e.g., 'usd')
        days: how many days of history to fetch (int or 'max')

    Returns:
        DataFrame with columns ['timestamp', 'price'] or OHLC if extended.
    """
    # CoinGecko returns prices as [timestamp, price] lists
    data = cg.get_coin_market_chart_by_id(
        id=symbol, vs_currency=vs_currency, days=days)
    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: resample (if needed), forward-fill, drop duplicates.

    Keep this function minimal; more complex imputation can be added later.
    """
    df = df.sort_index()
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates()
    df = df.fillna(method="ffill").dropna()
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append technical indicators to the DataFrame.

    Indicators included (as examples): RSI(14), MACD, SMA(20), EMA(50)
    """
    out = df.copy()
    price_col = "price"
    # RSI
    try:
        out["rsi_14"] = ta.momentum.rsi(out[price_col], window=14)
        # MACD
        macd = ta.trend.macd(out[price_col])
        macd_signal = ta.trend.macd_signal(out[price_col])
        out["macd"] = macd
        out["macd_signal"] = macd_signal
        # Moving averages
        out["sma_20"] = out[price_col].rolling(window=20).mean()
        out["ema_50"] = out[price_col].ewm(span=50, adjust=False).mean()
    except Exception:
        # If ta library is not available or fails, provide safe fallbacks
        out["rsi_14"] = np.nan
        out["macd"] = np.nan
        out["macd_signal"] = np.nan
        out["sma_20"] = np.nan
        out["ema_50"] = np.nan

    out = out.fillna(method="ffill").dropna()
    return out


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


if __name__ == "__main__":
    # Quick manual test
    df = fetch_price_data("bitcoin", "usd", days=30)
    df = clean_data(df)
    df = compute_indicators(df)
    print(df.tail())
