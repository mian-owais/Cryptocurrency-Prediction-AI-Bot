"""
trading_env.py
-------------
Trading environment for RL agent training. Implements a simple gym-like interface
and manages portfolio state, trade costs, and reward calculations.

Key classes:
 - TradingEnv: main environment (reset/step API)
 - Portfolio: tracks positions and handles trades

Notes:
 - State includes price-derived features: [RSI, MACD, EMAs, price_change, position]
 - Actions are discrete: [0=HOLD, 1=BUY, 2=SELL]
 - Reward is portfolio value change minus costs
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd


class Portfolio:
    """Track crypto positions and cash balance."""

    def __init__(self, cash: float = 10000.0, position: float = 0.0):
        self.initial_cash = cash
        self.cash = cash
        self.position = position
        self.trades = []

    def get_value(self, price: float) -> float:
        return self.cash + self.position * price

    def trade(self, price: float, action: int, trade_size: float = 0.1, cost: float = 0.001) -> Tuple[float, float]:
        """Execute a trade. Returns (pnl, cost).

        Args:
            price: current asset price
            action: 0=HOLD, 1=BUY, 2=SELL
            trade_size: position size as fraction of cash
            cost: transaction cost as fraction of trade value
        """
        if action == 0:  # HOLD
            return 0.0, 0.0

        trade_value = self.cash * \
            trade_size if action == 1 else self.position * price * trade_size
        trade_cost = trade_value * cost
        actual_value = trade_value - trade_cost

        if action == 1:  # BUY
            if self.cash < trade_value:
                return 0.0, 0.0
            self.position += actual_value / price
            self.cash -= trade_value
        else:  # SELL
            if self.position <= 0:
                return 0.0, 0.0
            self.position -= self.position * trade_size
            self.cash += actual_value

        self.trades.append({
            "action": "BUY" if action == 1 else "SELL",
            "price": price,
            "value": trade_value,
            "cost": trade_cost,
        })
        return actual_value - trade_value, trade_cost

    def get_position_size(self) -> float:
        """Return normalized position size in [-1, 1] range."""
        total = self.get_value(1.0)  # price=1 to get ratio
        if total <= 0:
            return 0.0
        return self.position / total


class TradingEnv:
    """Gym-like environment for crypto trading.

    State includes technical indicators and current position.
    Actions are discrete: [HOLD, BUY, SELL].
    Reward is portfolio value change minus transaction costs.
    """

    def __init__(self, df: pd.DataFrame, window: int = 20, trade_size: float = 0.1, cost: float = 0.001):
        """
        Args:
            df: DataFrame with features (MUST have 'close' or 'price' column)
            window: lookback window for features
            trade_size: max trade size as fraction of cash
            cost: transaction cost as fraction of trade value
        """
        self.df = df.copy()
        self.window = window
        self.trade_size = trade_size
        self.cost = cost
        self.idx = window
        self.portfolio = None
        self._prepare_features()

    def _prepare_features(self):
        """Prepare state features: technical indicators + returns."""
        df = self.df
        # ensure we have a price column
        if "close" in df.columns:
            df["price"] = df["close"]
        elif "price" not in df.columns:
            raise ValueError("DataFrame must have 'close' or 'price' column")

        # compute returns and volatility
        df["returns"] = df["price"].pct_change()
        df["volatility"] = df["returns"].rolling(self.window).std()

        # normalize price-based features
        self.features = df.select_dtypes(include=[np.number]).copy()
        # skip first column (usually price) to avoid look-ahead
        for col in self.features.columns[1:]:
            mean = self.features[col].mean()
            std = self.features[col].std()
            if std > 0:
                self.features[col] = (self.features[col] - mean) / std

    def _get_state(self) -> np.ndarray:
        """Return current state vector (features + position)."""
        features = self.features.iloc[self.idx].values
        position = self.portfolio.get_position_size()
        return np.append(features, position)

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.idx = self.window
        self.portfolio = Portfolio()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step: take action, get reward, return next state."""
        if self.idx >= len(self.df) - 1:
            return self._get_state(), 0.0, True, {}

        price = self.df.iloc[self.idx]["price"]
        next_price = self.df.iloc[self.idx + 1]["price"]

        # execute trade and get costs
        old_value = self.portfolio.get_value(price)
        pnl, cost = self.portfolio.trade(
            price, action, self.trade_size, self.cost)
        new_value = self.portfolio.get_value(next_price)

        # reward is value change minus costs
        reward = (new_value - old_value) - cost

        self.idx += 1
        done = self.idx >= len(self.df) - 1
        info = {"portfolio_value": new_value, "trade_cost": cost}

        return self._get_state(), float(reward), done, info


def create_env_from_data(processed_path: str = "data/processed/crypto_data.csv", coin: str = "bitcoin") -> TradingEnv:
    """Helper to create env from processed CSV data."""
    df = pd.read_csv(processed_path, index_col=[0, 1], parse_dates=[1])
    try:
        df = df.loc[coin].sort_index()
    except Exception:
        raise ValueError(f"Could not load data for {coin}")
    return TradingEnv(df)


if __name__ == "__main__":
    # Quick env demo
    try:
        env = create_env_from_data()
        state = env.reset()
        print("State shape:", state.shape)
        next_state, reward, done, info = env.step(1)  # BUY
        print("After BUY:", info)
        next_state, reward, done, info = env.step(2)  # SELL
        print("After SELL:", info)
    except Exception as e:
        print("Demo failed:", e)
