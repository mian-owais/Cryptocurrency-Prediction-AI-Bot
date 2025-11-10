"""
test_trading_env.py
------------------
Test suite for trading environment functionality:
- Portfolio management
- State calculations
- Trading mechanics
- Reward computation
"""

import unittest
import numpy as np
import pandas as pd
from src.trading_env import Portfolio, TradingEnv


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(cash=1000.0)

    def test_initial_state(self):
        self.assertEqual(self.portfolio.cash, 1000.0)
        self.assertEqual(self.portfolio.position, 0.0)
        self.assertEqual(len(self.portfolio.trades), 0)

    def test_get_value(self):
        self.portfolio.position = 2.0
        value = self.portfolio.get_value(price=100.0)
        self.assertEqual(value, 1200.0)  # 1000 cash + 2 * 100 position

    def test_buy_trade(self):
        pnl, cost = self.portfolio.trade(
            price=100.0, action=1, trade_size=0.1, cost=0.001)
        self.assertLess(self.portfolio.cash, 1000.0)
        self.assertGreater(self.portfolio.position, 0.0)
        self.assertEqual(len(self.portfolio.trades), 1)

    def test_sell_trade(self):
        # First buy
        self.portfolio.trade(price=100.0, action=1, trade_size=0.1, cost=0.001)
        initial_position = self.portfolio.position
        # Then sell
        pnl, cost = self.portfolio.trade(
            price=110.0, action=2, trade_size=0.1, cost=0.001)
        self.assertLess(self.portfolio.position, initial_position)
        self.assertEqual(len(self.portfolio.trades), 2)

    def test_invalid_trades(self):
        # Sell with no position
        pnl, cost = self.portfolio.trade(
            price=100.0, action=2, trade_size=0.1, cost=0.001)
        self.assertEqual(pnl, 0.0)
        self.assertEqual(cost, 0.0)
        # Buy with no cash
        self.portfolio.cash = 0.0
        pnl, cost = self.portfolio.trade(
            price=100.0, action=1, trade_size=0.1, cost=0.001)
        self.assertEqual(pnl, 0.0)
        self.assertEqual(cost, 0.0)


class TestTradingEnv(unittest.TestCase):
    def setUp(self):
        # Create sample price data
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        prices = np.linspace(100, 200, 100) + np.random.randn(100) * 5
        self.df = pd.DataFrame({
            "price": prices,
            "volume": np.random.rand(100) * 1000,
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        }, index=dates)
        self.env = TradingEnv(self.df, window=10)

    def test_init(self):
        self.assertIsNotNone(self.env.features)
        self.assertGreater(len(self.env.features.columns), 1)

    def test_reset(self):
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        # Features + position
        self.assertEqual(len(state), len(self.env.features.columns) + 1)

    def test_step(self):
        self.env.reset()
        next_state, reward, done, info = self.env.step(1)  # BUY
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("portfolio_value", info)
        self.assertIn("trade_cost", info)

    def test_episode(self):
        state = self.env.reset()
        done = False
        step = 0
        while not done and step < 1000:  # Avoid infinite loops
            action = np.random.randint(0, 3)  # Random action
            state, reward, done, info = self.env.step(action)
            step += 1
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
