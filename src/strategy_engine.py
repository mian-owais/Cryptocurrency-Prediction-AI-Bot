"""
strategy_engine.py
------------------
Purpose:
 - Convert model forecasts into actionable trade signals (BUY / SELL / HOLD).
 - Provide an interface to integrate portfolio optimization (CSP placeholder).

Key functions:
 - generate_trade_signal(current_price, predicted_price, thresholds): returns BUY/SELL/HOLD
 - csp_portfolio_optimization(assets, cov_matrix, returns): placeholder to plug into convex/simulation optimizer

Notes:
 - Signal rules in this module should be deterministic and easy to backtest.
 - Keep execution logic separate from signal generation.
"""

from typing import Dict, Any
import numpy as np
from typing import List, Optional
import itertools


def generate_trade_signal(current_price: float, predicted_price: float, pct_threshold: float = 0.02) -> str:
    """Simple rule-based signal generator.

    Args:
        current_price: latest observed price
        predicted_price: model's predicted next-step price
        pct_threshold: percent change threshold for BUY/SELL

    Returns:
        'BUY', 'SELL', or 'HOLD'
    """
    change = (predicted_price - current_price) / (current_price + 1e-9)
    if change > pct_threshold:
        return "BUY"
    elif change < -pct_threshold:
        return "SELL"
    else:
        return "HOLD"


def csp_portfolio_optimization(assets: List[str], expected_returns: np.ndarray, historical_prices: Optional[Dict[str, np.ndarray]] = None, cash_min: float = 0.1, max_position: float = 0.5, drawdown_limit: float = 0.05, step: float = 0.1) -> Dict[str, float]:
    """A discrete-search CSP-like portfolio optimizer.

    We discretize allocations in `step` increments (e.g., 0.1 = 10%) and search combinations
    that satisfy constraints:
      - cash >= cash_min
      - each position <= max_position
      - (optional) simulated max drawdown <= drawdown_limit (if historical_prices provided)

    Objective: maximize expected profit approximated by expected_returns @ allocation_vector

    Returns allocation dict mapping asset->allocation (summing to <= 1.0). Cash is implicit.
    """
    n = len(assets)
    if n == 0:
        return {}

    # build discrete allocation grid per asset (0..max_position with step)
    levels = int(max_position / step) + 1
    # may exceed max_position check later
    choices = [i * step for i in range(levels + 1)]

    best_alloc = None
    best_score = -np.inf

    # brute-force search — combinatorial but small if step is coarse
    for combo in itertools.product(choices, repeat=n):
        alloc = np.array(combo)
        # enforce per-asset max
        if np.any(alloc > max_position + 1e-9):
            continue
        total_alloc = alloc.sum()
        cash = 1.0 - total_alloc
        if cash < cash_min - 1e-9:
            continue

        # optional drawdown check
        if historical_prices is not None and drawdown_limit is not None:
            # rough portfolio price series by weighted sum of normalized series
            try:
                series = None
                for a, asset in zip(alloc, assets):
                    price = np.asarray(historical_prices.get(asset, []))
                    if price.size == 0:
                        raise ValueError("Missing historical price for asset")
                    norm = price / price[0]
                    if series is None:
                        series = a * norm
                    else:
                        series = series + a * norm
                # compute max drawdown
                peak = np.maximum.accumulate(series)
                dd = np.max((peak - series) / (peak + 1e-9))
                if dd > drawdown_limit:
                    continue
            except Exception:
                # if any issue, skip drawdown enforcement
                pass

        # score by expected_returns dot allocation
        score = float(np.dot(expected_returns, alloc))
        if score > best_score:
            best_score = score
            best_alloc = alloc.copy()

    if best_alloc is None:
        # fallback to equal-weight but respecting constraints
        alloc = np.minimum(np.ones(n) / n, max_position)
        total_alloc = alloc.sum()
        if 1.0 - total_alloc < cash_min:
            # reduce allocations proportionally
            factor = (1.0 - cash_min) / total_alloc if total_alloc > 0 else 0.0
            alloc = alloc * factor
        return {asset: float(a) for asset, a in zip(assets, alloc)}

    return {asset: float(a) for asset, a in zip(assets, best_alloc)}


if __name__ == "__main__":
    print("strategy_engine imported — implement backtest and optimizer later")
