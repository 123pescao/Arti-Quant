"""
Performance metrics and analysis.
Sharpe, max drawdown, profit factor, expectancy, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class MetricsCalculator:
    """Compute standard performance metrics from backtest results."""

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Annualized Sharpe ratio.
        Assumes daily returns.
        """
        if len(returns) < 2:
            return 0.0
        excess = returns - (risk_free_rate / periods_per_year)
        std = excess.std()
        if std < 1e-10 or np.isnan(std):
            return 0.0
        return (excess.mean() / std) * np.sqrt(periods_per_year)

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """
        Maximum drawdown as percentage.
        Negative value representing peak-to-trough decline.
        """
        if len(equity) < 2:
            return 0.0
        cummax = equity.cummax()
        dd = (equity - cummax) / cummax
        return dd.min()

    @staticmethod
    def profit_factor(trade_pnls: list[float]) -> float:
        """
        Profit factor = sum(wins) / abs(sum(losses))
        Range: 0 to ∞
        > 1.0 = profitable, < 1.0 = unprofitable
        """
        if len(trade_pnls) == 0:
            return 0.0
        gains = sum(pnl for pnl in trade_pnls if pnl > 0)
        losses = sum(pnl for pnl in trade_pnls if pnl < 0)
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        return gains / abs(losses)

    @staticmethod
    def win_rate(trade_pnls: list[float]) -> float:
        """Percentage of trades that were profitable."""
        if len(trade_pnls) == 0:
            return 0.0
        wins = sum(1 for pnl in trade_pnls if pnl > 0)
        return wins / len(trade_pnls)

    @staticmethod
    def expectancy(trade_pnls: list[float]) -> float:
        """Average profit per trade (in pips)."""
        if len(trade_pnls) == 0:
            return 0.0
        return np.mean(trade_pnls)

    @staticmethod
    def payoff_ratio(trade_pnls: list[float]) -> float:
        """
        Average win / average loss magnitude.
        Higher is better (winners should be bigger than losers).
        """
        if len(trade_pnls) == 0:
            return 0.0
        wins = [pnl for pnl in trade_pnls if pnl > 0]
        losses = [pnl for pnl in trade_pnls if pnl < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        return np.mean(wins) / abs(np.mean(losses))

    @staticmethod
    def consecutive_losses(trade_pnls: list[float]) -> int:
        """Longest streak of consecutive losing trades."""
        if len(trade_pnls) == 0:
            return 0
        max_streak = 0
        current_streak = 0
        for pnl in trade_pnls:
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    @staticmethod
    def monthly_returns(equity: pd.Series) -> pd.Series:
        """
        Returns aggregated by month.
        Useful for distribution analysis.
        """
        if len(equity) < 2:
            return pd.Series(dtype=float)
        # Resample to month-end
        monthly = equity.resample("ME").last()
        return monthly.pct_change().dropna()

    @staticmethod
    def recovery_factor(total_pnl: float, max_dd: float) -> float:
        """
        Recovery factor = net profit / max drawdown.
        Higher is better. Measures efficiency of recovery.
        """
        if max_dd == 0:
            return 0.0
        return total_pnl / abs(max_dd)

    @staticmethod
    def calmar_ratio(annual_return: float, max_dd: float) -> float:
        """
        Calmar ratio = annual return / max drawdown.
        Similar to recovery factor but on annualized basis.
        """
        if max_dd == 0:
            return 0.0
        return annual_return / abs(max_dd)

    @staticmethod
    def per_pair_stats(trades_df: pd.DataFrame) -> dict[str, dict]:
        """
        Breakdown of metrics by pair.
        """
        results = {}
        for pair in trades_df["pair"].unique():
            pair_trades = trades_df[trades_df["pair"] == pair]
            pnls = pair_trades["pnl_pips"].tolist()
            results[pair] = {
                "count": len(pair_trades),
                "win_count": sum(1 for p in pnls if p > 0),
                "loss_count": sum(1 for p in pnls if p < 0),
                "win_rate": MetricsCalculator.win_rate(pnls),
                "expectancy": MetricsCalculator.expectancy(pnls),
                "profit_factor": MetricsCalculator.profit_factor(pnls),
            }
        return results

    @staticmethod
    def annual_return(equity: pd.Series) -> float:
        """Annualized return (simple)."""
        if len(equity) < 2:
            return 0.0
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        # Estimate days
        if isinstance(equity.index, pd.DatetimeIndex):
            days = (equity.index[-1] - equity.index[0]).days
            return total_return * (365 / days) if days > 0 else 0.0
        return 0.0
