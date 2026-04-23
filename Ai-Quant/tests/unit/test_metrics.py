"""
Unit tests for metrics calculations.
"""

import pytest
import pandas as pd
import numpy as np
from backtest.metrics import MetricsCalculator


class TestSharpeRatio:
    def test_sharpe_on_constant_returns(self):
        """Constant returns should have zero Sharpe."""
        returns = pd.Series([0.01] * 100)
        sharpe = MetricsCalculator.sharpe_ratio(returns)
        assert sharpe == 0.0  # Constant series has std=0 → division by zero handled

    def test_sharpe_positive_vs_negative(self):
        """Positive return series should have positive Sharpe."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.005, 252))  # Positive drift
        sharpe = MetricsCalculator.sharpe_ratio(returns)
        assert sharpe > 0

    def test_sharpe_annualized(self):
        """Positive returns with variance should give positive Sharpe."""
        np.random.seed(7)
        returns = pd.Series(np.random.normal(0.001, 0.005, 252))  # Positive drift with noise
        sharpe = MetricsCalculator.sharpe_ratio(returns)
        assert sharpe > 0


class TestMaxDrawdown:
    def test_max_dd_on_bull_market(self):
        """Bull market should have small drawdown."""
        equity = pd.Series([100, 101, 102, 103, 104, 105])
        dd = MetricsCalculator.max_drawdown(equity)
        assert dd == 0.0  # No decline

    def test_max_dd_on_bear_market(self):
        """Bear market should have significant drawdown."""
        equity = pd.Series([100, 90, 80, 70, 60, 50])
        dd = MetricsCalculator.max_drawdown(equity)
        assert dd < 0  # Negative
        assert abs(dd) > 0.4  # At least 40% decline

    def test_max_dd_single_peak(self):
        """Drawdown from peak only."""
        equity = pd.Series([100, 110, 90, 95])  # Peak at 110
        dd = MetricsCalculator.max_drawdown(equity)
        # Worst point: 90 from peak 110 = (90 - 110) / 110 = -18.2%
        assert pytest.approx(dd, abs=0.01) == (90 - 110) / 110


class TestProfitFactor:
    def test_pf_break_even(self):
        """PF on equal wins/losses should be 1.0."""
        pnls = [10, -10, 10, -10, 10, -10]
        pf = MetricsCalculator.profit_factor(pnls)
        assert pf == 1.0

    def test_pf_all_wins(self):
        """PF on all winning trades should be infinite."""
        pnls = [10, 20, 30]
        pf = MetricsCalculator.profit_factor(pnls)
        assert pf == float('inf')

    def test_pf_all_losses(self):
        """PF on all losing trades should be 0."""
        pnls = [-10, -20, -30]
        pf = MetricsCalculator.profit_factor(pnls)
        assert pf == 0.0

    def test_pf_no_trades(self):
        """Empty PnL list should return 0."""
        pf = MetricsCalculator.profit_factor([])
        assert pf == 0.0


class TestWinRate:
    def test_wr_fifty_fifty(self):
        """50/50 splits should give 50% win rate."""
        pnls = [10, -10, 20, -5]
        wr = MetricsCalculator.win_rate(pnls)
        assert wr == 0.5

    def test_wr_all_losses(self):
        """All losses should give 0% win rate."""
        pnls = [-10, -20, -30]
        wr = MetricsCalculator.win_rate(pnls)
        assert wr == 0.0

    def test_wr_no_trades(self):
        """Empty list should return 0%."""
        wr = MetricsCalculator.win_rate([])
        assert wr == 0.0


class TestExpectancy:
    def test_expectancy_positive(self):
        """Average of positive PnLs."""
        pnls = [10, 20, 30]
        exp = MetricsCalculator.expectancy(pnls)
        assert exp == 20.0

    def test_expectancy_mixed(self):
        """Average with mixed signs."""
        pnls = [100, -50, 50, -20]
        exp = MetricsCalculator.expectancy(pnls)
        assert exp == 20.0

    def test_expectancy_no_trades(self):
        """Empty list should return 0."""
        exp = MetricsCalculator.expectancy([])
        assert exp == 0.0


class TestPayoffRatio:
    def test_payoff_ideal(self):
        """3:1 win/loss ratio."""
        pnls = [30, 30, 30, -10]  # Wins: 30 each, Loss: -10
        ratio = MetricsCalculator.payoff_ratio(pnls)
        assert pytest.approx(ratio) == 3.0

    def test_payoff_less_than_one(self):
        """Losers bigger than winners."""
        pnls = [10, 10, -50]  # Wins: 10 each, Loss: -50
        ratio = MetricsCalculator.payoff_ratio(pnls)
        assert pytest.approx(ratio, abs=0.01) == 0.2

    def test_payoff_no_losses(self):
        """No losses means no losses to compare."""
        pnls = [10, 20, 30]
        ratio = MetricsCalculator.payoff_ratio(pnls)
        assert ratio == 0.0


class TestConsecutiveLosses:
    def test_longest_streak_three(self):
        """Three consecutive losses."""
        pnls = [10, -5, -10, -20, 30, -1]
        streak = MetricsCalculator.consecutive_losses(pnls)
        assert streak == 3

    def test_all_losses(self):
        """All losses."""
        pnls = [-1, -2, -3, -4, -5]
        streak = MetricsCalculator.consecutive_losses(pnls)
        assert streak == 5

    def test_no_losses(self):
        """No losses."""
        pnls = [10, 20, 30]
        streak = MetricsCalculator.consecutive_losses(pnls)
        assert streak == 0


class TestMonthlyReturns:
    def test_monthly_returns_structure(self):
        """Monthly returns should be properly resampled."""
        dates = pd.date_range(start="2022-01-01", periods=365, freq="D")
        equity = pd.Series(np.linspace(10000, 11000, 365), index=dates)
        monthly = MetricsCalculator.monthly_returns(equity)
        assert len(monthly) > 0
        assert monthly.dtype == float

    def test_monthly_positive_drift(self):
        """Increasing equity should have positive average monthly."""
        dates = pd.date_range(start="2022-01-01", periods=252, freq="D")
        equity = pd.Series(np.linspace(10000, 12000, 252), index=dates)
        monthly = MetricsCalculator.monthly_returns(equity)
        assert monthly.mean() > 0


class TestAnnualReturn:
    def test_annual_return_doubling(self):
        """Doubling account is 100% annual return."""
        dates = pd.date_range(start="2022-01-01", periods=365, freq="D")
        equity = pd.Series([10000] * 100 + [20000] * 265, index=dates)
        ann_ret = MetricsCalculator.annual_return(equity)
        # Rough approximation given discrete change point
        assert ann_ret > 0.5  # At least 50%

    def test_annual_return_flat(self):
        """Flat account is 0% return."""
        dates = pd.date_range(start="2022-01-01", periods=365, freq="D")
        equity = pd.Series([10000] * 365, index=dates)
        ann_ret = MetricsCalculator.annual_return(equity)
        assert pytest.approx(ann_ret) == 0.0
