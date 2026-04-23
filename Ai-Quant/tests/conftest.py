"""
Pytest fixtures for backtest testing.
Provides sample data, configs, etc.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.data import DataLoader
from backtest.costs import CostConfig, SpreadScenario
from backtest.engine import BacktestEngine


@pytest.fixture
def sample_ohlcv_data():
    """Generate 500 days of synthetic daily data."""
    dates = pd.bdate_range(start="2020-01-01", periods=500, freq="B")
    np.random.seed(42)

    # Random walk
    returns = np.random.normal(0.0005, 0.008, 500)
    close = pd.Series(1.10 * np.exp(np.cumsum(returns)), index=dates)

    open_ = close.shift(1).fillna(1.10)
    high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.003, 500))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.003, 500))
    volume = np.random.uniform(100_000, 1_000_000, 500)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


@pytest.fixture
def sample_data_dict(sample_ohlcv_data):
    """Dict of OHLCV for all pairs."""
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]
    return {pair: sample_ohlcv_data.copy() for pair in pairs}


@pytest.fixture
def cost_config_normal():
    """Normal spread scenario."""
    return CostConfig(scenario=SpreadScenario.NORMAL)


@pytest.fixture
def cost_config_stress():
    """Stressed spread scenario (2x)."""
    return CostConfig(scenario=SpreadScenario.STRESS_2)


@pytest.fixture
def backtest_engine():
    """Default backtest engine."""
    return BacktestEngine(
        pairs=["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"],
        risk_per_trade_pct=0.0025,
        max_positions=2,
        max_usd_exposure=2,
    )


@pytest.fixture
def strategy_params():
    """Default strategy parameters."""
    return {
        "entry_period": 20,
        "exit_period": 10,
        "atr_period": 14,
        "atr_multiplier": 2.0,
    }
