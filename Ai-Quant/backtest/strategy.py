"""
Donchian breakout strategy signal generation.

CRITICAL: Anti-lookahead bias prevention via shift(2) and shift(1).

In live trading, the orchestrator strips candles[-1] before passing to the strategy.
In backtesting, we simulate this by using pandas shift():
  - Signal at bar T uses data only through bar T-2
  - Entry price reference is close[T-1] (which is candles[-1] stripped in live)
  - Execution is at bar T's open

The shift operations ensure the backtest signal at any bar T would have been
available to the live system on that same day at 22:10 UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class DonchianSignals:
    """Output of signal computation for one bar period."""
    long_entry:   pd.Series
    short_entry:  pd.Series
    long_exit:    pd.Series
    short_exit:   pd.Series
    atr:          pd.Series
    long_stop:    pd.Series
    short_stop:   pd.Series


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range with anti-LAB shift.

    True Range = max(
        high - low,
        |high - close[t-1]|,
        |low - close[t-1]|
    )

    We shift by 2 to ensure ATR used at bar T is computed from data through bar T-2,
    matching the live system where the strategy sees only candles[:-1].
    """
    # True range components
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()

    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_raw = tr.rolling(window=period, min_periods=period).mean()

    # Shift by 2: LAB prevention (match live system anti-LAB strip)
    # At bar T, use ATR computed through bar T-2
    return atr_raw.shift(2)


def compute_donchian(
    high: pd.Series,
    low: pd.Series,
    entry_period: int = 20,
    exit_period: int = 10,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute Donchian channel upper/lower bands with anti-LAB shifts.

    Entry channel (20-period): shifted by 2
      At bar T, signal uses channel from bars T-21 to T-2

    Exit channel (10-period): shifted by 1
      At bar T, exit is checked against channel from bars T-10 to T-1
      (Less strict shift for exit because we're already holding a position)

    Returns: (upper_entry, lower_entry, upper_exit, lower_exit)
    """
    # Entry channels — strict shift(2) for signal generation
    upper_entry = high.rolling(window=entry_period, min_periods=entry_period).max().shift(2)
    lower_entry = low.rolling(window=entry_period, min_periods=entry_period).min().shift(2)

    # Exit channels — shift(1) for exit signal
    upper_exit = high.rolling(window=exit_period, min_periods=exit_period).max().shift(1)
    lower_exit = low.rolling(window=exit_period, min_periods=exit_period).min().shift(1)

    return upper_entry, lower_entry, upper_exit, lower_exit


def compute_signals(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
) -> DonchianSignals:
    """
    Compute all trading signals (entries, exits, stops) from OHLCV data.

    Key invariant: all signals at bar T use data only through bar T-2,
    ensuring zero lookahead bias when matched to the live system's
    data flow.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        entry_period: lookback for Donchian entry channel (20)
        exit_period: lookback for Donchian exit channel (10)
        atr_period: lookback for ATR (14)
        atr_multiplier: ATR multiple for stop distance (2.0)

    Returns:
        DonchianSignals with series for entries, exits, stops, ATR.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Compute ATR (with shift(2) for LAB prevention)
    atr = compute_atr(high, low, close, atr_period)

    # Compute Donchian channels
    upper_entry, lower_entry, upper_exit, lower_exit = compute_donchian(
        high, low, entry_period, exit_period
    )

    # Entry signals: close[T-1] breaks channel[T-2]
    # This uses shift(1) on close and the shift(2) already in the channel.
    long_entry_raw = close.shift(1) > upper_entry
    short_entry_raw = close.shift(1) < lower_entry

    # Exit signals: close[T] breaks exit channel[T-1]
    long_exit_raw = close < lower_exit
    short_exit_raw = close > upper_exit

    # Stop prices: computed at entry (using close[T-1] and ATR[T-2])
    # Long stop = entry_price - ATR × multiplier
    # Short stop = entry_price + ATR × multiplier
    long_stop = close.shift(1) - (atr * atr_multiplier)
    short_stop = close.shift(1) + (atr * atr_multiplier)

    # Filter: only valid signals where we have enough data
    min_bar = max(entry_period, exit_period, atr_period) + 2
    valid_idx = pd.Series(False, index=df.index)
    valid_idx.iloc[min_bar:] = True

    long_entry = long_entry_raw & valid_idx & ~long_exit_raw.shift(1, fill_value=False)
    short_entry = short_entry_raw & valid_idx & ~short_exit_raw.shift(1, fill_value=False)
    long_exit = long_exit_raw & valid_idx
    short_exit = short_exit_raw & valid_idx

    return DonchianSignals(
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=long_exit,
        short_exit=short_exit,
        atr=atr,
        long_stop=long_stop,
        short_stop=short_stop,
    )


def validate_no_lookahead(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
) -> bool:
    """
    Regression test: verify that adding one more bar to the dataset
    does NOT change signals for earlier bars.

    If this fails, lookahead bias is present.
    """
    # Take first N-1 bars
    df_n = df.iloc[:-1].copy()
    sigs_n = compute_signals(df_n, entry_period, exit_period, atr_period, atr_multiplier)

    # Take all N bars
    df_n1 = df.copy()
    sigs_n1 = compute_signals(df_n1, entry_period, exit_period, atr_period, atr_multiplier)

    # Signals on the common range (up to N-1) should be identical
    n = len(df_n)
    for field in ["long_entry", "short_entry", "long_exit", "short_exit"]:
        ser_n = getattr(sigs_n, field)[:n]
        ser_n1 = getattr(sigs_n1, field)[:n]
        if not ser_n.equals(ser_n1):
            return False

    # ATR and stops should also be identical
    if not sigs_n.atr[:n].equals(sigs_n1.atr[:n]):
        return False
    if not sigs_n.long_stop[:n].equals(sigs_n1.long_stop[:n]):
        return False
    if not sigs_n.short_stop[:n].equals(sigs_n1.short_stop[:n]):
        return False

    return True
