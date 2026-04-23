"""
Production Donchian breakout strategy.

Anti-lookahead boundary contract:
  evaluate() receives candles with the in-progress bar already stripped by the caller.
  Signal logic uses only candles[-2] for the breakout check and candles[-3..N] for
  channel computation — consistent with the backtest's shift(2) convention.

  candles[-1] = most recently closed bar (T)
  candles[-2] = bar before last (T-1) → provides the close that broke the channel
  channel computed from candles[0 .. -3] (through T-2)

  This exactly mirrors backtest/strategy.py:
    close.shift(1) > upper_entry  (upper_entry is shift(2) of the rolling max)
  Equivalent here:
    candles[-2]["close"] > channel_high_computed_through_candles[-3]

Exit signal:
  close[T] < lower_exit channel (shift(1) in backtest)
  Equivalent here:
    candles[-1]["close"] < exit_channel_low_computed_through_candles[-2]

Stop loss:
  ATR computed from candles[0..-3], multiplied by atr_multiplier (2.0)
  stop = reference_close ± atr * atr_multiplier
  reference_close = candles[-2]["close"]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy
from .models import Direction, Signal


@dataclass(frozen=True)
class DonchianConfig:
    """Immutable configuration for the Donchian strategy."""
    entry_period:   int   = 20
    exit_period:    int   = 10
    atr_period:     int   = 14
    atr_multiplier: float = 2.0   # must not be changed in v1

    def __post_init__(self) -> None:
        if self.atr_multiplier != 2.0:
            raise ValueError(
                f"atr_multiplier must be 2.0 in v1, got {self.atr_multiplier}"
            )
        if self.entry_period < 2:
            raise ValueError("entry_period must be >= 2")
        if self.exit_period < 2:
            raise ValueError("exit_period must be >= 2")
        if self.atr_period < 2:
            raise ValueError("atr_period must be >= 2")


class DonchianStrategy(Strategy):
    """
    Donchian channel breakout strategy — production implementation.

    Entry: close[T-1] breaks the entry channel computed through T-2.
    Exit:  close[T]   breaks the exit  channel computed through T-1.
    Stop:  entry_close ± ATR(T-2) × atr_multiplier

    Produces at most one signal per evaluate() call:
      - LONG or SHORT entry if a new breakout is detected
      - FLAT if the current close breaks the exit channel on the opposite side
      - None if neither entry nor exit condition is met
    """

    def __init__(self, config: Optional[DonchianConfig] = None) -> None:
        self._config = config or DonchianConfig()

    @property
    def name(self) -> str:
        return (
            f"Donchian({self._config.entry_period}/{self._config.exit_period}/"
            f"ATR{self._config.atr_period}x{self._config.atr_multiplier})"
        )

    @property
    def min_candles(self) -> int:
        # Need enough bars to compute entry channel + ATR + 2-bar anti-LAB offset
        return max(self._config.entry_period, self._config.atr_period) + 3

    @property
    def config(self) -> DonchianConfig:
        return self._config

    def evaluate(
        self,
        pair: str,
        candles: pd.DataFrame,
    ) -> Optional[Signal]:
        """
        Evaluate candle history for pair and return a Signal or None.

        Caller contract: candles.iloc[-1] is the last CLOSED bar (in-progress stripped).
        """
        if len(candles) < self.min_candles:
            return None

        cfg = self._config

        high  = candles["high"]
        low   = candles["low"]
        close = candles["close"]

        # --- Entry channel: computed through candles[-3] (T-2 in backtest shift convention) ---
        # candles[:-2] excludes the last 2 bars, so the rolling window ends at T-2
        entry_high = high.iloc[:-2].rolling(window=cfg.entry_period, min_periods=cfg.entry_period).max().iloc[-1]
        entry_low  = low.iloc[:-2].rolling(window=cfg.exit_period,   min_periods=cfg.exit_period).min().iloc[-1]

        # --- Exit channel: computed through candles[-2] (T-1 in backtest shift convention) ---
        exit_high = high.iloc[:-1].rolling(window=cfg.exit_period,  min_periods=cfg.exit_period).max().iloc[-1]
        exit_low  = low.iloc[:-1].rolling(window=cfg.exit_period,   min_periods=cfg.exit_period).min().iloc[-1]

        # --- ATR: computed through candles[-3] (shift(2) in backtest) ---
        atr = self._compute_atr(high.iloc[:-2], low.iloc[:-2], close.iloc[:-2], cfg.atr_period)

        if pd.isna(entry_high) or pd.isna(entry_low) or pd.isna(atr):
            return None

        # Reference close for signal: candles[-2] (the bar whose close broke the channel)
        ref_close = close.iloc[-2]
        # Current close for exit check: candles[-1]
        cur_close = close.iloc[-1]

        signal_time = candles.index[-1]
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)

        source_bar = len(candles) - 1

        # --- Entry signals ---
        long_entry  = ref_close > entry_high
        short_entry = ref_close < entry_low

        # --- Exit signals (checked against current close) ---
        long_exit  = cur_close < exit_low
        short_exit = cur_close > exit_high

        # Guard: skip entry if exit already firing (same bar would flip immediately)
        if long_entry and not long_exit:
            stop = ref_close - atr * cfg.atr_multiplier
            return Signal(
                pair=pair,
                direction=Direction.LONG,
                entry_price=ref_close,
                stop_loss=stop,
                atr=atr,
                signal_time=signal_time,
                source_bar=source_bar,
            )

        if short_entry and not short_exit:
            stop = ref_close + atr * cfg.atr_multiplier
            return Signal(
                pair=pair,
                direction=Direction.SHORT,
                entry_price=ref_close,
                stop_loss=stop,
                atr=atr,
                signal_time=signal_time,
                source_bar=source_bar,
            )

        # --- FLAT: exit signal without a new entry ---
        if long_exit or short_exit:
            return Signal(
                pair=pair,
                direction=Direction.FLAT,
                entry_price=cur_close,
                stop_loss=None,
                atr=atr,
                signal_time=signal_time,
                source_bar=source_bar,
            )

        return None

    @staticmethod
    def _compute_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> float:
        """
        Compute the most recent ATR value from the provided (already-sliced) series.
        These series already exclude the last 2 bars (caller's anti-LAB responsibility).
        Returns NaN if insufficient data.
        """
        if len(close) < period + 1:
            return float("nan")

        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low  - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr_series = tr.rolling(window=period, min_periods=period).mean()
        val = atr_series.iloc[-1]
        return float(val) if not pd.isna(val) else float("nan")
