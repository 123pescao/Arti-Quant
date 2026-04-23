"""
Strategy interface.

All production strategies must inherit from Strategy and implement evaluate().

Anti-lookahead contract (enforced at this boundary):
  The caller MUST strip the last (incomplete) candle before passing candles in.
  evaluate() receives candles[-1] as the most recently CLOSED candle.
  It must never read beyond candles[-1].

  In live trading: the orchestrator calls evaluate() at 22:10 UTC after the
  daily candle closes, passing all closed candles with the in-progress candle stripped.

  In backtesting: BacktestEngine simulates this by calling compute_signals() with
  shift(2)/shift(1) offsets, which is equivalent to stripping the last bar.

  Both must agree: a signal fired on candle[N-1] enters on candle[N]'s open.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .models import Signal


class Strategy(ABC):
    """
    Abstract base for all production trading strategies.

    Implementations must be:
      - Pure: no side effects, no external I/O
      - Deterministic: same candles → same signal
      - Stateless: no internal mutable state between calls
    """

    @abstractmethod
    def evaluate(
        self,
        pair: str,
        candles: pd.DataFrame,
    ) -> Optional[Signal]:
        """
        Evaluate candle history and return a Signal or None.

        Args:
            pair:    FX pair identifier (e.g. "EUR_USD")
            candles: OHLCV DataFrame with columns [open, high, low, close, volume].
                     Index must be DatetimeIndex (UTC).
                     IMPORTANT: caller must have stripped the in-progress candle.
                     candles.iloc[-1] is the most recently CLOSED bar.

        Returns:
            Signal if a new entry/exit signal is generated, None if no action.
            Direction.FLAT signals an exit from an existing position.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @property
    @abstractmethod
    def min_candles(self) -> int:
        """
        Minimum number of closed candles required before a signal can be generated.
        evaluate() returns None if len(candles) < min_candles.
        """
