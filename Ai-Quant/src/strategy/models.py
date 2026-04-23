"""
Signal model: the output of the strategy layer.

A Signal is the unit of intent passed from the strategy to the risk engine
and then to the execution layer. It is immutable after construction.

Design constraints:
  - take_profit is always None in v1 (no TP rule)
  - stop_loss is always set (ATR-based, ATR multiplier = 2.0)
  - direction is always LONG or SHORT; FLAT signals an exit
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"   # exit / no trade


@dataclass(frozen=True)
class Signal:
    """
    Trading signal produced by the strategy for a single pair.

    Fields:
        pair:         FX pair identifier (e.g. "EUR_USD")
        direction:    LONG, SHORT, or FLAT (FLAT = exit existing position)
        entry_price:  Reference close price that triggered the signal.
                      Actual fill happens at next bar open with costs applied.
        stop_loss:    Absolute price for the stop order. Required for LONG/SHORT.
                      None only for FLAT signals.
        take_profit:  Always None in v1.
        atr:          ATR value used to compute the stop. Stored for auditability.
        signal_time:  UTC timestamp when the signal was generated (candle close time).
        source_bar:   Index of the candle that closed to produce this signal.
    """
    pair:         str
    direction:    Direction
    entry_price:  float
    stop_loss:    Optional[float]
    atr:          Optional[float]
    signal_time:  datetime
    source_bar:   int
    take_profit:  Optional[float] = None   # v1: always None

    def __post_init__(self) -> None:
        if self.direction in (Direction.LONG, Direction.SHORT):
            if self.stop_loss is None:
                raise ValueError(f"Signal for {self.pair} {self.direction} must have stop_loss set")
            if self.stop_loss <= 0:
                raise ValueError(f"stop_loss must be positive, got {self.stop_loss}")
            if self.entry_price <= 0:
                raise ValueError(f"entry_price must be positive, got {self.entry_price}")
            if self.direction == Direction.LONG and self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"LONG stop_loss {self.stop_loss} must be below entry_price {self.entry_price}"
                )
            if self.direction == Direction.SHORT and self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"SHORT stop_loss {self.stop_loss} must be above entry_price {self.entry_price}"
                )
        if self.take_profit is not None:
            raise ValueError("take_profit must be None in v1 — no TP rule")
