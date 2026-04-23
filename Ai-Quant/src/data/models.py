"""
Data layer typed models.

These are the raw API response types returned by the OANDA client.
They are kept separate from the domain models (risk.models, strategy.models)
so that the data layer can be swapped without touching business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from risk.models import OpenPosition


@dataclass(frozen=True)
class Candle:
    """One OANDA D1 bar."""
    pair:     str
    time:     datetime   # UTC bar-open timestamp
    open:     float
    high:     float
    low:      float
    close:    float
    volume:   int
    complete: bool       # False if bar is still forming; in-progress bars must be excluded


@dataclass(frozen=True)
class AccountSummary:
    """
    Raw account snapshot from the OANDA account summary endpoint.

    balance:        Realized account balance (no unrealized P&L).
    nav:            Net asset value = balance + unrealized_pl (equity).
    unrealized_pl:  Sum of unrealized P&L across all open positions.
    open_positions: Currently open trades mapped to OpenPosition objects.
    """
    balance:        float
    nav:            float
    unrealized_pl:  float
    open_positions: tuple[OpenPosition, ...]
