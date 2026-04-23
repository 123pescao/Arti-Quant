"""
Execution layer data models.

These are the typed contracts between the executor and the rest of the system.
All are frozen dataclasses (immutable) except where explicitly noted.

Order lifecycle:
  Signal → RiskDecision → Order → Fill (or OrderError)

SlippageRecord: persisted for every fill to track real vs expected prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"


class FillStatus(str, Enum):
    FILLED        = "filled"
    REJECTED      = "rejected"
    ALREADY_OPEN  = "already_open"   # position already exists for this pair (any direction)


@dataclass(frozen=True)
class Order:
    """
    An order ready to be submitted to the broker.

    pair:          OANDA instrument name (e.g. "EUR_USD")
    side:          BUY or SELL
    units:         Positive integer number of units
    stop_loss:     Stop price to attach as a broker-side stop loss order
    signal_id:     Opaque string key for idempotency (e.g. "EUR_USD_LONG_20240115")
    created_at:    UTC timestamp the order was created
    """
    pair:       str
    side:       OrderSide
    units:      int
    stop_loss:  float
    signal_id:  str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.units <= 0:
            raise ValueError(f"Order units must be positive, got {self.units}")
        if self.stop_loss <= 0:
            raise ValueError(f"Order stop_loss must be positive, got {self.stop_loss}")


@dataclass(frozen=True)
class Fill:
    """
    Confirmation of a filled order returned by the broker.

    order:         The original order that was submitted
    fill_price:    Actual execution price (may differ from mid due to spread/slippage).
                   Set to 0.0 when status==ALREADY_OPEN (fill price unknown after timeout
                   reconciliation). Callers must branch on status before using fill_price.
    broker_trade_id: Broker-assigned trade identifier for reconciliation
    filled_at:     UTC timestamp of fill confirmation
    status:        FILLED, REJECTED, or ALREADY_OPEN
    rejection_msg: Populated if status == REJECTED
    """
    order:           Order
    fill_price:      float
    broker_trade_id: Optional[str]
    filled_at:       datetime
    status:          FillStatus
    rejection_msg:   Optional[str] = None


@dataclass(frozen=True)
class SlippageRecord:
    """
    Slippage record logged for every fill attempt.

    Used for post-trade analysis and spread monitoring.

    pair:           FX pair
    side:           BUY or SELL
    expected_price: Mid price at signal time (from the last candle close)
    fill_price:     Actual broker fill price
    slippage_pips:  Sign-adjusted cost in pips. Convention: positive = worse for trader.
                    BUY:  (fill_price - mid) / pip_size  — positive when fill > mid (paid more)
                    SELL: (mid - fill_price) / pip_size  — positive when fill < mid (received less)
    signal_id:      Matches the Order.signal_id for join-key tracing
    recorded_at:    UTC timestamp
    """
    pair:           str
    side:           OrderSide
    expected_price: float
    fill_price:      float
    slippage_pips:  float
    signal_id:      str
    recorded_at:    datetime


@dataclass(frozen=True)
class SpreadSnapshot:
    """
    Current bid/ask snapshot from the broker at order submission time.

    Used to gate order submission: if live spread > max_spread_pips, skip.
    """
    pair:        str
    bid:         float
    ask:         float
    mid:         float
    spread_pips: float
    sampled_at:  datetime
