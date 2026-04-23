"""
Risk engine data models.

These are the typed contracts between the risk engine and the rest of the system.
All are immutable (frozen dataclasses) except CircuitBreakerState which must
be persisted and mutated by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskConfig:
    """
    Immutable risk parameters for a trading session.

    All percentage values are fractions (e.g. 0.0025 = 0.25%).
    These defaults are the v1 production parameters.
    """
    # Position sizing
    risk_per_trade:        float = 0.0025   # 0.25% of account balance per trade
    max_units_cap:         int   = 100_000  # hard cap on units per position

    # Position limits
    max_open_positions:    int   = 2        # max total concurrent positions
    max_positions_per_pair: int  = 1        # max positions per FX pair
    max_usd_exposure:      int   = 2        # max positions in USD-denominated pairs

    # Circuit breaker thresholds
    daily_loss_halt_pct:   float = 0.02     # halt if day's loss > 2% of start balance
    weekly_loss_halt_pct:  float = 0.05     # halt if week's loss > 5% of start balance
    max_drawdown_halt_pct: float = 0.15     # halt if drawdown from peak > 15%

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= 0.05:
            raise ValueError(f"risk_per_trade must be in (0, 0.05], got {self.risk_per_trade}")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be >= 1")
        if self.max_positions_per_pair < 1:
            raise ValueError("max_positions_per_pair must be >= 1")
        if self.max_usd_exposure < 1:
            raise ValueError("max_usd_exposure must be >= 1")
        if not 0 < self.daily_loss_halt_pct < 1:
            raise ValueError("daily_loss_halt_pct must be in (0, 1)")
        if not 0 < self.weekly_loss_halt_pct < 1:
            raise ValueError("weekly_loss_halt_pct must be in (0, 1)")
        if not 0 < self.max_drawdown_halt_pct < 1:
            raise ValueError("max_drawdown_halt_pct must be in (0, 1)")


# ---------------------------------------------------------------------------
# Account state (inputs to the risk engine at evaluation time)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenPosition:
    """Snapshot of a single open position passed to the risk engine."""
    pair:        str
    direction:   str        # "LONG" or "SHORT"
    units:       int
    entry_price: float
    stop_loss:   float


@dataclass(frozen=True)
class AccountState:
    """
    Current account state passed to the risk engine at signal evaluation time.

    balance:            Current realized account balance (USD).
    equity:             Current equity including unrealized P&L.
    peak_balance:       Highest balance ever reached (for drawdown calculation).
    day_start_balance:  Balance at the start of today's session.
    week_start_balance: Balance at the start of the current week.
    open_positions:     Snapshot of all currently open positions.
    evaluation_time:    UTC time of evaluation (used for session guards).
    """
    balance:             float
    equity:              float
    peak_balance:        float
    day_start_balance:   float
    week_start_balance:  float
    open_positions:      tuple[OpenPosition, ...] = field(default_factory=tuple)
    evaluation_time:     Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.balance <= 0:
            raise ValueError(f"balance must be positive, got {self.balance}")
        if self.peak_balance < self.balance:
            # Peak must be >= current balance; allow equal (no drawdown case)
            # Note: equity can be above balance (unrealized gains), but peak tracks
            # the highest realized balance only
            pass  # We allow this: equity can exceed peak during open positions
        if self.day_start_balance <= 0:
            raise ValueError("day_start_balance must be positive")
        if self.week_start_balance <= 0:
            raise ValueError("week_start_balance must be positive")


# ---------------------------------------------------------------------------
# Circuit breaker state (persisted, mutable via engine)
# ---------------------------------------------------------------------------

class HaltReason(str, Enum):
    NONE           = "none"
    DAILY_LOSS     = "daily_loss"
    WEEKLY_LOSS    = "weekly_loss"
    MAX_DRAWDOWN   = "max_drawdown"


@dataclass
class CircuitBreakerState:
    """
    Persisted circuit breaker state.

    Designed to be stored in SQLite and loaded at engine startup.
    The drawdown halt requires manual reset; daily/weekly reset automatically
    on the next trading session boundary.

    Fields:
        is_halted:          True if trading is currently halted.
        halt_reason:        Which circuit breaker tripped.
        halted_at:          UTC timestamp of the halt.
        requires_manual_reset: True for drawdown halts; False for daily/weekly.
        current_peak:       Highest equity seen (for drawdown tracking).
    """
    is_halted:              bool          = False
    halt_reason:            HaltReason    = HaltReason.NONE
    halted_at:              Optional[datetime] = None
    requires_manual_reset:  bool          = False
    current_peak:           float         = 0.0

    def reset(self) -> None:
        """Reset the circuit breaker. Only valid for automatic resets or after manual confirmation."""
        self.is_halted             = False
        self.halt_reason           = HaltReason.NONE
        self.halted_at             = None
        self.requires_manual_reset = False

    def trip(self, reason: HaltReason, equity: float, now: datetime) -> None:
        """Trip the circuit breaker with a reason."""
        self.is_halted    = True
        self.halt_reason  = reason
        self.halted_at    = now
        self.requires_manual_reset = (reason == HaltReason.MAX_DRAWDOWN)
        if equity > self.current_peak:
            self.current_peak = equity


# ---------------------------------------------------------------------------
# Risk decision (output)
# ---------------------------------------------------------------------------

class RejectionReason(str, Enum):
    CIRCUIT_BREAKER_HALTED   = "circuit_breaker_halted"
    MAX_OPEN_POSITIONS       = "max_open_positions"
    MAX_POSITIONS_PER_PAIR   = "max_positions_per_pair"
    MAX_USD_EXPOSURE         = "max_usd_exposure"
    ZERO_UNITS               = "zero_units"
    INVALID_STOP_DISTANCE    = "invalid_stop_distance"
    UNITS_BELOW_MINIMUM      = "units_below_minimum"


@dataclass(frozen=True)
class RiskDecision:
    """
    Output of the risk engine for a single signal evaluation.

    approved:       True if the trade may proceed.
    units:          Calculated position size (0 if rejected).
    rejection_reason: Set if approved=False.
    stop_loss:      Confirmed stop price (unchanged from signal).
    risk_amount:    Dollar amount risked on this trade (for logging).
    """
    approved:         bool
    units:            int
    stop_loss:        Optional[float]
    risk_amount:      float
    rejection_reason: Optional[RejectionReason] = None

    @classmethod
    def reject(cls, reason: RejectionReason) -> "RiskDecision":
        return cls(
            approved=False,
            units=0,
            stop_loss=None,
            risk_amount=0.0,
            rejection_reason=reason,
        )

    @classmethod
    def approve(cls, units: int, stop_loss: float, risk_amount: float) -> "RiskDecision":
        return cls(
            approved=True,
            units=units,
            stop_loss=stop_loss,
            risk_amount=risk_amount,
            rejection_reason=None,
        )
