"""
Risk engine: evaluates a signal against account state and risk rules.

Responsibilities:
  1. Position sizing   — risk_per_trade % of balance / stop distance
  2. Position limits   — max_open, max_per_pair, max_usd_exposure
  3. Circuit breakers  — daily loss, weekly loss, max drawdown
  4. RiskDecision      — approve or reject with reason

The engine is stateless between calls EXCEPT for the circuit breaker state,
which is passed in by the caller and mutated in place. The caller is responsible
for persisting the CircuitBreakerState to SQLite after each evaluate() call.

Pip value assumption:
  The engine uses a simplified pip_value_per_lot table.
  For v1, EUR_USD / GBP_USD / AUD_USD / USD_CAD: $10 per pip per standard lot.
  For USD_JPY: $9.09 per pip per standard lot at ~110 (approximate, acceptable for sizing).
  A more accurate model requires live quote conversion — deferred to Phase 6.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from strategy.models import Direction, Signal
from .models import (
    AccountState,
    CircuitBreakerState,
    HaltReason,
    OpenPosition,
    RejectionReason,
    RiskConfig,
    RiskDecision,
)

# Pip sizes per pair (matching backtest/costs.py)
_PIP_SIZE: dict[str, float] = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,
    "AUD_USD": 0.0001,
    "USD_CAD": 0.0001,
}

# Standard lot size
_LOT_SIZE = 100_000


class RiskEngine:
    """
    Evaluate a Signal against current account state and return a RiskDecision.

    Usage:
        engine = RiskEngine(config)
        cb_state = CircuitBreakerState(current_peak=initial_balance)
        decision = engine.evaluate(signal, account, cb_state)
        # persist cb_state to SQLite

    The engine mutates cb_state in place when a circuit breaker is tripped
    or when the peak equity is updated.
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self._config = config or RiskConfig()

    @property
    def config(self) -> RiskConfig:
        return self._config

    def evaluate(
        self,
        signal:    Signal,
        account:   AccountState,
        cb_state:  CircuitBreakerState,
    ) -> RiskDecision:
        """
        Evaluate signal against risk rules.

        Mutates cb_state if:
          - A new circuit breaker trips
          - Peak equity is updated (no halt, just bookkeeping)

        Returns RiskDecision (approved or rejected with reason).
        """
        now = account.evaluation_time or datetime.now(tz=timezone.utc)

        # --- 0. Update peak equity (always, regardless of signal direction) ---
        if account.equity > cb_state.current_peak:
            cb_state.current_peak = account.equity

        # --- 1. Circuit breaker check ---
        cb_result = self._check_circuit_breakers(account, cb_state, now)
        if cb_result is not None:
            return cb_result

        # --- 2. FLAT signals bypass position limit checks ---
        if signal.direction == Direction.FLAT:
            return RiskDecision.approve(units=0, stop_loss=None, risk_amount=0.0)

        # --- 3. Position limit checks ---
        limit_result = self._check_position_limits(signal, account)
        if limit_result is not None:
            return limit_result

        # --- 4. Position sizing ---
        return self._size_position(signal, account)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_circuit_breakers(
        self,
        account:  AccountState,
        cb_state: CircuitBreakerState,
        now:      datetime,
    ) -> Optional[RiskDecision]:
        """
        Check all circuit breakers. If any trip, update cb_state and return rejection.
        If already halted, return rejection immediately (no double-trip logic needed).
        """
        cfg = self._config

        # Already halted — reject immediately
        if cb_state.is_halted:
            return RiskDecision.reject(RejectionReason.CIRCUIT_BREAKER_HALTED)

        # Daily loss check: equity < day_start_balance × (1 - daily_loss_halt_pct)
        daily_threshold = account.day_start_balance * (1 - cfg.daily_loss_halt_pct)
        if account.equity < daily_threshold:
            cb_state.trip(HaltReason.DAILY_LOSS, account.equity, now)
            return RiskDecision.reject(RejectionReason.CIRCUIT_BREAKER_HALTED)

        # Weekly loss check: equity < week_start_balance × (1 - weekly_loss_halt_pct)
        weekly_threshold = account.week_start_balance * (1 - cfg.weekly_loss_halt_pct)
        if account.equity < weekly_threshold:
            cb_state.trip(HaltReason.WEEKLY_LOSS, account.equity, now)
            return RiskDecision.reject(RejectionReason.CIRCUIT_BREAKER_HALTED)

        # Max drawdown check: equity < peak × (1 - max_drawdown_halt_pct)
        if cb_state.current_peak > 0:
            dd_threshold = cb_state.current_peak * (1 - cfg.max_drawdown_halt_pct)
            if account.equity < dd_threshold:
                cb_state.trip(HaltReason.MAX_DRAWDOWN, account.equity, now)
                return RiskDecision.reject(RejectionReason.CIRCUIT_BREAKER_HALTED)

        return None

    def _check_position_limits(
        self,
        signal:  Signal,
        account: AccountState,
    ) -> Optional[RiskDecision]:
        """
        Validate position limits. Returns rejection if any limit is breached.
        """
        cfg       = self._config
        positions = account.open_positions

        # Max total open positions
        if len(positions) >= cfg.max_open_positions:
            return RiskDecision.reject(RejectionReason.MAX_OPEN_POSITIONS)

        # Max positions per pair (same pair, any direction)
        pair_count = sum(1 for p in positions if p.pair == signal.pair)
        if pair_count >= cfg.max_positions_per_pair:
            return RiskDecision.reject(RejectionReason.MAX_POSITIONS_PER_PAIR)

        # USD exposure limit
        if "USD" in signal.pair:
            usd_count = sum(1 for p in positions if "USD" in p.pair)
            if usd_count >= cfg.max_usd_exposure:
                return RiskDecision.reject(RejectionReason.MAX_USD_EXPOSURE)

        return None

    def _size_position(
        self,
        signal:  Signal,
        account: AccountState,
    ) -> RiskDecision:
        """
        Calculate position size using fixed fractional risk.

        Formula:
            risk_amount = balance × risk_per_trade
            stop_distance_pips = |entry_price - stop_loss| / pip_size
            pip_value_per_lot  = pip_size × lot_size   (simplified, USD quote pairs)
            units = risk_amount / (stop_distance_pips × pip_value_per_lot / lot_size)
                  = risk_amount / (stop_distance_pips × pip_size)
                  = risk_amount / stop_distance_price
        """
        cfg = self._config

        pip_size = _PIP_SIZE.get(signal.pair)
        if pip_size is None:
            return RiskDecision.reject(RejectionReason.INVALID_STOP_DISTANCE)

        stop_distance_price = abs(signal.entry_price - signal.stop_loss)
        if stop_distance_price <= 0:
            return RiskDecision.reject(RejectionReason.INVALID_STOP_DISTANCE)

        stop_distance_pips = stop_distance_price / pip_size
        if stop_distance_pips <= 0:
            return RiskDecision.reject(RejectionReason.INVALID_STOP_DISTANCE)

        risk_amount = account.balance * cfg.risk_per_trade

        # units = risk_amount / (stop_distance_pips × pip_value_per_unit)
        # pip_value_per_unit = pip_size (for USD-quoted pairs)
        # so: units = risk_amount / stop_distance_price
        units = int(risk_amount / stop_distance_price)
        units = min(units, cfg.max_units_cap)

        if units <= 0:
            return RiskDecision.reject(RejectionReason.ZERO_UNITS)

        return RiskDecision.approve(
            units=units,
            stop_loss=signal.stop_loss,
            risk_amount=risk_amount,
        )
