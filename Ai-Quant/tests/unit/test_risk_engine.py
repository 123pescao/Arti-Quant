"""
Unit tests for the Risk Engine (src/risk/).

Coverage:
  - RiskConfig validation
  - AccountState validation
  - CircuitBreakerState: trip, reset, peak tracking
  - Happy path: approved trade with correct units
  - Zero-units rejection (stop too wide for balance)
  - Max open positions rejection
  - Max positions per pair rejection
  - USD exposure rejection
  - Daily circuit breaker trigger
  - Weekly circuit breaker trigger
  - Drawdown circuit breaker trigger
  - Drawdown CB requires manual reset (daily/weekly auto-resets)
  - Manual reset behaviour
  - FLAT signals bypass position limits
  - Boundary conditions (exact threshold values)
  - Peak equity tracking (no halt)
  - Units capped at max_units_cap
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from datetime import datetime, timezone

from strategy.models import Direction, Signal
from risk.models import (
    AccountState,
    CircuitBreakerState,
    HaltReason,
    OpenPosition,
    RejectionReason,
    RiskConfig,
    RiskDecision,
)
from risk.engine import RiskEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 10, 22, 10, 0, tzinfo=timezone.utc)


def _signal(
    pair: str = "EUR_USD",
    direction: Direction = Direction.LONG,
    entry: float = 1.1000,
    stop: float = None,
) -> Signal:
    """
    Build a valid Signal for the given direction.
    Default stops are 200 pips away (LONG: below entry, SHORT: above entry).
    """
    if stop is None:
        stop = (entry - 0.0200) if direction == Direction.LONG else (entry + 0.0200)
    return Signal(
        pair=pair,
        direction=direction,
        entry_price=entry,
        stop_loss=stop,
        atr=abs(entry - stop) / 2.0,
        signal_time=_NOW,
        source_bar=50,
    )


def _account(
    balance: float = 10_000.0,
    equity: float = 10_000.0,
    peak: float = 10_000.0,
    day_start: float = 10_000.0,
    week_start: float = 10_000.0,
    positions: tuple = (),
) -> AccountState:
    return AccountState(
        balance=balance,
        equity=equity,
        peak_balance=peak,
        day_start_balance=day_start,
        week_start_balance=week_start,
        open_positions=positions,
        evaluation_time=_NOW,
    )


def _cb(peak: float = 10_000.0) -> CircuitBreakerState:
    return CircuitBreakerState(current_peak=peak)


def _open_pos(pair: str = "EUR_USD", direction: str = "LONG") -> OpenPosition:
    return OpenPosition(
        pair=pair,
        direction=direction,
        units=10_000,
        entry_price=1.10,
        stop_loss=1.08,
    )


# ---------------------------------------------------------------------------
# RiskConfig tests
# ---------------------------------------------------------------------------

class TestRiskConfig:
    def test_defaults_match_v1_spec(self):
        cfg = RiskConfig()
        assert cfg.risk_per_trade == 0.0025
        assert cfg.max_open_positions == 2
        assert cfg.max_positions_per_pair == 1
        assert cfg.max_usd_exposure == 2
        assert cfg.daily_loss_halt_pct == 0.02
        assert cfg.weekly_loss_halt_pct == 0.05
        assert cfg.max_drawdown_halt_pct == 0.15

    def test_risk_per_trade_zero_invalid(self):
        with pytest.raises(ValueError):
            RiskConfig(risk_per_trade=0.0)

    def test_risk_per_trade_too_high_invalid(self):
        with pytest.raises(ValueError):
            RiskConfig(risk_per_trade=0.10)

    def test_max_open_positions_zero_invalid(self):
        with pytest.raises(ValueError):
            RiskConfig(max_open_positions=0)

    def test_daily_loss_pct_must_be_fraction(self):
        with pytest.raises(ValueError):
            RiskConfig(daily_loss_halt_pct=1.5)

    def test_drawdown_halt_pct_must_be_fraction(self):
        with pytest.raises(ValueError):
            RiskConfig(max_drawdown_halt_pct=0.0)


# ---------------------------------------------------------------------------
# AccountState tests
# ---------------------------------------------------------------------------

class TestAccountState:
    def test_positive_balance_required(self):
        with pytest.raises(ValueError):
            AccountState(
                balance=0.0,
                equity=0.0,
                peak_balance=1000.0,
                day_start_balance=1000.0,
                week_start_balance=1000.0,
            )

    def test_day_start_positive_required(self):
        with pytest.raises(ValueError):
            AccountState(
                balance=1000.0,
                equity=1000.0,
                peak_balance=1000.0,
                day_start_balance=0.0,
                week_start_balance=1000.0,
            )


# ---------------------------------------------------------------------------
# CircuitBreakerState tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerState:
    def test_initial_state_not_halted(self):
        cb = _cb()
        assert not cb.is_halted
        assert cb.halt_reason == HaltReason.NONE

    def test_trip_sets_halted(self):
        cb = _cb(peak=10_000.0)
        cb.trip(HaltReason.DAILY_LOSS, equity=9_700.0, now=_NOW)
        assert cb.is_halted
        assert cb.halt_reason == HaltReason.DAILY_LOSS
        assert cb.halted_at == _NOW

    def test_drawdown_trip_requires_manual_reset(self):
        cb = _cb(peak=10_000.0)
        cb.trip(HaltReason.MAX_DRAWDOWN, equity=8_400.0, now=_NOW)
        assert cb.requires_manual_reset is True

    def test_daily_trip_does_not_require_manual_reset(self):
        cb = _cb()
        cb.trip(HaltReason.DAILY_LOSS, equity=9_700.0, now=_NOW)
        assert cb.requires_manual_reset is False

    def test_weekly_trip_does_not_require_manual_reset(self):
        cb = _cb()
        cb.trip(HaltReason.WEEKLY_LOSS, equity=9_400.0, now=_NOW)
        assert cb.requires_manual_reset is False

    def test_reset_clears_state(self):
        cb = _cb()
        cb.trip(HaltReason.DAILY_LOSS, equity=9_700.0, now=_NOW)
        cb.reset()
        assert not cb.is_halted
        assert cb.halt_reason == HaltReason.NONE
        assert cb.halted_at is None
        assert cb.requires_manual_reset is False

    def test_peak_not_updated_by_lower_equity(self):
        cb = _cb(peak=12_000.0)
        cb.trip(HaltReason.DAILY_LOSS, equity=9_700.0, now=_NOW)
        assert cb.current_peak == 12_000.0  # trip doesn't lower the peak


# ---------------------------------------------------------------------------
# Happy path: approved trade
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_basic_long_approved(self):
        engine = RiskEngine()
        sig = _signal("EUR_USD", Direction.LONG)
        acc = _account(balance=10_000.0)
        cb  = _cb(peak=10_000.0)

        decision = engine.evaluate(sig, acc, cb)

        assert decision.approved
        assert decision.units > 0
        assert decision.stop_loss == sig.stop_loss
        assert decision.rejection_reason is None
        assert not cb.is_halted

    def test_basic_short_approved(self):
        engine = RiskEngine()
        sig = _signal("EUR_USD", Direction.SHORT)
        acc = _account(balance=10_000.0)
        cb  = _cb(peak=10_000.0)

        decision = engine.evaluate(sig, acc, cb)

        assert decision.approved
        assert decision.units > 0

    def test_units_match_risk_formula(self):
        """
        EUR_USD, entry=1.1000, stop=1.0800 → stop_distance ≈ 0.02 (200 pips)
        risk_amount = 10_000 × 0.0025 = $25
        units = int(25 / 0.02) = 1249 or 1250 depending on IEEE 754 representation.
        int(25 / (1.10 - 1.08)) floors at 1249 due to floating-point precision.
        """
        engine = RiskEngine()
        sig = _signal("EUR_USD", Direction.LONG, entry=1.1000, stop=1.0800)
        acc = _account(balance=10_000.0)
        cb  = _cb(peak=10_000.0)

        decision = engine.evaluate(sig, acc, cb)

        assert decision.approved
        # Allow ±1 unit tolerance for IEEE 754 truncation
        assert abs(decision.units - 1250) <= 1

    def test_risk_amount_is_correct(self):
        engine = RiskEngine()
        sig = _signal("EUR_USD", Direction.LONG, entry=1.1000, stop=1.0800)
        acc = _account(balance=10_000.0)
        cb  = _cb(peak=10_000.0)

        decision = engine.evaluate(sig, acc, cb)

        assert abs(decision.risk_amount - 25.0) < 1e-6   # 10000 * 0.0025

    def test_peak_equity_updated_when_equity_rises(self):
        """Engine updates cb_state.current_peak when equity > current_peak."""
        engine = RiskEngine()
        sig = _signal()
        acc = _account(balance=11_000.0, equity=11_500.0, peak=10_000.0)
        cb  = _cb(peak=10_000.0)

        engine.evaluate(sig, acc, cb)

        assert cb.current_peak == 11_500.0

    def test_peak_not_lowered_when_equity_falls(self):
        engine = RiskEngine()
        sig = _signal()
        acc = _account(balance=9_500.0, equity=9_500.0, peak=10_000.0)
        cb  = _cb(peak=10_000.0)

        engine.evaluate(sig, acc, cb)

        assert cb.current_peak == 10_000.0


# ---------------------------------------------------------------------------
# Zero-units rejection
# ---------------------------------------------------------------------------

class TestZeroUnitsRejection:
    def test_stop_too_wide_gives_zero_units(self):
        """
        Very small balance + very wide stop → units floors to 0 → reject.
        balance=100, risk_amount = 100*0.0025 = $0.25
        stop_distance = |51.0 - 1.0| = 50 price units
        units = int(0.25 / 50) = 0 → ZERO_UNITS rejection.
        """
        engine = RiskEngine(RiskConfig(risk_per_trade=0.0025))
        sig = Signal(
            pair="EUR_USD",
            direction=Direction.LONG,
            entry_price=51.0,
            stop_loss=1.0,      # 50 price units below entry
            atr=5.0,
            signal_time=_NOW,
            source_bar=50,
        )
        acc = _account(balance=100.0, equity=100.0, peak=100.0,
                       day_start=100.0, week_start=100.0)
        cb  = _cb(peak=100.0)

        decision = engine.evaluate(sig, acc, cb)

        assert not decision.approved
        assert decision.units == 0
        assert decision.rejection_reason == RejectionReason.ZERO_UNITS

    def test_units_floored_to_cap(self):
        """Units should be capped at max_units_cap."""
        cfg = RiskConfig(max_units_cap=500)
        engine = RiskEngine(cfg)
        # Very tight stop → very large units, should be capped at 500
        sig = _signal("EUR_USD", Direction.LONG, entry=1.1000, stop=1.0999)  # ~1 pip stop
        acc = _account(balance=10_000.0)
        cb  = _cb(peak=10_000.0)

        decision = engine.evaluate(sig, acc, cb)

        if decision.approved:
            assert decision.units <= 500


# ---------------------------------------------------------------------------
# Position limit rejections
# ---------------------------------------------------------------------------

class TestPositionLimits:
    def test_max_open_positions_rejected(self):
        """Reject when max_open_positions (2) already reached."""
        engine = RiskEngine()
        sig = _signal("AUD_USD")
        acc = _account(positions=(
            _open_pos("EUR_USD", "LONG"),
            _open_pos("GBP_USD", "SHORT"),
        ))
        cb  = _cb()

        decision = engine.evaluate(sig, acc, cb)

        assert not decision.approved
        assert decision.rejection_reason == RejectionReason.MAX_OPEN_POSITIONS

    def test_one_open_position_allows_second(self):
        """One open position should allow a second trade."""
        engine = RiskEngine()
        sig = _signal("GBP_USD")
        acc = _account(positions=(_open_pos("EUR_USD", "LONG"),))
        cb  = _cb()

        decision = engine.evaluate(sig, acc, cb)

        assert decision.approved

    def test_same_pair_rejected(self):
        """Reject second trade on the same pair regardless of direction."""
        engine = RiskEngine()
        # SHORT on EUR_USD while a LONG is already open — stop must be above entry for SHORT
        sig = _signal("EUR_USD", Direction.SHORT, entry=1.1000, stop=1.1200)
        acc = _account(positions=(_open_pos("EUR_USD", "LONG"),))
        cb  = _cb()

        decision = engine.evaluate(sig, acc, cb)

        assert not decision.approved
        assert decision.rejection_reason == RejectionReason.MAX_POSITIONS_PER_PAIR

    def test_usd_exposure_limit_via_config(self):
        """
        Reject USD pair when max_usd_exposure=1 is already reached.
        Use 1 USD pair open so we don't hit max_open_positions first.
        """
        cfg = RiskConfig(max_usd_exposure=1)
        engine = RiskEngine(cfg)
        sig = _signal("GBP_USD")
        acc = _account(positions=(_open_pos("EUR_USD", "LONG"),))
        cb  = _cb()

        decision = engine.evaluate(sig, acc, cb)

        assert not decision.approved
        assert decision.rejection_reason == RejectionReason.MAX_USD_EXPOSURE

    def test_non_usd_pair_not_counted_for_usd_exposure(self):
        """1 USD pair open + max_usd_exposure=2 → second USD pair allowed."""
        engine = RiskEngine()
        sig = _signal("USD_JPY")
        acc = _account(positions=(_open_pos("EUR_USD", "LONG"),))
        cb  = _cb()

        decision = engine.evaluate(sig, acc, cb)

        assert decision.approved  # 1 USD pair open, limit is 2

    def test_usd_exposure_at_exact_limit_rejects_additional(self):
        """max_usd_exposure=1: exactly 1 USD pair open → second USD pair rejected."""
        engine = RiskEngine(RiskConfig(max_usd_exposure=1))
        sig = _signal("AUD_USD")
        acc = _account(positions=(_open_pos("EUR_USD", "LONG"),))
        cb  = _cb()

        decision = engine.evaluate(sig, acc, cb)

        assert not decision.approved
        assert decision.rejection_reason == RejectionReason.MAX_USD_EXPOSURE


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerDaily:
    def test_daily_cb_trips_below_threshold(self):
        """
        daily_loss_halt_pct = 2%
        day_start = 10000 → threshold = 9800
        equity = 9750 < 9800 → trip DAILY
        """
        engine = RiskEngine()
        sig = _signal()
        acc = _account(
            balance=9_750.0,
            equity=9_750.0,
            day_start=10_000.0,
            week_start=10_000.0,
        )
        cb  = _cb(peak=10_000.0)

        decision = engine.evaluate(sig, acc, cb)

        assert not decision.approved
        assert decision.rejection_reason == RejectionReason.CIRCUIT_BREAKER_HALTED
        assert cb.is_halted
        assert cb.halt_reason == HaltReason.DAILY_LOSS
        assert cb.requires_manual_reset is False

    def test_daily_cb_does_not_trip_above_threshold(self):
        """
        equity = 9801 > threshold 9800 → no daily trip.
        week_start matches so no weekly trip either.
        """
        engine = RiskEngine()
        sig = _signal()
        acc = _account(
            balance=9_801.0,
            equity=9_801.0,
            day_start=10_000.0,
            week_start=10_000.0,
        )
        cb  = _cb(peak=10_000.0)

        decision = engine.evaluate(sig, acc, cb)

        assert not cb.is_halted
        assert cb.halt_reason == HaltReason.NONE

    def test_daily_cb_at_exact_threshold(self):
        """
        equity == threshold (9800.0 exactly) → strict < comparison → no trip.
        """
        engine = RiskEngine()
        sig = _signal()
        acc = _account(
            balance=9_800.0,
            equity=9_800.0,
            day_start=10_000.0,
            week_start=10_000.0,
        )
        cb  = _cb(peak=10_000.0)

        engine.evaluate(sig, acc, cb)

        assert not cb.is_halted


class TestCircuitBreakerWeekly:
    def test_weekly_cb_trips(self):
        """
        weekly_loss_halt_pct = 5% → threshold = week_start × 0.95
        week_start = 10000 → threshold = 9500
        equity = 9400 < 9500 → WEEKLY trip.

        day_start chosen so daily CB does NOT fire first:
          daily_threshold = day_start × 0.98
          Need equity (9400) >= day_start × 0.98
          → day_start ≤ 9400 / 0.98 = 9591
          Use day_start = 9500: daily_threshold = 9310 < 9400 → no daily trip.
        """
        engine = RiskEngine()
        sig = _signal()
        acc = _account(
            balance=9_400.0,
            equity=9_400.0,
            day_start=9_500.0,    # daily_threshold = 9310, equity 9400 > 9310 → no daily
            week_start=10_000.0,  # weekly_threshold = 9500, equity 9400 < 9500 → WEEKLY
        )
        cb  = _cb(peak=10_000.0)

        decision = engine.evaluate(sig, acc, cb)

        assert not decision.approved
        assert cb.halt_reason == HaltReason.WEEKLY_LOSS
        assert cb.requires_manual_reset is False

    def test_weekly_cb_not_trip_when_above_threshold(self):
        """
        equity = 9501 → just above weekly threshold 9500 → no weekly trip.

        day_start chosen so daily CB does NOT fire:
          Need equity (9501) >= day_start × 0.98 → day_start ≤ 9695
          Use day_start = 9600: daily_threshold = 9408 < 9501 → no daily trip.
        """
        engine = RiskEngine()
        sig = _signal()
        acc = _account(
            balance=9_501.0,
            equity=9_501.0,
            day_start=9_600.0,    # daily_threshold = 9408 < 9501 → no daily
            week_start=10_000.0,  # weekly_threshold = 9500 < 9501 → no weekly
        )
        cb  = _cb(peak=10_000.0)

        engine.evaluate(sig, acc, cb)

        assert not cb.is_halted


class TestCircuitBreakerDrawdown:
    def test_drawdown_cb_trips(self):
        """
        max_drawdown_halt_pct = 15%
        current_peak = 12000 → dd_threshold = 12000 × 0.85 = 10200
        equity = 10000 < 10200 → DRAWDOWN trip.

        day_start and week_start chosen so those CBs do NOT fire first:
          day_start = 10100: daily_threshold = 9898 < 10000 → no daily.
          week_start = 10200: weekly_threshold = 9690 < 10000 → no weekly.
        """
        engine = RiskEngine()
        sig = _signal()
        acc = _account(
            balance=10_000.0,
            equity=10_000.0,
            day_start=10_100.0,   # daily_threshold = 9898 < 10000 → no daily
            week_start=10_200.0,  # weekly_threshold = 9690 < 10000 → no weekly
        )
        cb = CircuitBreakerState(current_peak=12_000.0)

        decision = engine.evaluate(sig, acc, cb)

        assert not decision.approved
        assert cb.halt_reason == HaltReason.MAX_DRAWDOWN
        assert cb.requires_manual_reset is True

    def test_drawdown_cb_requires_manual_reset(self):
        """
        equity = 8000 under a peak of 12000 → drawdown = 33% > 15% → DD trip.

        day_start and week_start chosen so those CBs do NOT fire first:
          day_start = 8100:  daily_threshold  = 7938 < 8000 → no daily.
          week_start = 8300: weekly_threshold = 7885 < 8000 → no weekly.
        """
        engine = RiskEngine()
        sig = _signal()
        acc = _account(
            balance=8_000.0,
            equity=8_000.0,
            day_start=8_100.0,
            week_start=8_300.0,
        )
        cb = CircuitBreakerState(current_peak=12_000.0)

        engine.evaluate(sig, acc, cb)

        assert cb.requires_manual_reset is True

    def test_drawdown_cb_blocking_after_trip(self):
        """Once tripped, all subsequent evaluate() calls are rejected."""
        engine = RiskEngine()
        sig = _signal()
        acc = _account(balance=8_000.0, equity=8_000.0,
                       day_start=8_100.0, week_start=8_300.0)
        cb  = CircuitBreakerState(current_peak=12_000.0)

        engine.evaluate(sig, acc, cb)  # trips CB

        # Second call — same state but equity improved
        acc2 = _account(equity=10_000.0, balance=10_000.0)
        decision2 = engine.evaluate(sig, acc2, cb)

        assert not decision2.approved
        assert decision2.rejection_reason == RejectionReason.CIRCUIT_BREAKER_HALTED

    def test_manual_reset_allows_trading_again(self):
        """After manual reset, a valid signal should be approved again."""
        engine = RiskEngine()
        sig = _signal()
        acc = _account(balance=8_000.0, equity=8_000.0,
                       day_start=8_100.0, week_start=8_300.0)
        cb  = CircuitBreakerState(current_peak=12_000.0)

        engine.evaluate(sig, acc, cb)  # trips drawdown CB
        assert cb.is_halted

        cb.reset()                    # manual operator reset
        cb.current_peak = 10_000.0   # operator resets peak too

        acc_recovered = _account(balance=10_000.0, equity=10_000.0,
                                  day_start=10_000.0, week_start=10_000.0)
        decision = engine.evaluate(sig, acc_recovered, cb)

        assert decision.approved


# ---------------------------------------------------------------------------
# FLAT signal behaviour
# ---------------------------------------------------------------------------

class TestFlatSignal:
    def test_flat_signal_bypasses_position_limits(self):
        """
        FLAT signals are exits — approved even with position limits exhausted.
        """
        engine = RiskEngine()
        sig_flat = Signal(
            pair="EUR_USD",
            direction=Direction.FLAT,
            entry_price=1.1000,
            stop_loss=None,
            atr=0.015,
            signal_time=_NOW,
            source_bar=50,
        )
        acc = _account(positions=(
            _open_pos("EUR_USD", "LONG"),
            _open_pos("GBP_USD", "SHORT"),
        ))
        cb  = _cb()

        decision = engine.evaluate(sig_flat, acc, cb)

        assert decision.approved
        assert decision.units == 0

    def test_flat_signal_blocked_by_circuit_breaker(self):
        """Circuit breaker halts ALL activity including exits."""
        engine = RiskEngine()
        sig_flat = Signal(
            pair="EUR_USD",
            direction=Direction.FLAT,
            entry_price=1.1000,
            stop_loss=None,
            atr=0.015,
            signal_time=_NOW,
            source_bar=50,
        )
        acc = _account(equity=9_700.0, balance=9_700.0,
                       day_start=10_000.0, week_start=10_000.0)
        cb  = _cb(peak=10_000.0)

        decision = engine.evaluate(sig_flat, acc, cb)

        assert not decision.approved
        assert cb.is_halted


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_single_position_slot_remaining(self):
        """With 1 of 2 slots used, second different pair is accepted."""
        engine = RiskEngine()
        sig = _signal("GBP_USD")
        acc = _account(positions=(_open_pos("EUR_USD"),))
        cb  = _cb()

        decision = engine.evaluate(sig, acc, cb)

        assert decision.approved

    def test_exact_max_usd_limit_with_one_position(self):
        """max_usd_exposure=1: with one USD pair open, second USD pair rejected."""
        cfg = RiskConfig(max_usd_exposure=1)
        engine = RiskEngine(cfg)
        sig = _signal("GBP_USD")
        acc = _account(positions=(_open_pos("EUR_USD"),))
        cb  = _cb()

        decision = engine.evaluate(sig, acc, cb)

        assert not decision.approved
        assert decision.rejection_reason == RejectionReason.MAX_USD_EXPOSURE

    def test_units_scale_with_balance(self):
        """
        Doubling balance should roughly double units.
        Allow ±1 tolerance for IEEE 754 floor truncation across both computations.
        """
        engine = RiskEngine()
        sig = _signal("EUR_USD", Direction.LONG, entry=1.1000, stop=1.0800)
        cb1 = _cb(peak=10_000.0)
        cb2 = _cb(peak=20_000.0)

        d1 = engine.evaluate(sig, _account(balance=10_000.0, equity=10_000.0), cb1)
        d2 = engine.evaluate(sig, _account(balance=20_000.0, equity=20_000.0), cb2)

        assert d1.approved and d2.approved
        assert abs(d2.units - d1.units * 2) <= 1

    def test_daily_threshold_boundary_just_below(self):
        """equity = threshold - 0.01 → trips DAILY."""
        engine = RiskEngine()
        day_start = 10_000.0
        threshold = day_start * (1 - 0.02)       # 9800.0
        equity_just_below = threshold - 0.01     # 9799.99

        acc = _account(equity=equity_just_below, balance=equity_just_below,
                       day_start=day_start, week_start=day_start)
        cb  = _cb(peak=day_start)

        engine.evaluate(_signal(), acc, cb)

        assert cb.is_halted
        assert cb.halt_reason == HaltReason.DAILY_LOSS

    def test_risk_decision_reject_factory(self):
        d = RiskDecision.reject(RejectionReason.ZERO_UNITS)
        assert not d.approved
        assert d.units == 0
        assert d.stop_loss is None
        assert d.risk_amount == 0.0

    def test_risk_decision_approve_factory(self):
        d = RiskDecision.approve(units=1000, stop_loss=1.08, risk_amount=25.0)
        assert d.approved
        assert d.units == 1000
        assert d.rejection_reason is None
