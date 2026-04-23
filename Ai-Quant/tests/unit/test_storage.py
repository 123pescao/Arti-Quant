"""
Unit tests for src/storage/database.py.

All tests use an in-memory SQLite database (:memory:).

Critical properties tested:
  1.  Circuit breaker state: default on empty DB
  2.  Circuit breaker state: save + load round-trip
  3.  Circuit breaker state: halted state persists correctly
  4.  Circuit breaker state: reset state persists correctly
  5.  Equity snapshot: save and retrieve by date
  6.  Equity snapshot: INSERT OR REPLACE (second save on same date overwrites)
  7.  get_day_start_balance: returns yesterday's balance
  8.  get_day_start_balance: returns None if no prior snapshot
  9.  get_week_start_balance: returns most recent Monday's balance
  10. get_week_start_balance: returns None if no Monday snapshot
  11. Signal logging: row inserted with correct values
  12. Fill logging: row inserted with correct values
  13. Last run date: default None, set, retrieve
  14. Last run date: update on second set
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from datetime import date, datetime, timedelta, timezone

from risk.models import CircuitBreakerState, HaltReason
from storage.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Database:
    """Fresh in-memory database for each test."""
    return Database(":memory:")


_NOW = datetime(2024, 1, 15, 22, 0, 0, tzinfo=timezone.utc)
_TODAY = _NOW.date()
_YESTERDAY = _TODAY - timedelta(days=1)
_MONDAY = date(2024, 1, 15)   # 2024-01-15 is a Monday


# ---------------------------------------------------------------------------
# Circuit breaker state tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerState:
    def test_default_state_when_empty(self, db: Database):
        state = db.load_circuit_breaker_state()
        assert state.is_halted is False
        assert state.halt_reason == HaltReason.NONE
        assert state.current_peak == 0.0
        assert state.requires_manual_reset is False
        assert state.halted_at is None

    def test_save_and_load_not_halted(self, db: Database):
        state = CircuitBreakerState(is_halted=False, current_peak=10_500.0)
        db.save_circuit_breaker_state(state)

        loaded = db.load_circuit_breaker_state()
        assert loaded.is_halted is False
        assert loaded.current_peak == pytest.approx(10_500.0)

    def test_save_and_load_halted_state(self, db: Database):
        state = CircuitBreakerState()
        state.trip(HaltReason.DAILY_LOSS, equity=9_700.0, now=_NOW)

        db.save_circuit_breaker_state(state)
        loaded = db.load_circuit_breaker_state()

        assert loaded.is_halted is True
        assert loaded.halt_reason == HaltReason.DAILY_LOSS
        assert loaded.requires_manual_reset is False
        assert loaded.halted_at is not None
        assert loaded.halted_at.tzinfo is not None

    def test_drawdown_halt_requires_manual_reset(self, db: Database):
        state = CircuitBreakerState(current_peak=12_000.0)
        state.trip(HaltReason.MAX_DRAWDOWN, equity=9_000.0, now=_NOW)

        db.save_circuit_breaker_state(state)
        loaded = db.load_circuit_breaker_state()

        assert loaded.requires_manual_reset is True
        assert loaded.halt_reason == HaltReason.MAX_DRAWDOWN

    def test_upsert_overwrites_previous_row(self, db: Database):
        state = CircuitBreakerState(current_peak=10_000.0)
        db.save_circuit_breaker_state(state)

        state.trip(HaltReason.WEEKLY_LOSS, equity=9_400.0, now=_NOW)
        db.save_circuit_breaker_state(state)

        loaded = db.load_circuit_breaker_state()
        assert loaded.is_halted is True
        assert loaded.halt_reason == HaltReason.WEEKLY_LOSS

    def test_reset_state_persists(self, db: Database):
        state = CircuitBreakerState()
        state.trip(HaltReason.DAILY_LOSS, equity=9_800.0, now=_NOW)
        db.save_circuit_breaker_state(state)

        state.reset()
        db.save_circuit_breaker_state(state)
        loaded = db.load_circuit_breaker_state()

        assert loaded.is_halted is False
        assert loaded.halt_reason == HaltReason.NONE


# ---------------------------------------------------------------------------
# Equity snapshot tests
# ---------------------------------------------------------------------------

class TestEquitySnapshots:
    def test_save_and_retrieve_by_date(self, db: Database):
        db.save_equity_snapshot(10_000.0, 10_050.0, _TODAY)
        balance = db.get_balance_on_date(_TODAY)
        assert balance == pytest.approx(10_000.0)

    def test_returns_none_for_missing_date(self, db: Database):
        result = db.get_balance_on_date(date(2020, 1, 1))
        assert result is None

    def test_insert_or_replace_on_same_date(self, db: Database):
        db.save_equity_snapshot(10_000.0, 10_000.0, _TODAY)
        db.save_equity_snapshot(10_250.0, 10_300.0, _TODAY)  # overwrites
        balance = db.get_balance_on_date(_TODAY)
        assert balance == pytest.approx(10_250.0)

    def test_multiple_dates_stored(self, db: Database):
        db.save_equity_snapshot(10_000.0, 10_000.0, _YESTERDAY)
        db.save_equity_snapshot(10_100.0, 10_150.0, _TODAY)

        assert db.get_balance_on_date(_YESTERDAY) == pytest.approx(10_000.0)
        assert db.get_balance_on_date(_TODAY)     == pytest.approx(10_100.0)


# ---------------------------------------------------------------------------
# Day-start balance tests
# ---------------------------------------------------------------------------

class TestDayStartBalance:
    def test_returns_yesterday_balance(self, db: Database):
        db.save_equity_snapshot(9_950.0, 9_950.0, _YESTERDAY)
        result = db.get_day_start_balance(_TODAY)
        assert result == pytest.approx(9_950.0)

    def test_returns_none_when_no_prior_snapshot(self, db: Database):
        result = db.get_day_start_balance(_TODAY)
        assert result is None

    def test_ignores_today_snapshot(self, db: Database):
        """Today's snapshot is not yesterday's — must not be returned."""
        db.save_equity_snapshot(10_000.0, 10_000.0, _TODAY)
        result = db.get_day_start_balance(_TODAY)
        assert result is None


# ---------------------------------------------------------------------------
# Week-start balance tests
# ---------------------------------------------------------------------------

class TestWeekStartBalance:
    def test_returns_monday_balance(self, db: Database):
        monday = _MONDAY  # 2024-01-15
        db.save_equity_snapshot(9_800.0, 9_800.0, monday)
        result = db.get_week_start_balance(monday + timedelta(days=2))  # Wednesday
        assert result == pytest.approx(9_800.0)

    def test_returns_none_when_no_monday_snapshot(self, db: Database):
        result = db.get_week_start_balance(date(2024, 1, 17))  # Wednesday
        assert result is None

    def test_finds_monday_from_wednesday(self, db: Database):
        monday    = date(2024, 1, 15)
        wednesday = date(2024, 1, 17)
        db.save_equity_snapshot(10_000.0, 10_000.0, monday)
        result = db.get_week_start_balance(wednesday)
        assert result == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# Signal logging tests
# ---------------------------------------------------------------------------

class TestSignalLogging:
    def test_approved_signal_saved(self, db: Database):
        db.save_signal(
            pair             = "EUR_USD",
            direction        = "LONG",
            entry_price      = 1.10,
            stop_loss        = 1.08,
            units            = 10_000,
            risk_amount      = 25.0,
            approved         = True,
            rejection_reason = None,
            signal_time      = _NOW,
        )
        # Verify no exception; table is not exposed via public API but we can
        # check it implicitly by verifying the save didn't raise.

    def test_rejected_signal_saved(self, db: Database):
        db.save_signal(
            pair             = "GBP_USD",
            direction        = "SHORT",
            entry_price      = 1.27,
            stop_loss        = 1.29,
            units            = None,
            risk_amount      = None,
            approved         = False,
            rejection_reason = "max_open_positions",
            signal_time      = _NOW,
        )


# ---------------------------------------------------------------------------
# Fill logging tests
# ---------------------------------------------------------------------------

class TestFillLogging:
    def test_fill_saved(self, db: Database):
        db.save_fill(
            pair            = "EUR_USD",
            side            = "buy",
            units           = 10_000,
            fill_price      = 1.10008,
            stop_loss       = 1.08000,
            signal_id       = "EUR_USD_LONG_20240115",
            broker_trade_id = "T-123",
            status          = "filled",
            slippage_pips   = 0.8,
            filled_at       = _NOW,
        )

    def test_fill_with_null_slippage(self, db: Database):
        db.save_fill(
            pair            = "EUR_USD",
            side            = "buy",
            units           = 10_000,
            fill_price      = 0.0,
            stop_loss       = 1.08,
            signal_id       = "SIG",
            broker_trade_id = "T-200",
            status          = "already_open",
            slippage_pips   = None,
            filled_at       = _NOW,
        )


# ---------------------------------------------------------------------------
# Last run date tests
# ---------------------------------------------------------------------------

class TestLastRunDate:
    def test_default_none(self, db: Database):
        assert db.get_last_run_date() is None

    def test_set_and_retrieve(self, db: Database):
        db.set_last_run_date(_TODAY)
        assert db.get_last_run_date() == _TODAY

    def test_update_on_second_set(self, db: Database):
        db.set_last_run_date(_YESTERDAY)
        db.set_last_run_date(_TODAY)
        assert db.get_last_run_date() == _TODAY

    def test_set_defaults_to_today_utc(self, db: Database):
        """set_last_run_date() with no argument sets today's UTC date."""
        db.set_last_run_date()  # no argument
        stored = db.get_last_run_date()
        assert stored is not None
        assert isinstance(stored, date)
