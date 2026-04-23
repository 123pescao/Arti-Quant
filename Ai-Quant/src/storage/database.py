"""
SQLite persistence layer.

Single Database class manages all persistent state for the live system.
All writes use transactions; all reads return typed values.

Schema (auto-created on first open):
  circuit_breaker   — one row, upserted on every engine evaluate()
  equity_snapshots  — one row per calendar day (UTC); used to derive day/week start balance
  signals           — one row per signal evaluated (approved or rejected)
  fills             — one row per fill or skip event
  last_run          — one row, the ISO date of the most recent orchestrator run
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Generator, Optional

from risk.models import CircuitBreakerState, HaltReason


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS circuit_breaker (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    is_halted             INTEGER NOT NULL,
    halt_reason           TEXT    NOT NULL,
    halted_at             TEXT,
    requires_manual_reset INTEGER NOT NULL,
    current_peak          REAL    NOT NULL,
    updated_at            TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    run_date  TEXT PRIMARY KEY,   -- ISO date "YYYY-MM-DD" (UTC)
    balance   REAL NOT NULL,
    equity    REAL NOT NULL,
    recorded_at TEXT NOT NULL     -- full UTC ISO timestamp
);

CREATE TABLE IF NOT EXISTS signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    pair             TEXT    NOT NULL,
    direction        TEXT    NOT NULL,
    entry_price      REAL,
    stop_loss        REAL,
    units            INTEGER,
    risk_amount      REAL,
    approved         INTEGER NOT NULL,
    rejection_reason TEXT,
    signal_time      TEXT    NOT NULL,
    recorded_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pair            TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    units           INTEGER NOT NULL,
    fill_price      REAL    NOT NULL,
    stop_loss       REAL    NOT NULL,
    signal_id       TEXT    NOT NULL,
    broker_trade_id TEXT,
    status          TEXT    NOT NULL,
    slippage_pips   REAL,
    filled_at       TEXT    NOT NULL,
    recorded_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS last_run (
    id       INTEGER PRIMARY KEY CHECK (id = 1),
    run_date TEXT    NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class Database:
    """
    Manages all SQLite persistence for the live trading system.

    Thread safety: each call opens a short-lived connection (or uses the
    injected connection for testing). Not designed for concurrent writers.
    """

    def __init__(self, path: str) -> None:
        """
        Open (or create) the SQLite database at `path`.

        Pass `:memory:` for an in-memory database (useful in tests).
        For in-memory databases, uses URI mode with shared cache so all connections
        see the same data.
        """
        self._path = path
        # For in-memory, use shared cache so tests work correctly
        if path == ":memory:":
            db_path = "file::memory:?cache=shared"
            use_uri = True
        else:
            db_path = path
            use_uri = False

        conn = sqlite3.connect(db_path, uri=use_uri)
        try:
            # WAL mode for file-based databases
            if path != ":memory:":
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                except Exception:
                    pass
            conn.executescript(_DDL)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Circuit breaker state
    # ------------------------------------------------------------------

    def save_circuit_breaker_state(self, state: CircuitBreakerState) -> None:
        """Upsert the single circuit breaker row."""
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO circuit_breaker
                    (id, is_halted, halt_reason, halted_at, requires_manual_reset, current_peak, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    is_halted             = excluded.is_halted,
                    halt_reason           = excluded.halt_reason,
                    halted_at             = excluded.halted_at,
                    requires_manual_reset = excluded.requires_manual_reset,
                    current_peak          = excluded.current_peak,
                    updated_at            = excluded.updated_at
                """,
                (
                    int(state.is_halted),
                    state.halt_reason.value,
                    state.halted_at.isoformat() if state.halted_at else None,
                    int(state.requires_manual_reset),
                    state.current_peak,
                    now,
                ),
            )

    def load_circuit_breaker_state(self) -> CircuitBreakerState:
        """
        Load the persisted circuit breaker state.
        Returns a default (not halted, peak=0) state if no row exists yet.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT is_halted, halt_reason, halted_at, requires_manual_reset, current_peak "
                "FROM circuit_breaker WHERE id = 1"
            ).fetchone()

        if row is None:
            return CircuitBreakerState()

        halted_at = None
        if row[2] is not None:
            halted_at = datetime.fromisoformat(row[2])
            if halted_at.tzinfo is None:
                halted_at = halted_at.replace(tzinfo=timezone.utc)

        return CircuitBreakerState(
            is_halted             = bool(row[0]),
            halt_reason           = HaltReason(row[1]),
            halted_at             = halted_at,
            requires_manual_reset = bool(row[3]),
            current_peak          = float(row[4]),
        )

    # ------------------------------------------------------------------
    # Equity snapshots
    # ------------------------------------------------------------------

    def save_equity_snapshot(
        self,
        balance:  float,
        equity:   float,
        run_date: Optional[date] = None,
    ) -> None:
        """
        Save the end-of-session equity snapshot for `run_date` (defaults to today UTC).

        Uses INSERT OR REPLACE so a second run on the same day updates the row.
        """
        d   = run_date or datetime.now(tz=timezone.utc).date()
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO equity_snapshots (run_date, balance, equity, recorded_at) "
                "VALUES (?, ?, ?, ?)",
                (d.isoformat(), balance, equity, now),
            )

    def get_balance_on_date(self, d: date) -> Optional[float]:
        """Return the balance recorded for `d`, or None if no snapshot exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT balance FROM equity_snapshots WHERE run_date = ?",
                (d.isoformat(),),
            ).fetchone()
        return float(row[0]) if row else None

    def get_day_start_balance(self, today: Optional[date] = None) -> Optional[float]:
        """
        Return yesterday's closing balance (= today's session start balance).

        For OANDA FX: the D1 bar opens at 22:00 UTC, so "today's session" runs
        from 22:00 UTC yesterday to 22:00 UTC today. The day-start balance is
        yesterday's closing balance.
        """
        d = (today or datetime.now(tz=timezone.utc).date()) - timedelta(days=1)
        return self.get_balance_on_date(d)

    def get_week_start_balance(self, today: Optional[date] = None) -> Optional[float]:
        """
        Return the most recent Monday's closing balance (= week-start balance).

        Walks back up to 7 days to find the last Monday snapshot. Returns None
        if no Monday snapshot exists (first week of operation).
        """
        d = today or datetime.now(tz=timezone.utc).date()
        # Find most recent Monday on or before today
        days_since_monday = d.weekday()  # Monday=0
        monday = d - timedelta(days=days_since_monday)

        # Try up to 2 weeks back (in case Monday was a holiday with no snapshot)
        for weeks_back in range(3):
            candidate = monday - timedelta(weeks=weeks_back)
            result = self.get_balance_on_date(candidate)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------
    # Signal logging
    # ------------------------------------------------------------------

    def save_signal(
        self,
        pair:             str,
        direction:        str,
        entry_price:      Optional[float],
        stop_loss:        Optional[float],
        units:            Optional[int],
        risk_amount:      Optional[float],
        approved:         bool,
        rejection_reason: Optional[str],
        signal_time:      datetime,
    ) -> None:
        """Log a signal evaluation result (approved or rejected)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals
                    (pair, direction, entry_price, stop_loss, units, risk_amount,
                     approved, rejection_reason, signal_time, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pair, direction, entry_price, stop_loss, units, risk_amount,
                    int(approved), rejection_reason,
                    signal_time.isoformat(), _now_iso(),
                ),
            )

    # ------------------------------------------------------------------
    # Fill logging
    # ------------------------------------------------------------------

    def save_fill(
        self,
        pair:           str,
        side:           str,
        units:          int,
        fill_price:     float,
        stop_loss:      float,
        signal_id:      str,
        broker_trade_id: Optional[str],
        status:         str,
        slippage_pips:  Optional[float],
        filled_at:      datetime,
    ) -> None:
        """Log a fill event (FILLED, ALREADY_OPEN, or broker-rejected)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fills
                    (pair, side, units, fill_price, stop_loss, signal_id,
                     broker_trade_id, status, slippage_pips, filled_at, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pair, side, units, fill_price, stop_loss, signal_id,
                    broker_trade_id, status, slippage_pips,
                    filled_at.isoformat(), _now_iso(),
                ),
            )

    # ------------------------------------------------------------------
    # Run-date tracking (idempotency)
    # ------------------------------------------------------------------

    def get_last_run_date(self) -> Optional[date]:
        """Return the date of the most recent orchestrator run, or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT run_date FROM last_run WHERE id = 1").fetchone()
        if row is None:
            return None
        return date.fromisoformat(row[0])

    def set_last_run_date(self, d: Optional[date] = None) -> None:
        """Record today's date as the last run date."""
        run_date = (d or datetime.now(tz=timezone.utc).date()).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO last_run (id, run_date) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET run_date = excluded.run_date",
                (run_date,),
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
