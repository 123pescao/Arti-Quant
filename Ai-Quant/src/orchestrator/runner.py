"""
Phase 7 — Orchestrator.

The single entry point for each live trading session. Called once per day at
run_hour_utc (default 22:00 UTC, immediately after the D1 bar closes).

Run sequence (fail-closed at each step):
  1. Time guard         — reject if current UTC hour != run_hour_utc
  2. Idempotency check  — reject if already ran today (prevents double-execution on restart)
  3. Load CB state      — from storage; halted state blocks all pairs
  4. Fetch account      — balance, equity, open positions from OANDA
  5. Build AccountState — inject day/week start balances from storage
  6. Per-pair loop      — for each pair, independently:
     a. Fetch D1 candles
     b. Evaluate strategy → Signal or None
     c. Evaluate risk engine → RiskDecision
     d. Execute if approved → Fill + SlippageRecord
     e. Persist signal + fill
  7. Save CB state      — persist any circuit breaker mutations
  8. Save equity snap   — today's balance/equity for future day/week start queries
  9. Set last run date  — idempotency marker
  10. Send daily report — summary alert via Telegram

Error isolation: exceptions in one pair are caught, logged, and do NOT abort
processing of remaining pairs. The error is surfaced in OrchestratorResult.

Fail-closed rules:
  - No account state → abort entire run (no safe defaults for position sizing)
  - No candles for a pair → skip that pair (log warning)
  - Execution error → log error + alert, skip that pair
  - Alert failure → log only, never re-raise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional, Protocol

from strategy.base import Strategy
from strategy.models import Direction, Signal
from risk.engine import RiskEngine
from risk.models import AccountState, CircuitBreakerState, RiskDecision
from execution.executor import Executor
from execution.models import Fill, SlippageRecord

from alerting.telegram import Alerter
from data.models import AccountSummary
from storage.database import Database
from .config import OrchestratorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data client protocol (satisfied by data.OandaClient)
# ---------------------------------------------------------------------------

class DataClient(Protocol):
    """Data access interface required by the orchestrator."""

    def fetch_candles_df(self, pair: str, count: int, granularity: str = "D"):
        """Return a pandas DataFrame of complete D1 bars."""
        ...

    def fetch_account_summary(self) -> AccountSummary:
        """Return current account balance, NAV, and open positions."""
        ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """Outcome of processing a single pair in one session."""
    pair:     str
    signal:   Optional[Signal]          = None
    decision: Optional[RiskDecision]    = None
    fill:     Optional[Fill]            = None
    slippage: Optional[SlippageRecord]  = None
    error:    Optional[str]             = None

    @property
    def executed(self) -> bool:
        return self.fill is not None and self.fill.status.value == "filled"

    @property
    def skipped(self) -> bool:
        return self.signal is None or self.signal.direction == Direction.FLAT

    @property
    def rejected(self) -> bool:
        return (
            self.decision is not None
            and not self.decision.approved
            and self.signal is not None
            and self.signal.direction != Direction.FLAT
        )


@dataclass
class OrchestratorResult:
    """Summary of one full orchestrator run."""
    ran:                    bool
    skipped_reason:         Optional[str]           = None
    circuit_breaker_halted: bool                    = False
    pair_results:           dict[str, PairResult]   = field(default_factory=dict)
    errors:                 list[str]               = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Coordinates one live trading session.

    All dependencies are injected. Use the factory create_orchestrator() for
    the production wiring. Pass mocks in tests.
    """

    def __init__(
        self,
        config:      OrchestratorConfig,
        data_client: DataClient,
        storage:     Database,
        alerter:     Alerter,
        strategy:    Strategy,
        risk_engine: RiskEngine,
        executor:    Executor,
    ) -> None:
        self._cfg         = config
        self._data        = data_client
        self._storage     = storage
        self._alerter     = alerter
        self._strategy    = strategy
        self._risk_engine = risk_engine
        self._executor    = executor

    def run(self, now: Optional[datetime] = None) -> OrchestratorResult:
        """
        Execute one trading session.

        `now` defaults to the current UTC time. Pass explicitly in tests to
        control time without monkey-patching.
        """
        now = now or datetime.now(tz=timezone.utc)

        # --- 1. Time guard ---
        if now.hour != self._cfg.run_hour_utc:
            reason = (
                f"Time guard: UTC hour {now.hour} != run_hour {self._cfg.run_hour_utc}"
            )
            logger.info(reason)
            return OrchestratorResult(ran=False, skipped_reason=reason)

        # --- 2. Idempotency: skip if already ran today ---
        today = now.date()
        last_run = self._storage.get_last_run_date()
        if last_run == today:
            reason = f"Already ran for {today.isoformat()} — skipping"
            logger.info(reason)
            return OrchestratorResult(ran=False, skipped_reason=reason)

        logger.info("Orchestrator starting — %s UTC", now.isoformat())

        # --- 3. Load circuit breaker state ---
        cb_state = self._storage.load_circuit_breaker_state()

        # --- 4. Fetch account summary ---
        try:
            summary = self._data.fetch_account_summary()
        except Exception as exc:
            msg = f"Fatal: cannot fetch account state — {exc}"
            logger.error(msg)
            self._alerter.send(f"[BULL-RUN ERROR] {msg}")
            return OrchestratorResult(ran=False, skipped_reason=msg)

        # --- 5. Build AccountState ---
        day_start  = self._storage.get_day_start_balance(today) or summary.balance
        week_start = self._storage.get_week_start_balance(today) or summary.balance

        account = AccountState(
            balance            = summary.balance,
            equity             = summary.nav,
            peak_balance       = summary.balance,
            day_start_balance  = day_start,
            week_start_balance = week_start,
            open_positions     = summary.open_positions,
            evaluation_time    = now,
        )

        # --- 6. Per-pair loop ---
        result = OrchestratorResult(ran=True)

        if cb_state.is_halted:
            result.circuit_breaker_halted = True
            logger.warning(
                "Circuit breaker halted (%s) — skipping all pairs",
                cb_state.halt_reason.value,
            )

        for pair in self._cfg.pairs:
            pr = PairResult(pair=pair)
            result.pair_results[pair] = pr

            if cb_state.is_halted:
                pr.error = f"Skipped: circuit breaker halted ({cb_state.halt_reason.value})"
                continue

            try:
                self._process_pair(pair, account, cb_state, pr, now)
            except Exception as exc:
                msg = f"{pair}: unhandled error — {exc}"
                logger.exception(msg)
                pr.error = msg
                result.errors.append(msg)
                self._alerter.send(f"[BULL-RUN ERROR] {msg}")

        # --- 7. Save CB state (always — peak equity may have been updated) ---
        self._storage.save_circuit_breaker_state(cb_state)

        # --- 8. Save equity snapshot ---
        self._storage.save_equity_snapshot(summary.balance, summary.nav, today)

        # --- 9. Record run date for idempotency ---
        self._storage.set_last_run_date(today)

        # --- 10. Send daily report ---
        report = _build_report(result, account, cb_state, now)
        logger.info(report)
        self._alerter.send(report)

        return result

    def _process_pair(
        self,
        pair:     str,
        account:  AccountState,
        cb_state: CircuitBreakerState,
        pr:       PairResult,
        now:      datetime,
    ) -> None:
        """Process a single pair: fetch → signal → risk → execute → persist."""

        # Fetch candles
        df = self._data.fetch_candles_df(pair, self._cfg.candle_count)
        if df.empty or len(df) < self._strategy.min_candles:
            pr.error = f"Insufficient candles ({len(df)} < {self._strategy.min_candles})"
            logger.warning("%s: %s", pair, pr.error)
            return

        # Strategy
        signal = self._strategy.evaluate(pair, df)
        pr.signal = signal

        if signal is None:
            logger.info("%s: no signal", pair)
            return

        logger.info("%s: signal %s entry=%.5f stop=%.5f",
                    pair, signal.direction.value, signal.entry_price,
                    signal.stop_loss if signal.stop_loss else 0.0)

        if signal.direction == Direction.FLAT:
            logger.info("%s: FLAT signal — no execution", pair)
            return

        # Risk engine
        decision = self._risk_engine.evaluate(signal, account, cb_state)
        pr.decision = decision

        _log_signal(self._storage, signal, decision)

        if not decision.approved:
            logger.info("%s: rejected — %s", pair, decision.rejection_reason.value)
            return

        # Execute
        fill, slippage = self._executor.execute(signal, decision)
        pr.fill    = fill
        pr.slippage = slippage

        if fill is not None:
            _log_fill(self._storage, signal, fill, slippage)
            logger.info(
                "%s: fill status=%s trade_id=%s",
                pair, fill.status.value, fill.broker_trade_id,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_signal(storage: Database, signal: Signal, decision: RiskDecision) -> None:
    """Persist a signal + risk decision to storage."""
    try:
        storage.save_signal(
            pair             = signal.pair,
            direction        = signal.direction.value,
            entry_price      = signal.entry_price,
            stop_loss        = signal.stop_loss,
            units            = decision.units if decision.approved else None,
            risk_amount      = decision.risk_amount if decision.approved else None,
            approved         = decision.approved,
            rejection_reason = decision.rejection_reason.value if decision.rejection_reason else None,
            signal_time      = signal.signal_time,
        )
    except Exception as exc:
        logger.error("Failed to persist signal for %s: %s", signal.pair, exc)


def _log_fill(
    storage:  Database,
    signal:   Signal,
    fill:     Fill,
    slippage: Optional[SlippageRecord],
) -> None:
    """Persist a fill event to storage."""
    try:
        storage.save_fill(
            pair            = fill.order.pair,
            side            = fill.order.side.value,
            units           = fill.order.units,
            fill_price      = fill.fill_price,
            stop_loss       = fill.order.stop_loss,
            signal_id       = fill.order.signal_id,
            broker_trade_id = fill.broker_trade_id,
            status          = fill.status.value,
            slippage_pips   = slippage.slippage_pips if slippage else None,
            filled_at       = fill.filled_at,
        )
    except Exception as exc:
        logger.error("Failed to persist fill for %s: %s", signal.pair, exc)


def _build_report(
    result:   OrchestratorResult,
    account:  AccountState,
    cb_state: CircuitBreakerState,
    now:      datetime,
) -> str:
    """Build a human-readable daily report string for the Telegram alert."""
    lines = [
        f"[BULL-RUN] Daily Report — {now.strftime('%Y-%m-%d %H:%M')} UTC",
        f"Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}",
    ]

    if cb_state.is_halted:
        lines.append(f"⚠ CIRCUIT BREAKER: {cb_state.halt_reason.value.upper()}")

    for pair, pr in result.pair_results.items():
        if pr.error:
            lines.append(f"  {pair}: ERROR — {pr.error}")
        elif pr.skipped:
            lines.append(f"  {pair}: no signal")
        elif pr.rejected:
            lines.append(f"  {pair}: rejected ({pr.decision.rejection_reason.value})")
        elif pr.executed:
            sl = pr.fill.order.stop_loss
            lines.append(
                f"  {pair}: FILLED {pr.signal.direction.value} "
                f"{pr.fill.order.units:,} units @ {pr.fill.fill_price:.5f} "
                f"SL={sl:.5f}"
            )
        elif pr.fill is not None:
            lines.append(f"  {pair}: {pr.fill.status.value}")

    if result.errors:
        lines.append(f"Errors: {len(result.errors)}")

    return "\n".join(lines)
