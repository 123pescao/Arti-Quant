"""
Execution layer: translates a RiskDecision + Signal into a live OANDA order.

Responsibilities:
  1. Spread guard       — fetch live bid/ask; reject if spread > max_spread_pips
  2. Order creation     — build Order from Signal + RiskDecision
  3. Idempotency        — skip if broker already holds any open trade for this pair
                          (v1: max 1 position per pair regardless of direction)
  4. Submission         — POST market order with attached stop-loss to OANDA v20 REST API
  5. Reconciliation     — after a transient timeout, re-check open trades before retrying
                          to prevent duplicate submission when OANDA silently filled the order
  6. Slippage logging   — compare expected mid to actual fill price, record SlippageRecord
  7. Retry logic        — up to 3 attempts with exponential backoff on transient failures

The executor is stateless: all state (open positions, slippage log) lives in the
caller-supplied OandaClient. The caller is responsible for persisting SlippageRecords.

OANDA v20 API assumptions:
  - Base URL and account ID are provided via OandaConfig
  - Auth token is provided via OandaConfig.access_token
  - Units: positive = buy, negative = sell (OANDA convention)
  - Bracket orders: market order body includes stopLossOnFill
  - Transient errors: HTTP 429, 503, 504 trigger retry; others propagate immediately

Ambiguous timeout handling:
  If OANDA receives and fills the order but the TCP response is lost (HTTP 504),
  the executor detects this via a reconciliation query after the error and before
  the retry. If a position is already open for the pair, the original order
  is assumed to have landed and the retry is suppressed (returns ALREADY_OPEN fill).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Protocol

from strategy.models import Direction, Signal
from risk.models import RiskDecision

from .models import Fill, FillStatus, Order, OrderSide, SlippageRecord, SpreadSnapshot

logger = logging.getLogger(__name__)

# Pip sizes per pair — must match risk/engine.py
_PIP_SIZE: dict[str, float] = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,
    "AUD_USD": 0.0001,
    "USD_CAD": 0.0001,
}

_TRANSIENT_HTTP_CODES = {429, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Protocol interfaces (dependency-injected for testability)
# ---------------------------------------------------------------------------

class OandaClient(Protocol):
    """Minimal broker interface required by the executor."""

    def get_spread_snapshot(self, pair: str) -> SpreadSnapshot:
        """Fetch current bid/ask for the pair."""
        ...

    def get_open_trade_ids(self, pair: str) -> list[str]:
        """
        Return all broker trade IDs currently open for this pair, any direction.

        Used for both pre-submission idempotency (skip if any position open)
        and post-timeout reconciliation (detect silent fills before retry).
        """
        ...

    def submit_market_order(
        self,
        pair:      str,
        units:     int,     # positive=buy, negative=sell (OANDA sign convention)
        stop_loss: float,
        signal_id: str,
    ) -> Fill:
        """Submit market order with stop-loss attached. Returns Fill."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ExecutorConfig:
    """
    Execution-layer configuration.

    max_spread_pips:  Reject the order if live spread exceeds this threshold.
                      Default 3.0 pips covers normal EUR/USD conditions.
    """
    def __init__(self, max_spread_pips: float = 3.0) -> None:
        if max_spread_pips <= 0:
            raise ValueError(f"max_spread_pips must be positive, got {max_spread_pips}")
        self.max_spread_pips = max_spread_pips


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Translates an approved RiskDecision into a live OANDA order.

    Usage:
        executor = Executor(client, config)
        fill, slippage = executor.execute(signal, decision)
        # persist slippage to SQLite (None when fill.status != FILLED)

    Returns (Fill, Optional[SlippageRecord]).
    SlippageRecord is None when execution is skipped (spread guard, idempotency)
    or when reconciled after a timeout (fill_price unknown).
    """

    def __init__(
        self,
        client: OandaClient,
        config: Optional[ExecutorConfig] = None,
    ) -> None:
        self._client = client
        self._config = config or ExecutorConfig()

    @property
    def config(self) -> ExecutorConfig:
        return self._config

    def execute(
        self,
        signal:   Signal,
        decision: RiskDecision,
    ) -> tuple[Optional[Fill], Optional[SlippageRecord]]:
        """
        Execute an approved RiskDecision.

        Returns:
            (Fill[FILLED], SlippageRecord) on confirmed broker fill.
            (Fill[ALREADY_OPEN], None) when reconciled after a timeout.
            (None, None) if skipped (spread guard or pre-submission idempotency).

        Raises:
            RuntimeError if decision is not approved, or signal is FLAT.
            ExecutionError on non-transient broker errors or retries exhausted.
        """
        if not decision.approved:
            raise RuntimeError(
                f"execute() called with rejected decision: {decision.rejection_reason}"
            )

        if signal.direction == Direction.FLAT:
            raise RuntimeError("execute() called with FLAT signal — FLAT signals are close-only")

        now = datetime.now(tz=timezone.utc)

        # --- 1. Spread guard ---
        snapshot = self._client.get_spread_snapshot(signal.pair)
        if snapshot.spread_pips > self._config.max_spread_pips:
            logger.warning(
                "Spread guard: %s spread=%.1f pips > max=%.1f pips — skipping",
                signal.pair, snapshot.spread_pips, self._config.max_spread_pips,
            )
            return None, None

        # --- 2. Build order ---
        side = OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL
        signal_id = _build_signal_id(signal)

        # --- 3. Pre-submission idempotency: skip if ANY position open for this pair ---
        # Enforces v1 rule: max 1 open position per pair, any direction.
        open_ids = self._client.get_open_trade_ids(signal.pair)
        if open_ids:
            logger.info(
                "Idempotency: %s already has open trade(s) %s — skipping",
                signal.pair, open_ids,
            )
            return None, None

        order = Order(
            pair       = signal.pair,
            side       = side,
            units      = decision.units,
            stop_loss  = decision.stop_loss,  # type: ignore[arg-type]  # approved => not None
            signal_id  = signal_id,
            created_at = now,
        )

        # --- 4. Submit with retries and reconciliation ---
        fill = self._submit_with_retry(order)

        # --- 5. Slippage logging (only for confirmed fills; not for reconciled ALREADY_OPEN) ---
        slippage = None
        if fill.status == FillStatus.FILLED:
            pip_size = _PIP_SIZE.get(signal.pair, 0.0001)
            raw_slippage = fill.fill_price - snapshot.mid
            # BUY: positive raw = paid more than mid = cost (positive pips)
            # SELL: positive raw = received more than mid = benefit (negative pips, inverted)
            directional_slippage = raw_slippage if side == OrderSide.BUY else -raw_slippage
            slippage_pips = directional_slippage / pip_size

            slippage = SlippageRecord(
                pair            = signal.pair,
                side            = side,
                expected_price  = snapshot.mid,
                fill_price      = fill.fill_price,
                slippage_pips   = slippage_pips,
                signal_id       = signal_id,
                recorded_at     = fill.filled_at,
            )
            logger.info(
                "Fill: %s %s units=%d fill=%.5f mid=%.5f slippage=%.2f pips trade_id=%s",
                signal.pair, side.value, decision.units,
                fill.fill_price, snapshot.mid, slippage_pips,
                fill.broker_trade_id,
            )

        return fill, slippage

    def _submit_with_retry(self, order: Order) -> Fill:
        """
        Submit order to broker with up to _MAX_RETRIES attempts on transient errors.

        After each transient error, reconciles against open positions before retrying.
        If OANDA silently filled the order during a timeout, returns ALREADY_OPEN
        without re-submitting.
        """
        oanda_units = order.units if order.side == OrderSide.BUY else -order.units
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                fill = self._client.submit_market_order(
                    pair      = order.pair,
                    units     = oanda_units,
                    stop_loss = order.stop_loss,
                    signal_id = order.signal_id,
                )
                return fill
            except ExecutionError as exc:
                if exc.http_status not in _TRANSIENT_HTTP_CODES:
                    raise
                last_exc = exc

                # Reconcile: check if the order silently landed before retrying.
                # Prevents duplicate submission when OANDA filled but the response
                # was lost due to a network timeout (HTTP 504 ambiguous case).
                reconciled = self._reconcile_after_timeout(order)
                if reconciled is not None:
                    return reconciled

                if attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF_BASE_SECONDS * (2 ** attempt)
                    logger.warning(
                        "Transient broker error (HTTP %s) on attempt %d/%d — retrying in %.1fs",
                        exc.http_status, attempt + 1, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        "Transient broker error (HTTP %s) on attempt %d/%d — no more retries",
                        exc.http_status, attempt + 1, _MAX_RETRIES,
                    )

        raise ExecutionError(
            f"Order submission failed after {_MAX_RETRIES} attempts: {last_exc}",
            http_status=0,
        ) from last_exc

    def _reconcile_after_timeout(self, order: Order) -> Optional[Fill]:
        """
        After a transient network error, query the broker for open positions.

        If any position is open for the pair, the original order is assumed to have
        landed silently. Returns an ALREADY_OPEN fill with fill_price=0.0 (unknown)
        and suppresses the retry. Returns None if no position is found (safe to retry).
        """
        open_ids = self._client.get_open_trade_ids(order.pair)
        if not open_ids:
            return None
        logger.warning(
            "Reconciliation: %s has open trade(s) %s after timeout — "
            "assuming order landed; suppressing retry to prevent duplication",
            order.pair, open_ids,
        )
        return Fill(
            order           = order,
            fill_price      = 0.0,   # unknown — position confirmed via reconciliation, not fill response
            broker_trade_id = open_ids[0],
            filled_at       = datetime.now(tz=timezone.utc),
            status          = FillStatus.ALREADY_OPEN,
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ExecutionError(Exception):
    """Raised when broker returns a non-retryable error or retries are exhausted."""

    def __init__(self, message: str, http_status: int) -> None:
        super().__init__(message)
        self.http_status = http_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_signal_id(signal: Signal) -> str:
    """
    Build a deterministic signal identifier for idempotency checks.

    Format: "{PAIR}_{DIRECTION}_{DATE}" e.g. "EUR_USD_LONG_20240115"
    Unique per pair+direction+day — prevents duplicate orders on the same signal.
    """
    date_str = signal.signal_time.strftime("%Y%m%d")
    return f"{signal.pair}_{signal.direction.value}_{date_str}"
