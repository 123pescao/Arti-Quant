"""
Unit tests for the execution layer (src/execution/).

Critical properties tested:
  1.  ExecutorConfig rejects invalid spread thresholds
  2.  Order model enforces positive units and stop_loss
  3.  Happy path: approved LONG/SHORT decision → Fill + SlippageRecord returned
  4.  Spread guard: skip submission if live spread > max_spread_pips
  5.  Idempotency (pre-submission): skip if broker has any open trade for the pair
      (v1 rule: max 1 per pair regardless of direction — checked per-pair, not per-pair+side)
  6.  Idempotency (cross-direction): an existing LONG blocks a new SHORT on the same pair
  7.  Slippage sign: positive pips = worse for trader; negative pips = better for trader
      BUY fill above mid → positive pips (paid more = cost)
      SELL fill above mid → negative pips (received more = benefit)
  8.  execute() raises RuntimeError on rejected RiskDecision
  9.  execute() raises RuntimeError on FLAT signal
  10. Retry: transient HTTP 429/503/504 trigger up to 3 retries, then raise ExecutionError
  11. Retry: non-transient error propagates immediately (no retry)
  12. Reconciliation: after transient timeout, if OANDA already filled the order,
      executor detects it via get_open_trade_ids and returns ALREADY_OPEN (no retry)
  13. Reconciliation: if no open position found after timeout, retry proceeds normally
  14. ALREADY_OPEN fill → no SlippageRecord (fill_price unknown)
  15. signal_id format: "{PAIR}_{DIRECTION}_{YYYYMMDD}"
  16. OANDA units sign: BUY → positive, SELL → negative
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from strategy.models import Direction, Signal
from risk.models import RiskDecision, RejectionReason

from execution.executor import Executor, ExecutionError, ExecutorConfig, _build_signal_id
from execution.models import (
    Fill, FillStatus, Order, OrderSide, SlippageRecord, SpreadSnapshot,
)


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

_TS = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def _signal(
    pair:      str       = "EUR_USD",
    direction: Direction = Direction.LONG,
    entry:     float     = 1.1000,
    stop:      float     = None,
    ts:        datetime  = _TS,
) -> Signal:
    if stop is None:
        stop = (entry - 0.0200) if direction == Direction.LONG else (entry + 0.0200)
    return Signal(
        pair        = pair,
        direction   = direction,
        entry_price = entry,
        stop_loss   = stop,
        atr         = 0.01,
        signal_time = ts,
        source_bar  = 99,
    )


def _decision_approve(units: int = 10_000, stop: float = 1.0800) -> RiskDecision:
    return RiskDecision.approve(units=units, stop_loss=stop, risk_amount=25.0)


def _decision_reject() -> RiskDecision:
    return RiskDecision.reject(RejectionReason.MAX_OPEN_POSITIONS)


def _snapshot(
    pair:   str   = "EUR_USD",
    bid:    float = 1.0998,
    ask:    float = 1.1002,
    pips:   float = 0.4,
) -> SpreadSnapshot:
    mid = (bid + ask) / 2
    return SpreadSnapshot(
        pair        = pair,
        bid         = bid,
        ask         = ask,
        mid         = mid,
        spread_pips = pips,
        sampled_at  = _TS,
    )


def _fill(
    order:    Order,
    price:    float       = 1.1002,
    trade_id: str         = "T-001",
    status:   FillStatus  = FillStatus.FILLED,
) -> Fill:
    return Fill(
        order           = order,
        fill_price      = price,
        broker_trade_id = trade_id,
        filled_at       = _TS,
        status          = status,
    )


def _make_client(
    spread:           SpreadSnapshot = None,
    open_ids:         list           = None,
    fill_price:       float          = 1.1002,
    fill_status:      FillStatus     = FillStatus.FILLED,
    submit_side_effect               = None,
) -> MagicMock:
    """
    Build a mock OandaClient.

    get_open_trade_ids(pair) returns open_ids for all calls (no side filtering).
    """
    client = MagicMock()
    client.get_spread_snapshot.return_value = spread or _snapshot()
    client.get_open_trade_ids.return_value  = open_ids or []

    if submit_side_effect is not None:
        client.submit_market_order.side_effect = submit_side_effect
    else:
        def _submit(pair, units, stop_loss, signal_id):
            order = Order(
                pair       = pair,
                side       = OrderSide.BUY if units > 0 else OrderSide.SELL,
                units      = abs(units),
                stop_loss  = stop_loss,
                signal_id  = signal_id,
                created_at = _TS,
            )
            return _fill(order, price=fill_price, status=fill_status)
        client.submit_market_order.side_effect = _submit

    return client


# ---------------------------------------------------------------------------
# ExecutorConfig tests
# ---------------------------------------------------------------------------

class TestExecutorConfig:
    def test_default_max_spread(self):
        cfg = ExecutorConfig()
        assert cfg.max_spread_pips == 3.0

    def test_custom_max_spread(self):
        cfg = ExecutorConfig(max_spread_pips=2.0)
        assert cfg.max_spread_pips == 2.0

    def test_zero_spread_invalid(self):
        with pytest.raises(ValueError, match="max_spread_pips must be positive"):
            ExecutorConfig(max_spread_pips=0.0)

    def test_negative_spread_invalid(self):
        with pytest.raises(ValueError, match="max_spread_pips must be positive"):
            ExecutorConfig(max_spread_pips=-1.0)


# ---------------------------------------------------------------------------
# Order model tests
# ---------------------------------------------------------------------------

class TestOrderModel:
    def test_order_creation(self):
        order = Order(
            pair      = "EUR_USD",
            side      = OrderSide.BUY,
            units     = 10_000,
            stop_loss = 1.08,
            signal_id = "EUR_USD_LONG_20240115",
            created_at = _TS,
        )
        assert order.units == 10_000
        assert order.side == OrderSide.BUY

    def test_order_zero_units_raises(self):
        with pytest.raises(ValueError, match="units must be positive"):
            Order(
                pair      = "EUR_USD",
                side      = OrderSide.BUY,
                units     = 0,
                stop_loss = 1.08,
                signal_id = "x",
                created_at = _TS,
            )

    def test_order_negative_stop_raises(self):
        with pytest.raises(ValueError, match="stop_loss must be positive"):
            Order(
                pair      = "EUR_USD",
                side      = OrderSide.BUY,
                units     = 1000,
                stop_loss = -1.0,
                signal_id = "x",
                created_at = _TS,
            )

    def test_order_is_frozen(self):
        order = Order(
            pair="EUR_USD", side=OrderSide.BUY, units=1000,
            stop_loss=1.08, signal_id="x", created_at=_TS,
        )
        with pytest.raises(Exception):
            order.units = 999  # type: ignore


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_long_returns_fill_and_slippage(self):
        sig      = _signal("EUR_USD", Direction.LONG, entry=1.1000, stop=1.0800)
        decision = _decision_approve(units=10_000, stop=1.0800)
        client   = _make_client(fill_price=1.1002)

        executor = Executor(client)
        fill, slippage = executor.execute(sig, decision)

        assert fill is not None
        assert fill.status == FillStatus.FILLED
        assert slippage is not None

    def test_short_returns_fill_and_slippage(self):
        sig      = _signal("EUR_USD", Direction.SHORT, entry=1.1000, stop=1.1200)
        decision = RiskDecision.approve(units=10_000, stop_loss=1.1200, risk_amount=25.0)
        client   = _make_client(fill_price=1.0998)

        executor = Executor(client)
        fill, slippage = executor.execute(sig, decision)

        assert fill is not None
        assert fill.status == FillStatus.FILLED
        assert slippage is not None

    def test_buy_units_positive_to_broker(self):
        """OANDA convention: positive units = buy."""
        sig      = _signal(direction=Direction.LONG)
        decision = _decision_approve(units=5_000)
        client   = _make_client()

        Executor(client).execute(sig, decision)

        submitted_units = client.submit_market_order.call_args[1]["units"]
        assert submitted_units == 5_000

    def test_sell_units_negative_to_broker(self):
        """OANDA convention: negative units = sell."""
        sig      = _signal(direction=Direction.SHORT, stop=1.1200)
        decision = RiskDecision.approve(units=5_000, stop_loss=1.1200, risk_amount=25.0)
        client   = _make_client()

        Executor(client).execute(sig, decision)

        submitted_units = client.submit_market_order.call_args[1]["units"]
        assert submitted_units == -5_000

    def test_stop_loss_passed_to_broker(self):
        sig      = _signal(stop=1.0750)
        decision = _decision_approve(stop=1.0750)
        client   = _make_client()

        Executor(client).execute(sig, decision)

        submitted_stop = client.submit_market_order.call_args[1]["stop_loss"]
        assert submitted_stop == pytest.approx(1.0750)


# ---------------------------------------------------------------------------
# Slippage calculation tests
# ---------------------------------------------------------------------------

class TestSlippageCalculation:
    def test_buy_fill_above_mid_is_positive_slippage(self):
        """
        BUY fill at 1.1002, mid=1.1000 → +2.0 pips (positive = worse: paid more than mid).
        Formula: (fill - mid) / pip_size = 0.0002 / 0.0001 = +2.0
        """
        snap = _snapshot(bid=1.0999, ask=1.1001)  # mid=1.1000
        sig  = _signal(direction=Direction.LONG)
        dec  = _decision_approve()
        client = _make_client(spread=snap, fill_price=1.1002)

        _, slippage = Executor(client).execute(sig, dec)

        assert slippage is not None
        assert slippage.slippage_pips == pytest.approx(2.0, abs=0.01)

    def test_buy_fill_below_mid_is_negative_slippage(self):
        """
        BUY fill at 1.0998, mid=1.1000 → -2.0 pips (negative = better: paid less than mid).
        Formula: (fill - mid) / pip_size = -0.0002 / 0.0001 = -2.0
        """
        snap = _snapshot(bid=1.0999, ask=1.1001)  # mid=1.1000
        sig  = _signal(direction=Direction.LONG)
        dec  = _decision_approve()
        client = _make_client(spread=snap, fill_price=1.0998)

        _, slippage = Executor(client).execute(sig, dec)

        assert slippage is not None
        assert slippage.slippage_pips == pytest.approx(-2.0, abs=0.01)

    def test_sell_fill_above_mid_is_negative_slippage(self):
        """
        SELL fill at 1.1002, mid=1.1000 → -2.0 pips (negative = better: received more than mid).
        Formula: (mid - fill) / pip_size = (1.1000 - 1.1002) / 0.0001 = -2.0
        A SELL at a higher price benefits the trader (more received), so this is better (-).
        """
        snap = _snapshot(bid=1.0999, ask=1.1001)  # mid=1.1000
        sig  = _signal(direction=Direction.SHORT, stop=1.1200)
        dec  = RiskDecision.approve(units=10_000, stop_loss=1.1200, risk_amount=25.0)
        client = _make_client(spread=snap, fill_price=1.1002)

        _, slippage = Executor(client).execute(sig, dec)

        assert slippage is not None
        assert slippage.slippage_pips == pytest.approx(-2.0, abs=0.01)

    def test_sell_fill_below_mid_is_positive_slippage(self):
        """
        SELL fill at 1.0998, mid=1.1000 → +2.0 pips (positive = worse: received less than mid).
        Formula: (mid - fill) / pip_size = (1.1000 - 1.0998) / 0.0001 = +2.0
        A SELL at a lower price hurts the trader (less received), so this is worse (+).
        """
        snap = _snapshot(bid=1.0999, ask=1.1001)  # mid=1.1000
        sig  = _signal(direction=Direction.SHORT, stop=1.1200)
        dec  = RiskDecision.approve(units=10_000, stop_loss=1.1200, risk_amount=25.0)
        client = _make_client(spread=snap, fill_price=1.0998)

        _, slippage = Executor(client).execute(sig, dec)

        assert slippage is not None
        assert slippage.slippage_pips == pytest.approx(2.0, abs=0.01)

    def test_slippage_fields_populated(self):
        snap = _snapshot(bid=1.0999, ask=1.1001)
        sig  = _signal(pair="EUR_USD", direction=Direction.LONG)
        dec  = _decision_approve()
        client = _make_client(spread=snap, fill_price=1.1003)

        _, slippage = Executor(client).execute(sig, dec)

        assert slippage.pair == "EUR_USD"
        assert slippage.side == OrderSide.BUY
        assert slippage.expected_price == pytest.approx(1.1000)
        assert slippage.fill_price     == pytest.approx(1.1003)
        assert slippage.signal_id.startswith("EUR_USD_LONG_")


# ---------------------------------------------------------------------------
# Spread guard tests
# ---------------------------------------------------------------------------

class TestSpreadGuard:
    def test_wide_spread_skips_execution(self):
        snap   = _snapshot(pips=5.0)
        sig    = _signal()
        dec    = _decision_approve()
        client = _make_client(spread=snap)

        executor = Executor(client, ExecutorConfig(max_spread_pips=3.0))
        fill, slippage = executor.execute(sig, dec)

        assert fill is None
        assert slippage is None
        client.submit_market_order.assert_not_called()

    def test_spread_at_exactly_max_is_allowed(self):
        """spread == max_spread_pips is accepted (boundary: > not >=)."""
        snap   = _snapshot(pips=3.0)
        sig    = _signal()
        dec    = _decision_approve()
        client = _make_client(spread=snap)

        executor = Executor(client, ExecutorConfig(max_spread_pips=3.0))
        fill, slippage = executor.execute(sig, dec)

        assert fill is not None

    def test_spread_just_above_max_rejected(self):
        snap   = _snapshot(pips=3.1)
        sig    = _signal()
        dec    = _decision_approve()
        client = _make_client(spread=snap)

        executor = Executor(client, ExecutorConfig(max_spread_pips=3.0))
        fill, _ = executor.execute(sig, dec)

        assert fill is None


# ---------------------------------------------------------------------------
# Idempotency tests
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_existing_open_trade_skips_submission(self):
        sig    = _signal(direction=Direction.LONG)
        dec    = _decision_approve()
        client = _make_client(open_ids=["T-123"])

        fill, slippage = Executor(client).execute(sig, dec)

        assert fill is None
        assert slippage is None
        client.submit_market_order.assert_not_called()

    def test_no_open_trade_proceeds(self):
        sig    = _signal(direction=Direction.LONG)
        dec    = _decision_approve()
        client = _make_client(open_ids=[])

        fill, _ = Executor(client).execute(sig, dec)

        assert fill is not None
        client.submit_market_order.assert_called_once()

    def test_idempotency_checks_pair_only(self):
        """
        The idempotency check queries get_open_trade_ids(pair) with NO side argument.
        v1 rule: max 1 open position per pair, any direction.
        """
        sig    = _signal(pair="GBP_USD", direction=Direction.LONG)
        dec    = _decision_approve()
        client = _make_client()

        Executor(client).execute(sig, dec)

        # Must be called with pair only — no OrderSide argument
        client.get_open_trade_ids.assert_called_once_with("GBP_USD")

    def test_idempotency_blocks_opposite_direction_on_same_pair(self):
        """
        An existing LONG on EUR_USD must block a new SHORT on EUR_USD.
        The check is per-pair (not per pair+side), enforcing v1: max 1 per pair.
        """
        sig    = _signal(pair="EUR_USD", direction=Direction.SHORT, stop=1.1200)
        dec    = RiskDecision.approve(units=10_000, stop_loss=1.1200, risk_amount=25.0)
        # open_ids represents a LONG already open — idempotency must still fire for SHORT
        client = _make_client(open_ids=["T-LONG-123"])

        fill, slippage = Executor(client).execute(sig, dec)

        assert fill is None
        assert slippage is None
        client.submit_market_order.assert_not_called()

    def test_idempotency_does_not_block_different_pair(self):
        """Open position on GBP_USD must NOT block a new trade on EUR_USD."""
        sig    = _signal(pair="EUR_USD", direction=Direction.LONG)
        dec    = _decision_approve()

        client = MagicMock()
        client.get_spread_snapshot.return_value = _snapshot()
        # Return empty for EUR_USD, non-empty would come from GBP_USD — simulate correct routing
        client.get_open_trade_ids.return_value = []  # EUR_USD has no open trades

        def _submit(pair, units, stop_loss, signal_id):
            order = Order(pair=pair, side=OrderSide.BUY, units=abs(units),
                          stop_loss=stop_loss, signal_id=signal_id, created_at=_TS)
            return _fill(order)
        client.submit_market_order.side_effect = _submit

        fill, _ = Executor(client).execute(sig, dec)

        assert fill is not None
        client.submit_market_order.assert_called_once()


# ---------------------------------------------------------------------------
# Reconciliation tests (ambiguous timeout protection)
# ---------------------------------------------------------------------------

class TestReconciliation:
    def test_reconcile_detects_silent_fill_returns_already_open(self):
        """
        Scenario: OANDA receives and fills the order but the response is lost (HTTP 504).
        Reconciliation detects the open position and returns ALREADY_OPEN, suppressing retry.
        """
        sig = _signal()
        dec = _decision_approve()

        submit_calls = {"n": 0}

        def _submit(pair, units, stop_loss, signal_id):
            submit_calls["n"] += 1
            raise ExecutionError("Gateway timeout", http_status=504)

        # Pre-submission check: no open positions
        # Post-timeout reconciliation: position appeared (silent fill)
        get_ids_calls = {"n": 0}

        def _get_open_ids(pair):
            get_ids_calls["n"] += 1
            if get_ids_calls["n"] <= 1:  # first call = pre-submission idempotency check
                return []
            return ["T-SILENT-FILL"]   # reconciliation: position is now open

        client = MagicMock()
        client.get_spread_snapshot.return_value = _snapshot()
        client.get_open_trade_ids.side_effect = _get_open_ids
        client.submit_market_order.side_effect = _submit

        with patch("execution.executor.time.sleep"):
            fill, slippage = Executor(client).execute(sig, dec)

        assert fill is not None
        assert fill.status == FillStatus.ALREADY_OPEN
        assert fill.broker_trade_id == "T-SILENT-FILL"
        assert slippage is None                          # fill_price unknown — no slippage record
        assert submit_calls["n"] == 1                    # no duplicate submission

    def test_reconcile_finds_nothing_allows_retry(self):
        """
        Scenario: order times out AND OANDA did not process it.
        Reconciliation finds no open position, so the retry proceeds and succeeds.
        """
        sig = _signal()
        dec = _decision_approve()

        attempt = {"n": 0}

        def _submit(pair, units, stop_loss, signal_id):
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise ExecutionError("Gateway timeout", http_status=504)
            order = Order(pair=pair, side=OrderSide.BUY, units=abs(units),
                          stop_loss=stop_loss, signal_id=signal_id, created_at=_TS)
            return _fill(order, price=1.1002)

        client = MagicMock()
        client.get_spread_snapshot.return_value = _snapshot()
        client.get_open_trade_ids.return_value = []   # no open positions at any check
        client.submit_market_order.side_effect = _submit

        with patch("execution.executor.time.sleep"):
            fill, slippage = Executor(client).execute(sig, dec)

        assert fill is not None
        assert fill.status == FillStatus.FILLED
        assert slippage is not None
        assert attempt["n"] == 2   # initial attempt + one retry

    def test_already_open_fill_produces_no_slippage_record(self):
        """
        ALREADY_OPEN fill has fill_price=0.0 (unknown), so no SlippageRecord is produced.
        """
        sig = _signal()
        dec = _decision_approve()

        def _submit(pair, units, stop_loss, signal_id):
            raise ExecutionError("Gateway timeout", http_status=504)

        get_ids_calls = {"n": 0}

        def _get_open_ids(pair):
            get_ids_calls["n"] += 1
            if get_ids_calls["n"] <= 1:
                return []
            return ["T-SILENT-FILL"]

        client = MagicMock()
        client.get_spread_snapshot.return_value = _snapshot()
        client.get_open_trade_ids.side_effect = _get_open_ids
        client.submit_market_order.side_effect = _submit

        with patch("execution.executor.time.sleep"):
            fill, slippage = Executor(client).execute(sig, dec)

        assert fill.status == FillStatus.ALREADY_OPEN
        assert fill.fill_price == 0.0
        assert slippage is None

    def test_already_open_broker_trade_id_is_reconciled_id(self):
        """broker_trade_id in the ALREADY_OPEN fill is taken from get_open_trade_ids."""
        sig = _signal()
        dec = _decision_approve()

        get_ids_calls = {"n": 0}

        def _get_open_ids(pair):
            get_ids_calls["n"] += 1
            if get_ids_calls["n"] <= 1:
                return []
            return ["T-RECONCILED-999", "T-OTHER"]  # first id used

        client = MagicMock()
        client.get_spread_snapshot.return_value = _snapshot()
        client.get_open_trade_ids.side_effect = _get_open_ids
        client.submit_market_order.side_effect = ExecutionError("Timeout", http_status=504)

        with patch("execution.executor.time.sleep"):
            fill, _ = Executor(client).execute(sig, dec)

        assert fill.broker_trade_id == "T-RECONCILED-999"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_rejected_decision_raises(self):
        sig = _signal()
        dec = _decision_reject()
        client = _make_client()

        with pytest.raises(RuntimeError, match="rejected decision"):
            Executor(client).execute(sig, dec)

    def test_flat_signal_raises(self):
        sig = Signal(
            pair        = "EUR_USD",
            direction   = Direction.FLAT,
            entry_price = 1.10,
            stop_loss   = None,
            atr         = 0.01,
            signal_time = _TS,
            source_bar  = 99,
        )
        dec    = _decision_approve()
        client = _make_client()

        with pytest.raises(RuntimeError, match="FLAT signal"):
            Executor(client).execute(sig, dec)

    def test_non_transient_error_propagates_immediately(self):
        sig    = _signal()
        dec    = _decision_approve()
        client = _make_client(
            submit_side_effect=ExecutionError("Bad request", http_status=400)
        )

        with pytest.raises(ExecutionError):
            Executor(client).execute(sig, dec)

        assert client.submit_market_order.call_count == 1


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_transient_429_retries_three_times(self):
        sig    = _signal()
        dec    = _decision_approve()
        client = _make_client(
            submit_side_effect=ExecutionError("Rate limited", http_status=429)
        )
        # Reconciliation always finds nothing → retries continue
        client.get_open_trade_ids.return_value = []

        with patch("execution.executor.time.sleep"):
            with pytest.raises(ExecutionError):
                Executor(client).execute(sig, dec)

        assert client.submit_market_order.call_count == 3

    def test_transient_503_retries(self):
        sig    = _signal()
        dec    = _decision_approve()
        client = _make_client(
            submit_side_effect=ExecutionError("Service unavailable", http_status=503)
        )
        client.get_open_trade_ids.return_value = []

        with patch("execution.executor.time.sleep"):
            with pytest.raises(ExecutionError):
                Executor(client).execute(sig, dec)

        assert client.submit_market_order.call_count == 3

    def test_succeeds_on_second_attempt_after_transient_error(self):
        """First call raises 503, second call succeeds."""
        sig = _signal()
        dec = _decision_approve()

        attempt = {"n": 0}

        def _submit(pair, units, stop_loss, signal_id):
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise ExecutionError("Service unavailable", http_status=503)
            order = Order(
                pair=pair, side=OrderSide.BUY, units=abs(units),
                stop_loss=stop_loss, signal_id=signal_id, created_at=_TS,
            )
            return _fill(order, price=1.1002)

        client = _make_client(submit_side_effect=_submit)
        client.get_open_trade_ids.return_value = []  # reconciliation finds nothing

        with patch("execution.executor.time.sleep"):
            fill, slippage = Executor(client).execute(sig, dec)

        assert fill is not None
        assert fill.status == FillStatus.FILLED
        assert client.submit_market_order.call_count == 2

    def test_backoff_sleep_called_on_retry(self):
        """Exponential backoff: sleep(1.0), sleep(2.0) between the first two retries only."""
        sig    = _signal()
        dec    = _decision_approve()
        client = _make_client(
            submit_side_effect=ExecutionError("Rate limited", http_status=429)
        )
        client.get_open_trade_ids.return_value = []

        with patch("execution.executor.time.sleep") as mock_sleep:
            with pytest.raises(ExecutionError):
                Executor(client).execute(sig, dec)

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 2.0]

    def test_non_transient_does_not_sleep(self):
        """Non-transient errors (HTTP 400) propagate immediately with no sleep."""
        sig    = _signal()
        dec    = _decision_approve()
        client = _make_client(
            submit_side_effect=ExecutionError("Bad request", http_status=400)
        )

        with patch("execution.executor.time.sleep") as mock_sleep:
            with pytest.raises(ExecutionError):
                Executor(client).execute(sig, dec)

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# signal_id tests
# ---------------------------------------------------------------------------

class TestSignalId:
    def test_signal_id_format(self):
        sig = _signal(pair="EUR_USD", direction=Direction.LONG, ts=datetime(2024, 1, 15, tzinfo=timezone.utc))
        sid = _build_signal_id(sig)
        assert sid == "EUR_USD_LONG_20240115"

    def test_short_signal_id(self):
        sig = _signal(pair="GBP_USD", direction=Direction.SHORT,
                      ts=datetime(2024, 3, 7, tzinfo=timezone.utc), stop=1.2800)
        sid = _build_signal_id(sig)
        assert sid == "GBP_USD_SHORT_20240307"

    def test_signal_id_passed_to_broker(self):
        sig    = _signal(pair="EUR_USD", direction=Direction.LONG, ts=datetime(2024, 1, 15, tzinfo=timezone.utc))
        dec    = _decision_approve()
        client = _make_client()

        Executor(client).execute(sig, dec)

        submitted_id = client.submit_market_order.call_args[1]["signal_id"]
        assert submitted_id == "EUR_USD_LONG_20240115"
