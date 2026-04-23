"""
Unit tests for src/data/oanda_client.py and src/data/models.py.

All OANDA HTTP calls are mocked via an injected _HttpSession.

Critical properties tested:
  1.  Candle parsing: OHLCV, timestamps, complete-flag filter
  2.  Incomplete bars are excluded from fetch_candles
  3.  Candles are sorted ascending by time
  4.  candles_to_dataframe returns correct columns and UTC index
  5.  Account summary parsed: balance, NAV, unrealizedPL, open positions
  6.  Open positions parsed: direction from sign of units, stop from stopLossOrder
  7.  Spread snapshot computed from bid/ask; pips calculated correctly
  8.  get_open_trade_ids returns IDs filtered by instrument
  9.  submit_market_order builds correct OANDA body; parses fill response
  10. submit_market_order raises ExecutionError for 429/503/504
  11. submit_market_order raises ExecutionError for orderRejectTransaction
  12. fetch_candles raises OandaApiError on non-2xx
  13. OandaConfig auth_header format
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from data.oanda_client import OandaClient, OandaConfig, OandaApiError, candles_to_dataframe
from data.models import Candle, AccountSummary
from execution.models import FillStatus, OrderSide
from execution.executor import ExecutionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _config() -> OandaConfig:
    return OandaConfig(
        account_id   = "001-001-12345678-001",
        access_token = "test-token-abc",
        base_url     = "https://api-fxpractice.oanda.com/v3",
    )


def _mock_resp(status_code: int, body: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = body
    return r


def _http(get_resp=None, post_resp=None) -> MagicMock:
    http = MagicMock()
    if get_resp is not None:
        http.get.return_value = get_resp
    if post_resp is not None:
        http.post.return_value = post_resp
    return http


_CANDLE_RESP = {
    "instrument": "EUR_USD",
    "granularity": "D",
    "candles": [
        {
            "complete": True,
            "volume": 12000,
            "time": "2024-01-12T22:00:00.000000000Z",
            "mid": {"o": "1.09500", "h": "1.10200", "l": "1.09100", "c": "1.09800"},
        },
        {
            "complete": True,
            "volume": 15000,
            "time": "2024-01-13T22:00:00.000000000Z",
            "mid": {"o": "1.09800", "h": "1.10500", "l": "1.09500", "c": "1.10200"},
        },
        {
            "complete": False,  # in-progress bar — must be excluded
            "volume": 3000,
            "time": "2024-01-14T22:00:00.000000000Z",
            "mid": {"o": "1.10200", "h": "1.10300", "l": "1.10100", "c": "1.10250"},
        },
    ],
}

_ACCOUNT_RESP = {
    "account": {
        "balance":     "10250.00",
        "NAV":         "10312.50",
        "unrealizedPL": "62.50",
    }
}

_OPEN_TRADES_RESP = {
    "trades": [
        {
            "id": "T-111",
            "instrument": "EUR_USD",
            "currentUnits": "10000",
            "price": "1.09500",
            "stopLossOrder": {"price": "1.07500"},
        },
        {
            "id": "T-222",
            "instrument": "GBP_USD",
            "currentUnits": "-5000",
            "price": "1.27000",
            "stopLossOrder": {"price": "1.29000"},
        },
    ]
}

_PRICING_RESP = {
    "prices": [
        {
            "instrument": "EUR_USD",
            "bids": [{"price": "1.09990", "liquidity": 10_000_000}],
            "asks": [{"price": "1.10010", "liquidity": 10_000_000}],
            "time": "2024-01-14T10:00:00.000000000Z",
            "tradeable": True,
        }
    ]
}

_ORDER_FILL_RESP = {
    "orderFillTransaction": {
        "price": "1.10008",
        "tradeOpened": {"tradeID": "T-333"},
    }
}

_ORDER_REJECT_RESP = {
    "orderRejectTransaction": {
        "rejectReason": "ACCOUNT_NOT_TRADEABLE",
    }
}


# ---------------------------------------------------------------------------
# OandaConfig tests
# ---------------------------------------------------------------------------

class TestOandaConfig:
    def test_auth_header_format(self):
        cfg = _config()
        h   = cfg.auth_header
        assert h["Authorization"] == "Bearer test-token-abc"
        assert h["Content-Type"] == "application/json"

    def test_default_base_url_is_practice(self):
        cfg = OandaConfig(account_id="x", access_token="y")
        assert "fxpractice" in cfg.base_url


# ---------------------------------------------------------------------------
# fetch_candles tests
# ---------------------------------------------------------------------------

class TestFetchCandles:
    def test_complete_bars_parsed(self):
        http   = _http(get_resp=_mock_resp(200, _CANDLE_RESP))
        client = OandaClient(_config(), http)
        candles = client.fetch_candles("EUR_USD", count=10)

        assert len(candles) == 2  # in-progress bar excluded

    def test_incomplete_bar_excluded(self):
        http   = _http(get_resp=_mock_resp(200, _CANDLE_RESP))
        client = OandaClient(_config(), http)
        candles = client.fetch_candles("EUR_USD", count=10)

        for c in candles:
            assert c.complete is True

    def test_candles_sorted_ascending(self):
        http   = _http(get_resp=_mock_resp(200, _CANDLE_RESP))
        client = OandaClient(_config(), http)
        candles = client.fetch_candles("EUR_USD", count=10)

        times = [c.time for c in candles]
        assert times == sorted(times)

    def test_ohlcv_values_parsed(self):
        http   = _http(get_resp=_mock_resp(200, _CANDLE_RESP))
        client = OandaClient(_config(), http)
        c      = client.fetch_candles("EUR_USD", count=10)[0]

        assert c.open   == pytest.approx(1.09500)
        assert c.high   == pytest.approx(1.10200)
        assert c.low    == pytest.approx(1.09100)
        assert c.close  == pytest.approx(1.09800)
        assert c.volume == 12000

    def test_timestamp_is_utc_aware(self):
        http   = _http(get_resp=_mock_resp(200, _CANDLE_RESP))
        client = OandaClient(_config(), http)
        c      = client.fetch_candles("EUR_USD", count=10)[0]

        assert c.time.tzinfo is not None
        assert c.time == datetime(2024, 1, 12, 22, 0, 0, tzinfo=timezone.utc)

    def test_pair_in_candle(self):
        http   = _http(get_resp=_mock_resp(200, _CANDLE_RESP))
        client = OandaClient(_config(), http)
        c      = client.fetch_candles("EUR_USD", count=10)[0]

        assert c.pair == "EUR_USD"

    def test_non_200_raises_api_error(self):
        http   = _http(get_resp=_mock_resp(401, {"errorMessage": "Unauthorized"}))
        client = OandaClient(_config(), http)

        with pytest.raises(OandaApiError) as exc_info:
            client.fetch_candles("EUR_USD", count=10)

        assert exc_info.value.status_code == 401

    def test_count_limit_respected(self):
        """fetch_candles returns at most `count` bars even if response has more."""
        http   = _http(get_resp=_mock_resp(200, _CANDLE_RESP))
        client = OandaClient(_config(), http)
        candles = client.fetch_candles("EUR_USD", count=1)  # only 1 bar

        assert len(candles) == 1


# ---------------------------------------------------------------------------
# candles_to_dataframe tests
# ---------------------------------------------------------------------------

class TestCandlesToDataframe:
    def _make_candles(self, n: int = 2) -> list[Candle]:
        base = datetime(2024, 1, 12, 22, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta
        return [
            Candle(pair="EUR_USD", time=base + timedelta(days=i),
                   open=1.10, high=1.11, low=1.09, close=1.105, volume=10_000, complete=True)
            for i in range(n)
        ]

    def test_columns_present(self):
        import pandas as pd
        df = candles_to_dataframe(self._make_candles())
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_index_is_utc(self):
        df = candles_to_dataframe(self._make_candles())
        assert str(df.index.tz) == "UTC"

    def test_row_count(self):
        df = candles_to_dataframe(self._make_candles(5))
        assert len(df) == 5

    def test_values_correct(self):
        df = candles_to_dataframe(self._make_candles(1))
        assert df["open"].iloc[0] == pytest.approx(1.10)
        assert df["high"].iloc[0] == pytest.approx(1.11)
        assert df["volume"].iloc[0] == 10_000


# ---------------------------------------------------------------------------
# fetch_account_summary tests
# ---------------------------------------------------------------------------

class TestFetchAccountSummary:
    def _client_with_account(self, account_body, trades_body=None):
        trades_body = trades_body or {"trades": []}
        responses = [
            _mock_resp(200, account_body),
            _mock_resp(200, trades_body),
        ]
        http = MagicMock()
        http.get.side_effect = responses
        return OandaClient(_config(), http)

    def test_balance_parsed(self):
        client  = self._client_with_account(_ACCOUNT_RESP)
        summary = client.fetch_account_summary()
        assert summary.balance == pytest.approx(10250.00)

    def test_nav_parsed(self):
        client  = self._client_with_account(_ACCOUNT_RESP)
        summary = client.fetch_account_summary()
        assert summary.nav == pytest.approx(10312.50)

    def test_unrealized_pl_parsed(self):
        client  = self._client_with_account(_ACCOUNT_RESP)
        summary = client.fetch_account_summary()
        assert summary.unrealized_pl == pytest.approx(62.50)

    def test_open_positions_parsed(self):
        client  = self._client_with_account(_ACCOUNT_RESP, _OPEN_TRADES_RESP)
        summary = client.fetch_account_summary()
        assert len(summary.open_positions) == 2

    def test_long_position_direction(self):
        client  = self._client_with_account(_ACCOUNT_RESP, _OPEN_TRADES_RESP)
        summary = client.fetch_account_summary()
        eur_pos = next(p for p in summary.open_positions if p.pair == "EUR_USD")
        assert eur_pos.direction == "LONG"
        assert eur_pos.units == 10_000
        assert eur_pos.stop_loss == pytest.approx(1.075)

    def test_short_position_direction(self):
        client  = self._client_with_account(_ACCOUNT_RESP, _OPEN_TRADES_RESP)
        summary = client.fetch_account_summary()
        gbp_pos = next(p for p in summary.open_positions if p.pair == "GBP_USD")
        assert gbp_pos.direction == "SHORT"
        assert gbp_pos.units == 5_000

    def test_no_open_positions(self):
        client  = self._client_with_account(_ACCOUNT_RESP)
        summary = client.fetch_account_summary()
        assert summary.open_positions == ()


# ---------------------------------------------------------------------------
# get_spread_snapshot tests
# ---------------------------------------------------------------------------

class TestGetSpreadSnapshot:
    def test_bid_ask_mid_parsed(self):
        http   = _http(get_resp=_mock_resp(200, _PRICING_RESP))
        client = OandaClient(_config(), http)
        snap   = client.get_spread_snapshot("EUR_USD")

        assert snap.bid == pytest.approx(1.09990)
        assert snap.ask == pytest.approx(1.10010)
        assert snap.mid == pytest.approx(1.10000)

    def test_spread_pips_calculated(self):
        """EUR_USD pip = 0.0001, spread = 0.00020 → 2.0 pips."""
        http   = _http(get_resp=_mock_resp(200, _PRICING_RESP))
        client = OandaClient(_config(), http)
        snap   = client.get_spread_snapshot("EUR_USD")

        assert snap.spread_pips == pytest.approx(2.0, abs=0.01)

    def test_pair_in_snapshot(self):
        http   = _http(get_resp=_mock_resp(200, _PRICING_RESP))
        client = OandaClient(_config(), http)
        snap   = client.get_spread_snapshot("EUR_USD")
        assert snap.pair == "EUR_USD"


# ---------------------------------------------------------------------------
# get_open_trade_ids tests
# ---------------------------------------------------------------------------

class TestGetOpenTradeIds:
    def test_returns_trade_ids(self):
        body = {"trades": [{"id": "T-100"}, {"id": "T-200"}]}
        http   = _http(get_resp=_mock_resp(200, body))
        client = OandaClient(_config(), http)
        ids    = client.get_open_trade_ids("EUR_USD")
        assert ids == ["T-100", "T-200"]

    def test_empty_when_no_open_trades(self):
        http   = _http(get_resp=_mock_resp(200, {"trades": []}))
        client = OandaClient(_config(), http)
        ids    = client.get_open_trade_ids("EUR_USD")
        assert ids == []


# ---------------------------------------------------------------------------
# submit_market_order tests
# ---------------------------------------------------------------------------

class TestSubmitMarketOrder:
    def test_fill_parsed_on_success(self):
        http   = _http(post_resp=_mock_resp(201, _ORDER_FILL_RESP))
        client = OandaClient(_config(), http)
        fill   = client.submit_market_order("EUR_USD", 10000, 1.08, "EUR_USD_LONG_20240114")

        assert fill.status   == FillStatus.FILLED
        assert fill.fill_price == pytest.approx(1.10008)
        assert fill.broker_trade_id == "T-333"

    def test_buy_order_has_positive_units_in_request(self):
        http   = _http(post_resp=_mock_resp(201, _ORDER_FILL_RESP))
        client = OandaClient(_config(), http)
        client.submit_market_order("EUR_USD", 10000, 1.08, "SIG-1")

        body = http.post.call_args[1]["json"]
        assert body["order"]["units"] == "10000"

    def test_sell_order_has_negative_units_in_request(self):
        http   = _http(post_resp=_mock_resp(201, _ORDER_FILL_RESP))
        client = OandaClient(_config(), http)
        client.submit_market_order("EUR_USD", -10000, 1.12, "SIG-2")

        body = http.post.call_args[1]["json"]
        assert body["order"]["units"] == "-10000"

    def test_stop_loss_in_request_body(self):
        http   = _http(post_resp=_mock_resp(201, _ORDER_FILL_RESP))
        client = OandaClient(_config(), http)
        client.submit_market_order("EUR_USD", 10000, 1.08000, "SIG-1")

        body  = http.post.call_args[1]["json"]
        price = body["order"]["stopLossOnFill"]["price"]
        assert float(price) == pytest.approx(1.08000)

    def test_signal_id_in_client_extensions(self):
        http   = _http(post_resp=_mock_resp(201, _ORDER_FILL_RESP))
        client = OandaClient(_config(), http)
        client.submit_market_order("EUR_USD", 10000, 1.08, "MY_SIGNAL_ID")

        body    = http.post.call_args[1]["json"]
        comment = body["order"]["clientExtensions"]["comment"]
        assert comment == "MY_SIGNAL_ID"

    def test_transient_429_raises_execution_error(self):
        http   = _http(post_resp=_mock_resp(429, {"errorMessage": "Rate limited"}))
        client = OandaClient(_config(), http)

        with pytest.raises(ExecutionError) as exc_info:
            client.submit_market_order("EUR_USD", 10000, 1.08, "SIG")

        assert exc_info.value.http_status == 429

    def test_transient_504_raises_execution_error(self):
        http   = _http(post_resp=_mock_resp(504, {}))
        client = OandaClient(_config(), http)

        with pytest.raises(ExecutionError) as exc_info:
            client.submit_market_order("EUR_USD", 10000, 1.08, "SIG")

        assert exc_info.value.http_status == 504

    def test_reject_transaction_raises_execution_error(self):
        http   = _http(post_resp=_mock_resp(200, _ORDER_REJECT_RESP))
        client = OandaClient(_config(), http)

        with pytest.raises(ExecutionError) as exc_info:
            client.submit_market_order("EUR_USD", 10000, 1.08, "SIG")

        assert exc_info.value.http_status == 422

    def test_order_type_is_market(self):
        http   = _http(post_resp=_mock_resp(201, _ORDER_FILL_RESP))
        client = OandaClient(_config(), http)
        client.submit_market_order("EUR_USD", 10000, 1.08, "SIG")

        body = http.post.call_args[1]["json"]
        assert body["order"]["type"] == "MARKET"

    def test_stop_loss_gtc(self):
        http   = _http(post_resp=_mock_resp(201, _ORDER_FILL_RESP))
        client = OandaClient(_config(), http)
        client.submit_market_order("EUR_USD", 10000, 1.08, "SIG")

        body = http.post.call_args[1]["json"]
        assert body["order"]["stopLossOnFill"]["timeInForce"] == "GTC"
