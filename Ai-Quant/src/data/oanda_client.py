"""
OANDA v20 REST API client.

Implements the OandaClient Protocol required by src/execution/executor.py and provides
additional methods for candle ingestion and account state fetching used by the orchestrator.

Scope (v1):
  - D1 candle fetching (historical bars, complete bars only)
  - Account summary (balance, NAV, open positions)
  - Current pricing / bid-ask spread snapshot
  - Open trade lookup (for executor idempotency)
  - Market order submission with stop-loss on fill

No live streaming. No intraday bars. No WebSocket.

OANDA v20 REST API:
  Live:     https://api-fxtrade.oanda.com/v3
  Practice: https://api-fxpractice.oanda.com/v3
  Auth:     Authorization: Bearer {token}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Protocol
import pandas as pd

from execution.models import Fill, FillStatus, Order, OrderSide, SpreadSnapshot
from risk.models import OpenPosition
from .models import AccountSummary, Candle

logger = logging.getLogger(__name__)

# Pip sizes — must match risk/engine.py and execution/executor.py
_PIP_SIZE: dict[str, float] = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,
    "AUD_USD": 0.0001,
    "USD_CAD": 0.0001,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OandaConfig:
    """
    OANDA v20 API credentials and routing.

    account_id:   OANDA account identifier (e.g. "001-001-12345678-001")
    access_token: Bearer token from OANDA developer portal
    base_url:     API base URL. Use practice URL for demo mode.
    """
    account_id:   str
    access_token: str
    base_url:     str = "https://api-fxpractice.oanda.com/v3"

    @property
    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# HTTP client protocol (injected for testability)
# ---------------------------------------------------------------------------

class _HttpSession(Protocol):
    """Minimal interface over requests.Session used by OandaClient."""

    def get(self, url: str, headers: dict, params: dict = None) -> "_Response":
        ...

    def post(self, url: str, headers: dict, json: dict = None) -> "_Response":
        ...


class _Response(Protocol):
    status_code: int

    def json(self) -> dict:
        ...


# ---------------------------------------------------------------------------
# OANDA client
# ---------------------------------------------------------------------------

class OandaClient:
    """
    Implements the executor.OandaClient protocol and provides data-layer methods.

    Usage:
        import requests
        config = OandaConfig(account_id=..., access_token=...)
        session = requests.Session()
        client = OandaClient(config, session)
        df = client.fetch_candles_df("EUR_USD", count=30)
    """

    def __init__(self, config: OandaConfig, http: _HttpSession) -> None:
        self._cfg  = config
        self._http = http

    # ------------------------------------------------------------------
    # Data layer methods (used by orchestrator)
    # ------------------------------------------------------------------

    def fetch_candles(self, pair: str, count: int, granularity: str = "D") -> list[Candle]:
        """
        Fetch the most recent `count` bars for `pair`.

        Filters to complete bars only. The OANDA response may include one
        in-progress bar at the end; it is discarded here.
        """
        url = f"{self._cfg.base_url}/instruments/{pair}/candles"
        params = {
            "granularity": granularity,
            "count": str(count + 1),  # +1 to account for possible in-progress bar
            "price": "M",             # mid prices (bid/ask midpoint)
        }
        resp = self._http.get(url, headers=self._cfg.auth_header, params=params)
        _check_response(resp, f"fetch candles {pair}")

        raw = resp.json()
        candles = []
        for bar in raw.get("candles", []):
            if not bar.get("complete", False):
                continue
            mid = bar["mid"]
            ts  = pd.Timestamp(bar["time"]).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            candles.append(Candle(
                pair     = pair,
                time     = ts,
                open     = float(mid["o"]),
                high     = float(mid["h"]),
                low      = float(mid["l"]),
                close    = float(mid["c"]),
                volume   = int(bar.get("volume", 0)),
                complete = True,
            ))

        candles.sort(key=lambda c: c.time)
        return candles[-count:]  # return at most `count` bars

    def fetch_candles_df(self, pair: str, count: int, granularity: str = "D") -> pd.DataFrame:
        """
        Fetch complete bars and return as a DataFrame compatible with Strategy.evaluate().

        Columns: open, high, low, close, volume
        Index:   UTC-aware DatetimeIndex (bar-open timestamps)
        """
        candles = self.fetch_candles(pair, count, granularity)
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return candles_to_dataframe(candles)

    def fetch_account_summary(self) -> AccountSummary:
        """Fetch current account balance, NAV, and open positions."""
        # Account summary
        url  = f"{self._cfg.base_url}/accounts/{self._cfg.account_id}/summary"
        resp = self._http.get(url, headers=self._cfg.auth_header)
        _check_response(resp, "fetch account summary")
        acct = resp.json()["account"]

        # Open trades (needed to build OpenPosition objects)
        open_positions = self._fetch_open_positions()

        return AccountSummary(
            balance        = float(acct["balance"]),
            nav            = float(acct["NAV"]),
            unrealized_pl  = float(acct.get("unrealizedPL", "0")),
            open_positions = tuple(open_positions),
        )

    # ------------------------------------------------------------------
    # executor.OandaClient protocol implementation
    # ------------------------------------------------------------------

    def get_spread_snapshot(self, pair: str) -> SpreadSnapshot:
        """Fetch current bid/ask for the pair and return a SpreadSnapshot."""
        url    = f"{self._cfg.base_url}/accounts/{self._cfg.account_id}/pricing"
        params = {"instruments": pair}
        resp   = self._http.get(url, headers=self._cfg.auth_header, params=params)
        _check_response(resp, f"get spread {pair}")

        price = resp.json()["prices"][0]
        bid   = float(price["bids"][0]["price"])
        ask   = float(price["asks"][0]["price"])
        mid   = (bid + ask) / 2.0
        pip   = _PIP_SIZE.get(pair, 0.0001)

        ts = pd.Timestamp(price["time"]).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return SpreadSnapshot(
            pair        = pair,
            bid         = bid,
            ask         = ask,
            mid         = mid,
            spread_pips = (ask - bid) / pip,
            sampled_at  = ts,
        )

    def get_open_trade_ids(self, pair: str) -> list[str]:
        """
        Return all open broker trade IDs for the pair, any direction.
        Used by the executor for pre-submission idempotency and post-timeout reconciliation.
        """
        url    = f"{self._cfg.base_url}/accounts/{self._cfg.account_id}/openTrades"
        params = {"instrument": pair}
        resp   = self._http.get(url, headers=self._cfg.auth_header, params=params)
        _check_response(resp, f"get open trades {pair}")

        return [t["id"] for t in resp.json().get("trades", [])]

    def submit_market_order(
        self,
        pair:      str,
        units:     int,
        stop_loss: float,
        signal_id: str,
    ) -> Fill:
        """
        Submit a market order with a broker-side stop-loss.

        units:     Positive = buy, negative = sell (OANDA convention).
        stop_loss: Stop price (GTC stop-loss order attached to the trade).
        signal_id: Stored as a client comment for audit trail.
        """
        from execution.executor import ExecutionError

        url  = f"{self._cfg.base_url}/accounts/{self._cfg.account_id}/orders"
        body = {
            "order": {
                "type":       "MARKET",
                "instrument": pair,
                "units":      str(units),
                "stopLossOnFill": {
                    "price":         f"{stop_loss:.5f}",
                    "timeInForce":   "GTC",
                },
                "clientExtensions": {
                    "comment": signal_id,
                },
            }
        }
        resp = self._http.post(url, headers=self._cfg.auth_header, json=body)

        if resp.status_code in {429, 503, 504}:
            raise ExecutionError(
                f"OANDA transient error: HTTP {resp.status_code}",
                http_status=resp.status_code,
            )
        if resp.status_code not in {200, 201}:
            raise ExecutionError(
                f"OANDA order rejected: HTTP {resp.status_code} — {resp.json()}",
                http_status=resp.status_code,
            )

        data = resp.json()

        if "orderRejectTransaction" in data:
            reject_reason = data["orderRejectTransaction"].get("rejectReason", "unknown")
            # Treat broker reject as a non-retryable ExecutionError
            raise ExecutionError(
                f"OANDA order rejected: {reject_reason}",
                http_status=422,
            )

        fill_tx   = data["orderFillTransaction"]
        fill_price = float(fill_tx["price"])
        trade_id  = fill_tx.get("tradeOpened", {}).get("tradeID")

        now = datetime.now(tz=timezone.utc)
        side = OrderSide.BUY if units > 0 else OrderSide.SELL
        order = Order(
            pair       = pair,
            side       = side,
            units      = abs(units),
            stop_loss  = stop_loss,
            signal_id  = signal_id,
            created_at = now,
        )
        return Fill(
            order           = order,
            fill_price      = fill_price,
            broker_trade_id = trade_id,
            filled_at       = now,
            status          = FillStatus.FILLED,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_open_positions(self) -> list[OpenPosition]:
        """Fetch and parse all open trades into OpenPosition objects."""
        url  = f"{self._cfg.base_url}/accounts/{self._cfg.account_id}/openTrades"
        resp = self._http.get(url, headers=self._cfg.auth_header)
        _check_response(resp, "fetch open trades")

        positions = []
        for trade in resp.json().get("trades", []):
            raw_units = int(trade["currentUnits"])
            direction = "LONG" if raw_units > 0 else "SHORT"
            stop_loss_order = trade.get("stopLossOrder")
            stop_loss = float(stop_loss_order["price"]) if stop_loss_order else 0.0
            positions.append(OpenPosition(
                pair        = trade["instrument"],
                direction   = direction,
                units       = abs(raw_units),
                entry_price = float(trade["price"]),
                stop_loss   = stop_loss,
            ))
        return positions


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
    """
    Convert a list of Candle objects to a DataFrame for Strategy.evaluate().

    Index:   UTC-aware DatetimeIndex (bar-open timestamps, ascending)
    Columns: open, high, low, close, volume
    """
    rows = [
        {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
        for c in candles
    ]
    index = pd.DatetimeIndex([c.time for c in candles], tz="UTC")
    return pd.DataFrame(rows, index=index)


def _check_response(resp: _Response, context: str) -> None:
    """Raise OandaApiError for non-2xx responses that are not handled by the caller."""
    if resp.status_code not in {200, 201}:
        raise OandaApiError(
            f"OANDA API error [{context}]: HTTP {resp.status_code} — {resp.json()}",
            status_code=resp.status_code,
        )


class OandaApiError(Exception):
    """Non-retryable OANDA API error (HTTP 4xx other than 429)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code
