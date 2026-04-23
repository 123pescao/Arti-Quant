"""
Telegram alert dispatcher.

Sends structured messages to a Telegram chat via the Bot API.
Errors in delivery are logged but never re-raised — a failed alert must not
crash the orchestrator or prevent state from being persisted.

Alerter protocol: any class implementing send(text) satisfies the interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# ---------------------------------------------------------------------------
# Protocol (allows mock injection in tests and alternative implementations)
# ---------------------------------------------------------------------------

class Alerter(Protocol):
    """Minimal alerting interface required by the orchestrator."""

    def send(self, text: str) -> None:
        """Send a plain-text message. Must not raise on failure."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TelegramConfig:
    """
    Telegram Bot API credentials.

    bot_token: Token from @BotFather (e.g. "123456:ABC-...")
    chat_id:   Target chat or channel ID (e.g. "-1001234567890")
    """
    bot_token: str
    chat_id:   str


# ---------------------------------------------------------------------------
# HTTP session protocol (injectable for testing)
# ---------------------------------------------------------------------------

class _HttpSession(Protocol):
    def post(self, url: str, json: dict = None, timeout: float = None) -> "_Response":
        ...


class _Response(Protocol):
    status_code: int

    def json(self) -> dict:
        ...


# ---------------------------------------------------------------------------
# TelegramAlerter
# ---------------------------------------------------------------------------

class TelegramAlerter:
    """
    Sends messages to a Telegram chat.

    Delivery failures are caught and logged; they never propagate to the caller.
    The orchestrator must not halt because of a failed alert.
    """

    def __init__(self, config: TelegramConfig, http: _HttpSession) -> None:
        self._config = config
        self._http   = http
        self._url    = _TELEGRAM_API.format(token=config.bot_token)

    def send(self, text: str) -> None:
        """Send a plain-text message. Silently logs on failure."""
        try:
            resp = self._http.post(
                self._url,
                json={"chat_id": self._config.chat_id, "text": text},
                timeout=10.0,
            )
            if resp.status_code != 200:
                logger.error(
                    "Telegram delivery failed: HTTP %s — %s",
                    resp.status_code, resp.json(),
                )
        except Exception as exc:
            logger.error("Telegram delivery error (suppressed): %s", exc)
