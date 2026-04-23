"""
Unit tests for src/alerting/telegram.py.

Critical properties tested:
  1. Message body contains chat_id and text
  2. HTTP POST is made to the correct Telegram API endpoint
  3. Delivery failure (non-200) is logged but not raised
  4. Network exception is caught and not re-raised
  5. TelegramAlerter satisfies the Alerter protocol (has send method)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from unittest.mock import MagicMock, patch

from alerting.telegram import TelegramAlerter, TelegramConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _config() -> TelegramConfig:
    return TelegramConfig(bot_token="123456:ABC", chat_id="-100987654")


def _mock_http(status_code: int = 200) -> MagicMock:
    http = MagicMock()
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"ok": True} if status_code == 200 else {"ok": False, "description": "err"}
    http.post.return_value = resp
    return http


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTelegramAlerter:
    def test_post_made_to_correct_url(self):
        http    = _mock_http()
        alerter = TelegramAlerter(_config(), http)
        alerter.send("hello world")

        url = http.post.call_args[0][0]
        assert "123456:ABC" in url
        assert "sendMessage" in url

    def test_chat_id_in_request_body(self):
        http    = _mock_http()
        alerter = TelegramAlerter(_config(), http)
        alerter.send("test message")

        body = http.post.call_args[1]["json"]
        assert body["chat_id"] == "-100987654"

    def test_text_in_request_body(self):
        http    = _mock_http()
        alerter = TelegramAlerter(_config(), http)
        alerter.send("my test alert")

        body = http.post.call_args[1]["json"]
        assert body["text"] == "my test alert"

    def test_non_200_does_not_raise(self):
        """Telegram delivery failure must be silently absorbed."""
        http    = _mock_http(status_code=400)
        alerter = TelegramAlerter(_config(), http)

        alerter.send("test")  # must not raise

    def test_network_exception_does_not_raise(self):
        """Any network exception must be caught and logged, not re-raised."""
        http = MagicMock()
        http.post.side_effect = ConnectionError("network down")
        alerter = TelegramAlerter(_config(), http)

        alerter.send("test")  # must not raise

    def test_timeout_passed_to_post(self):
        """Request must have a finite timeout to avoid hanging the orchestrator."""
        http    = _mock_http()
        alerter = TelegramAlerter(_config(), http)
        alerter.send("x")

        kwargs = http.post.call_args[1]
        assert "timeout" in kwargs
        assert kwargs["timeout"] > 0

    def test_protocol_compatibility(self):
        """TelegramAlerter must be usable as Alerter (duck-type check)."""
        http    = _mock_http()
        alerter = TelegramAlerter(_config(), http)
        assert callable(alerter.send)
