"""
Orchestrator configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_DEFAULT_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]

# Minimum complete bars required before the strategy can generate a signal.
# Must be >= DonchianStrategy.min_candles (23). Set to 30 to provide buffer.
_DEFAULT_CANDLE_COUNT = 30


@dataclass(frozen=True)
class OrchestratorConfig:
    """
    Configuration for a single orchestrator run.

    pairs:          FX pairs to evaluate each session.
    run_hour_utc:   UTC hour at which the orchestrator is authorised to run.
                    Default 22 = immediately after the D1 bar closes (NY convention).
    candle_count:   Number of complete D1 bars to fetch per pair.
    demo_mode:      If True, skip real order submission (log only). Unused in v1
                    — real demo-mode safety comes from using the practice API URL.
    """
    pairs:          tuple[str, ...]  = field(default_factory=lambda: tuple(_DEFAULT_PAIRS))
    run_hour_utc:   int              = 22
    candle_count:   int              = _DEFAULT_CANDLE_COUNT
    demo_mode:      bool             = True

    def __post_init__(self) -> None:
        if not 0 <= self.run_hour_utc <= 23:
            raise ValueError(f"run_hour_utc must be 0-23, got {self.run_hour_utc}")
        if self.candle_count < 23:
            raise ValueError(f"candle_count must be >= 23, got {self.candle_count}")
        if not self.pairs:
            raise ValueError("pairs must not be empty")
