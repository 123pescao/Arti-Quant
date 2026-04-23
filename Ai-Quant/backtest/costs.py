"""
Cost modeling: spread, execution slippage, and overnight swap.

All costs are expressed in pips. Conversions to currency P&L happen
in the engine using pip_value, which is pair- and lot-size-specific.

Spread scenarios:
  NORMAL    — baseline spread estimated at 22:10 UTC
  STRESS_15 — 1.5× baseline (moderate liquidity stress)
  STRESS_2  — 2.0× baseline (thin market / news aftermath)

Slippage model:
  Market order at 22:10 UTC (low liquidity window).
  Base slippage drawn from a conservative fixed estimate per pair.
  Stress scenarios amplify this proportionally.

Swap model:
  Conservative fixed daily pip rates per pair per direction.
  Derived from approximate retail OANDA rates (2023–2024 average).
  Both long and short are modeled independently because retail brokers
  charge a spread on the swap rate itself — both sides are often negative.
  These values MUST be revisited quarterly as central bank rates evolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class SpreadScenario(str, Enum):
    NORMAL    = "normal"
    STRESS_15 = "stress_1.5x"
    STRESS_2  = "stress_2x"


_SCENARIO_MULTIPLIER: dict[SpreadScenario, float] = {
    SpreadScenario.NORMAL:    1.0,
    SpreadScenario.STRESS_15: 1.5,
    SpreadScenario.STRESS_2:  2.0,
}

# Baseline half-spreads in pips at 22:10 UTC (post-NY close, pre-Tokyo open).
# These are conservative estimates — actual OANDA practice spreads measured
# in the 22:00–23:00 UTC window are typically 1.2×–2× London-session spreads.
# Half-spread is what the trader pays on each side (entry or exit).
_BASE_HALF_SPREAD_PIPS: dict[str, float] = {
    "EUR_USD": 0.7,
    "GBP_USD": 1.0,
    "USD_JPY": 0.6,
    "AUD_USD": 0.9,
    "USD_CAD": 1.0,
}

# Base execution slippage in pips (one-way, at market order fill).
# This is additional adverse price movement beyond the spread,
# caused by thin order book at 22:10 UTC.
_BASE_SLIPPAGE_PIPS: dict[str, float] = {
    "EUR_USD": 0.4,
    "GBP_USD": 0.6,
    "USD_JPY": 0.4,
    "AUD_USD": 0.5,
    "USD_CAD": 0.5,
}

# Daily swap rates in pips per 100,000 units (1 standard lot).
# Negative = cost to holder. Positive = credit to holder.
# These approximate retail OANDA rates (2023–2024 mid-period).
# Long USD_JPY and short EUR_USD historically near-zero or slightly positive
# but retail fees erode this — we model conservatively.
#
# WARNING: Swap rates change with every central bank policy decision.
# These values are approximations for backtesting only.
_DAILY_SWAP_PIPS: dict[str, dict[Literal["LONG", "SHORT"], float]] = {
    "EUR_USD": {"LONG": -0.52, "SHORT": -0.18},
    "GBP_USD": {"LONG": -0.48, "SHORT": -0.21},
    "USD_JPY": {"LONG":  0.62, "SHORT": -1.10},  # positive carry long USD
    "AUD_USD": {"LONG": -0.38, "SHORT": -0.22},
    "USD_CAD": {"LONG":  0.28, "SHORT": -0.70},
}

_PIP_SIZE: dict[str, float] = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,
    "AUD_USD": 0.0001,
    "USD_CAD": 0.0001,
}


@dataclass(frozen=True)
class CostConfig:
    """
    Immutable cost configuration for a single backtest run.
    Override any per-pair default by populating the override dicts.
    """
    scenario:               SpreadScenario = SpreadScenario.NORMAL
    half_spread_overrides:  dict[str, float] = field(default_factory=dict)
    slippage_overrides:     dict[str, float] = field(default_factory=dict)
    swap_overrides:         dict[str, dict[str, float]] = field(default_factory=dict)

    def half_spread(self, pair: str) -> float:
        base = self.half_spread_overrides.get(pair, _BASE_HALF_SPREAD_PIPS[pair])
        return base * _SCENARIO_MULTIPLIER[self.scenario]

    def slippage(self, pair: str) -> float:
        base = self.slippage_overrides.get(pair, _BASE_SLIPPAGE_PIPS[pair])
        return base * _SCENARIO_MULTIPLIER[self.scenario]

    def daily_swap_pips(self, pair: str, direction: Literal["LONG", "SHORT"]) -> float:
        """
        Returns pip cost per day for the given pair and direction.
        Negative = daily cost, positive = daily credit.
        Scale this by (units / 100_000) to get actual pip cost for position size.
        """
        pair_swaps = self.swap_overrides.get(pair, _DAILY_SWAP_PIPS[pair])
        return pair_swaps[direction]

    def pip_size(self, pair: str) -> float:
        return _PIP_SIZE[pair]

    def entry_cost_pips(self, pair: str) -> float:
        """Total one-way cost at entry: half-spread + slippage."""
        return self.half_spread(pair) + self.slippage(pair)

    def exit_cost_pips(self, pair: str) -> float:
        """Total one-way cost at exit: half-spread + slippage."""
        return self.half_spread(pair) + self.slippage(pair)

    def apply_entry_price(
        self,
        pair:      str,
        raw_price: float,
        direction: Literal["LONG", "SHORT"],
    ) -> float:
        """
        Return the realistic fill price at entry.
        LONG:  price moves against us (we pay more)
        SHORT: price moves against us (we receive less)
        """
        cost = self.entry_cost_pips(pair) * self.pip_size(pair)
        if direction == "LONG":
            return raw_price + cost
        return raw_price - cost

    def apply_exit_price(
        self,
        pair:      str,
        raw_price: float,
        direction: Literal["LONG", "SHORT"],
    ) -> float:
        """
        Return the realistic fill price at exit.
        LONG exit:  we sell — receive less than market
        SHORT exit: we buy  — pay more than market
        """
        cost = self.exit_cost_pips(pair) * self.pip_size(pair)
        if direction == "LONG":
            return raw_price - cost
        return raw_price + cost

    def apply_stop_fill(
        self,
        pair:      str,
        stop_price: float,
        bar_open:   float,
        direction: Literal["LONG", "SHORT"],
    ) -> float:
        """
        Stop-loss fill price. Two cases:
        1. No gap: fill at stop_price (with slippage only — spread already in stop distance)
        2. Gap through stop: fill at bar_open (worse case, gapped beyond stop)

        Slippage is added adversely in both cases.
        """
        pip = self.pip_size(pair)
        slip = self.slippage(pair) * pip

        if direction == "LONG":
            if bar_open <= stop_price:
                # Gapped below stop — fill at open (already worse)
                return bar_open - slip
            return stop_price - slip
        else:
            if bar_open >= stop_price:
                # Gapped above stop — fill at open
                return bar_open + slip
            return stop_price + slip


def all_scenarios() -> list[CostConfig]:
    """Return one CostConfig per SpreadScenario for sensitivity testing."""
    return [CostConfig(scenario=s) for s in SpreadScenario]


def pips_to_price(pips: float, pair: str) -> float:
    return pips * _PIP_SIZE[pair]


def price_to_pips(price_diff: float, pair: str) -> float:
    return price_diff / _PIP_SIZE[pair]
