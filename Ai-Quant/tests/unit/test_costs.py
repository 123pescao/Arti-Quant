"""
Unit tests for cost modeling.
"""

import pytest
from backtest.costs import CostConfig, SpreadScenario, price_to_pips, pips_to_price


class TestSpreadScenarios:
    def test_normal_spread(self, cost_config_normal):
        """Normal scenario should use 1x multiplier."""
        assert cost_config_normal.scenario == SpreadScenario.NORMAL
        assert cost_config_normal.half_spread("EUR_USD") == 0.7
        assert cost_config_normal.half_spread("GBP_USD") == 1.0

    def test_stress_1_5x(self):
        """1.5x scenario multiplies spreads."""
        config = CostConfig(scenario=SpreadScenario.STRESS_15)
        assert config.half_spread("EUR_USD") == pytest.approx(0.7 * 1.5)

    def test_stress_2x(self, cost_config_stress):
        """2x scenario doubles spreads."""
        assert cost_config_stress.half_spread("EUR_USD") == pytest.approx(0.7 * 2.0)
        assert cost_config_stress.half_spread("GBP_USD") == pytest.approx(1.0 * 2.0)


class TestSlippageAndSwap:
    def test_daily_swap_pips(self, cost_config_normal):
        """Swap rates should be retrievable per pair and direction."""
        eur_long_swap = cost_config_normal.daily_swap_pips("EUR_USD", "LONG")
        assert eur_long_swap < 0  # Cost

        eur_short_swap = cost_config_normal.daily_swap_pips("EUR_USD", "SHORT")
        assert eur_short_swap < 0  # Cost (retail)

        jpy_long_swap = cost_config_normal.daily_swap_pips("USD_JPY", "LONG")
        assert jpy_long_swap > 0  # Credit

    def test_slippage_pips(self, cost_config_normal):
        """Slippage should be positive (cost to trader)."""
        assert cost_config_normal.slippage("EUR_USD") > 0
        assert cost_config_normal.slippage("GBP_USD") > 0


class TestFillPrices:
    def test_entry_price_long(self, cost_config_normal):
        """LONG entry should move price against us (higher)."""
        raw_price = 1.1000
        fill = cost_config_normal.apply_entry_price("EUR_USD", raw_price, "LONG")
        assert fill > raw_price

    def test_entry_price_short(self, cost_config_normal):
        """SHORT entry should move price against us (lower)."""
        raw_price = 1.1000
        fill = cost_config_normal.apply_entry_price("EUR_USD", raw_price, "SHORT")
        assert fill < raw_price

    def test_exit_price_long(self, cost_config_normal):
        """LONG exit (selling) should move against us (lower)."""
        raw_price = 1.1050
        fill = cost_config_normal.apply_exit_price("EUR_USD", raw_price, "LONG")
        assert fill < raw_price

    def test_exit_price_short(self, cost_config_normal):
        """SHORT exit (buying) should move against us (higher)."""
        raw_price = 1.0950
        fill = cost_config_normal.apply_exit_price("EUR_USD", raw_price, "SHORT")
        assert fill > raw_price

    def test_stop_fill_no_gap(self, cost_config_normal):
        """Stop fill without gap should be at stop price minus slippage."""
        stop_price = 1.0950
        bar_open = 1.0980  # Opened above stop (no gap)
        fill = cost_config_normal.apply_stop_fill("EUR_USD", stop_price, bar_open, "LONG")
        # Should fill at or below stop due to slippage
        assert fill <= stop_price

    def test_stop_fill_gap_down(self, cost_config_normal):
        """Stop fill with gap should fill at open (worse case)."""
        stop_price = 1.0950
        bar_open = 1.0900  # Opened below stop (gap down)
        fill = cost_config_normal.apply_stop_fill("EUR_USD", stop_price, bar_open, "LONG")
        # Should fill at open or worse (gapped through)
        assert fill <= bar_open


class TestPipConversion:
    def test_pips_to_price(self):
        """Convert pips to price difference."""
        assert pips_to_price(1.0, "EUR_USD") == pytest.approx(0.0001)
        assert pips_to_price(100.0, "EUR_USD") == pytest.approx(0.01)
        assert pips_to_price(1.0, "USD_JPY") == pytest.approx(0.01)

    def test_price_to_pips(self):
        """Convert price difference to pips."""
        assert price_to_pips(0.0001, "EUR_USD") == pytest.approx(1.0)
        assert price_to_pips(0.01, "EUR_USD") == pytest.approx(100.0)
        assert price_to_pips(0.01, "USD_JPY") == pytest.approx(1.0)

    def test_roundtrip(self):
        """Roundtrip should be consistent."""
        orig_pips = 50.0
        price_diff = pips_to_price(orig_pips, "EUR_USD")
        back_to_pips = price_to_pips(price_diff, "EUR_USD")
        assert back_to_pips == pytest.approx(orig_pips)
