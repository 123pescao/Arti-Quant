"""
Backtest simulation engine.

Event-driven loop that:
1. Checks stops/exits on open positions
2. Processes entry signals
3. Enforces position limits (max 2 concurrent, max 1 per pair, USD correlation)
4. Records trades with slippage tracking
5. Updates daily equity with swap costs
6. Produces trade log and equity curve

Key principle: explicit, verifiable, not black-box.
Every trade has documented entry/exit prices and costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from .costs import CostConfig, price_to_pips
from .strategy import compute_signals, DonchianSignals


@dataclass
class Position:
    """An open trade."""
    pair:              str
    direction:         Literal["LONG", "SHORT"]
    entry_price:       float          # actual fill price (includes costs)
    entry_price_raw:   float          # raw market price (before costs)
    stop_loss:         float          # absolute price
    units:             int            # positive
    entry_bar_idx:     int
    entry_date:        datetime
    cumulative_swap:   float = 0.0    # accumulated swap in currency


@dataclass
class Trade:
    """A closed trade record."""
    pair:              str
    direction:         Literal["LONG", "SHORT"]
    units:             int
    entry_date:        datetime
    entry_price_raw:   float          # reference market price at signal
    entry_price:       float          # actual fill with costs
    entry_cost_pips:   float
    exit_date:         datetime
    exit_price_raw:    float          # reference price at exit
    exit_price:        float          # actual fill with costs
    exit_cost_pips:    float
    stop_loss:         float
    exit_reason:       Literal["DONCHIAN_EXIT", "STOP_LOSS"]
    pnl_pips:          float          # net profit in pips (including costs)
    pnl_currency:      float          # profit in currency
    holding_days:      int
    swap_paid:         float          # cumulative swap cost


@dataclass
class BacktestResult:
    """Full backtest output."""
    trades:            list[Trade]
    equity_curve:      pd.Series      # date → balance
    positions_curve:   pd.Series      # date → unrealized P&L
    daily_returns:     pd.Series      # date → daily % return
    trade_count:       int
    win_count:         int
    loss_count:        int
    profit_factor:     float
    sharpe_ratio:      float
    max_drawdown:      float
    avg_win_pips:      float
    avg_loss_pips:     float
    params:            dict           # strategy params used
    cost_config:       CostConfig     # cost model used


class BacktestEngine:
    """
    Simulate trading strategy with realistic execution and costs.
    """

    def __init__(
        self,
        pairs: list[str],
        risk_per_trade_pct: float = 0.0025,
        max_positions: int = 2,
        max_usd_exposure: int = 2,
        max_units_per_trade: int = 100_000,
    ):
        self.pairs = pairs
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_positions = max_positions
        self.max_usd_exposure = max_usd_exposure
        self.max_units_per_trade = max_units_per_trade

        self.open_positions: dict[str, Position] = {}
        self.closed_trades: list[Trade] = []
        self.equity_history: list[tuple[datetime, float]] = []

    def run(
        self,
        data: dict[str, pd.DataFrame],
        initial_balance: float = 10_000.0,
        params: dict = None,
        cost_config: CostConfig = None,
    ) -> BacktestResult:
        """
        Run full backtest simulation.

        Args:
            data: dict[pair] → OHLCV DataFrame
            initial_balance: starting account balance in USD
            params: strategy params (entry_period, exit_period, atr_period, atr_multiplier)
            cost_config: cost model (spreads, slippage, swap)

        Returns:
            BacktestResult with full trade log and metrics.
        """
        if params is None:
            params = {
                "entry_period": 20,
                "exit_period": 10,
                "atr_period": 14,
                "atr_multiplier": 2.0,
            }
        if cost_config is None:
            cost_config = CostConfig()

        self._reset()

        # Compute signals for all pairs
        signals_all = {}
        for pair in self.pairs:
            sigs = compute_signals(data[pair], **params)
            signals_all[pair] = sigs

        # Get date index (use first pair)
        dates = data[self.pairs[0]].index

        balance = initial_balance
        self.equity_history.append((dates[0], balance))

        # Main simulation loop
        for bar_idx in range(len(dates)):
            date = dates[bar_idx]

            # --- 1. Apply daily swap to open positions ---
            for pair, pos in list(self.open_positions.items()):
                swap_pips = cost_config.daily_swap_pips(pair, pos.direction)
                swap_cost = swap_pips * cost_config.pip_size(pair) * (pos.units / 100_000)
                pos.cumulative_swap += swap_cost
                balance -= swap_cost

            # --- 2. Process stops and exits ---
            for pair, pos in list(self.open_positions.items()):
                bar_data = data[pair].iloc[bar_idx]
                sigs = signals_all[pair]

                # Check stop loss (intrabar)
                stop_hit = self._check_stop(pair, pos, bar_data, cost_config)
                if stop_hit:
                    exit_price, exit_price_raw, exit_cost = stop_hit
                    pnl = pos.units * (
                        (exit_price - pos.entry_price) if pos.direction == "LONG"
                        else (pos.entry_price - exit_price)
                    )
                    self._close_position(
                        pair,
                        pos,
                        exit_price,
                        exit_price_raw,
                        exit_cost,
                        date,
                        reason="STOP_LOSS",
                        pnl_currency=pnl,
                    )
                    balance += pnl
                    continue

                # Check Donchian exit (at close)
                exit_signal = (sigs.long_exit.iloc[bar_idx] if pos.direction == "LONG"
                              else sigs.short_exit.iloc[bar_idx])
                if exit_signal:
                    exit_price_raw = bar_data["close"]
                    exit_price = cost_config.apply_exit_price(pair, exit_price_raw, pos.direction)
                    exit_cost_pips = price_to_pips(
                        abs(exit_price - exit_price_raw), pair
                    )
                    pnl = pos.units * (
                        (exit_price - pos.entry_price) if pos.direction == "LONG"
                        else (pos.entry_price - exit_price)
                    )
                    self._close_position(
                        pair,
                        pos,
                        exit_price,
                        exit_price_raw,
                        exit_cost_pips,
                        date,
                        reason="DONCHIAN_EXIT",
                        pnl_currency=pnl,
                    )
                    balance += pnl

            # --- 3. Process entry signals ---
            usd_count = sum(1 for pair in self.open_positions if "USD" in pair)
            for pair in self.pairs:
                if pair in self.open_positions:
                    continue  # max 1 per pair
                if len(self.open_positions) >= self.max_positions:
                    break  # max total positions

                bar_data = data[pair].iloc[bar_idx]
                sigs = signals_all[pair]

                direction = None
                if sigs.long_entry.iloc[bar_idx]:
                    direction = "LONG"
                elif sigs.short_entry.iloc[bar_idx]:
                    direction = "SHORT"
                else:
                    continue

                # USD exposure check
                if "USD" in pair and usd_count >= self.max_usd_exposure:
                    continue

                # Entry price
                entry_price_raw = bar_data["open"]
                entry_price = cost_config.apply_entry_price(pair, entry_price_raw, direction)
                entry_cost_pips = price_to_pips(abs(entry_price - entry_price_raw), pair)

                # Stop loss
                stop_price = (sigs.long_stop.iloc[bar_idx] if direction == "LONG"
                             else sigs.short_stop.iloc[bar_idx])

                # Position sizing
                stop_distance_pips = price_to_pips(abs(stop_price - entry_price), pair)
                risk_amount = balance * self.risk_per_trade_pct
                pip_value = cost_config.pip_size(pair)  # e.g., 0.0001 for EUR_USD
                # For 1 lot (100,000 units), 1 pip = pip_value * 100,000 in the quote currency
                # We need to invert this: units = risk_amount / (pips × pip_value × lot_value)
                lot_value = 100_000
                units = int(risk_amount / (stop_distance_pips * pip_value * lot_value))
                units = min(units, self.max_units_per_trade)

                if units <= 0:
                    continue

                # Open position
                pos = Position(
                    pair=pair,
                    direction=direction,
                    entry_price=entry_price,
                    entry_price_raw=entry_price_raw,
                    stop_loss=stop_price,
                    units=units,
                    entry_bar_idx=bar_idx,
                    entry_date=date,
                )
                self.open_positions[pair] = pos

            # --- 4. Update equity ---
            unrealized = self._compute_unrealized(data, bar_idx, balance)
            self.equity_history.append((date, balance + unrealized))

        # Close any remaining open positions at market price on final day
        final_bar_idx = len(dates) - 1
        for pair, pos in list(self.open_positions.items()):
            exit_price_raw = data[pair].iloc[final_bar_idx]["close"]
            exit_price = cost_config.apply_exit_price(pair, exit_price_raw, pos.direction)
            exit_cost_pips = price_to_pips(abs(exit_price - exit_price_raw), pair)
            pnl = pos.units * (
                (exit_price - pos.entry_price) if pos.direction == "LONG"
                else (pos.entry_price - exit_price)
            )
            self._close_position(
                pair,
                pos,
                exit_price,
                exit_price_raw,
                exit_cost_pips,
                dates[final_bar_idx],
                reason="END_OF_BACKTEST",
                pnl_currency=pnl,
            )
            balance += pnl

        return self._build_result(params, cost_config)

    def _check_stop(
        self,
        pair: str,
        pos: Position,
        bar_data: pd.Series,
        cost_config: CostConfig,
    ) -> Optional[tuple[float, float, float]]:
        """
        Check if stop loss was hit.
        Returns (fill_price, raw_price, cost_pips) or None.
        raw_price: stop_loss price or bar open if gapped.
        fill_price: raw_price with slippage applied.
        """
        if pos.direction == "LONG":
            if bar_data["low"] <= pos.stop_loss:
                raw = bar_data["open"] if bar_data["open"] <= pos.stop_loss else pos.stop_loss
                fill = cost_config.apply_stop_fill(
                    pair, pos.stop_loss, bar_data["open"], "LONG"
                )
                return fill, raw, cost_config.slippage(pair)
        else:
            if bar_data["high"] >= pos.stop_loss:
                raw = bar_data["open"] if bar_data["open"] >= pos.stop_loss else pos.stop_loss
                fill = cost_config.apply_stop_fill(
                    pair, pos.stop_loss, bar_data["open"], "SHORT"
                )
                return fill, raw, cost_config.slippage(pair)
        return None

    def _close_position(
        self,
        pair: str,
        pos: Position,
        exit_price: float,
        exit_price_raw: float,
        exit_cost_pips: float,
        date: datetime,
        reason: str,
        pnl_currency: float,
    ) -> None:
        """
        Record closed trade.
        pnl_currency is pre-computed by caller as: units × price_diff (fill prices).
        pnl_pips is derived from fill prices and already net of all entry/exit costs.
        """
        holding_days = (date - pos.entry_date).days

        if pos.direction == "LONG":
            pnl_pips = price_to_pips(exit_price - pos.entry_price, pair)
        else:
            pnl_pips = price_to_pips(pos.entry_price - exit_price, pair)

        trade = Trade(
            pair=pair,
            direction=pos.direction,
            units=pos.units,
            entry_date=pos.entry_date,
            entry_price_raw=pos.entry_price_raw,
            entry_price=pos.entry_price,
            entry_cost_pips=price_to_pips(abs(pos.entry_price - pos.entry_price_raw), pair),
            exit_date=date,
            exit_price_raw=exit_price_raw,
            exit_price=exit_price,
            exit_cost_pips=exit_cost_pips,
            stop_loss=pos.stop_loss,
            exit_reason=reason,
            pnl_pips=pnl_pips,
            pnl_currency=pnl_currency,
            holding_days=holding_days,
            swap_paid=pos.cumulative_swap,
        )
        self.closed_trades.append(trade)
        del self.open_positions[pair]

    def _compute_unrealized(self, data: dict, bar_idx: int, balance: float) -> float:
        """Sum of unrealized P&L on all open positions (uses fill prices, consistent with balance)."""
        unrealized = 0.0
        for pair, pos in self.open_positions.items():
            current_price = data[pair].iloc[bar_idx]["close"]
            if pos.direction == "LONG":
                unrealized += (current_price - pos.entry_price) * pos.units
            else:
                unrealized += (pos.entry_price - current_price) * pos.units
        return unrealized

    def _build_result(self, params: dict, cost_config: CostConfig) -> BacktestResult:
        """Construct final result object with metrics."""
        equity_series = pd.Series(
            [eq for _, eq in self.equity_history],
            index=[dt for dt, _ in self.equity_history],
        )
        daily_returns = equity_series.pct_change().dropna()

        trade_pnls = [t.pnl_pips for t in self.closed_trades]
        wins = sum(1 for pnl in trade_pnls if pnl > 0)
        losses = sum(1 for pnl in trade_pnls if pnl < 0)

        if len(trade_pnls) > 0:
            profit_factor = (sum(p for p in trade_pnls if p > 0) or 0.001) / (
                -sum(p for p in trade_pnls if p < 0) or 0.001
            )
            avg_win = np.mean([p for p in trade_pnls if p > 0]) if wins > 0 else 0
            avg_loss = np.mean([p for p in trade_pnls if p < 0]) if losses > 0 else 0
        else:
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0

        # Max drawdown
        cummax = equity_series.cummax()
        dd = (equity_series - cummax) / cummax
        max_dd = dd.min()

        # Sharpe
        if len(daily_returns) > 0:
            std = daily_returns.std()
            sharpe = (daily_returns.mean() / std) * np.sqrt(252) if std >= 1e-10 else 0.0
        else:
            sharpe = 0.0

        return BacktestResult(
            trades=self.closed_trades,
            equity_curve=equity_series,
            positions_curve=pd.Series(dtype=float),
            daily_returns=daily_returns,
            trade_count=len(self.closed_trades),
            win_count=wins,
            loss_count=losses,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_win_pips=avg_win,
            avg_loss_pips=avg_loss,
            params=params,
            cost_config=cost_config,
        )

    def _reset(self) -> None:
        """Clear state for new run."""
        self.open_positions = {}
        self.closed_trades = []
        self.equity_history = []
