"""
Report generation: CSV exports, charts, sensitivity tables.
Produces fully auditable output for Phase 3 validation.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional

from .engine import BacktestResult, Trade
from .metrics import MetricsCalculator


class ReportGenerator:
    """Generate backtest output files and charts."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        result: BacktestResult,
        prefix: str = "backtest",
    ) -> dict[str, Path]:
        """Generate all reports. Returns dict[name] → file paths."""
        paths = {}
        paths["trade_log"] = self.export_trade_log(result, prefix)
        paths["equity_curve_chart"] = self.chart_equity_curve(result, prefix)
        paths["drawdown_chart"] = self.chart_drawdown(result, prefix)
        paths["monthly_heatmap"] = self.chart_monthly_heatmap(result, prefix)
        paths["metrics_summary"] = self.export_metrics_summary(result, prefix)
        return paths

    def export_trade_log(self, result: BacktestResult, prefix: str = "backtest") -> Path:
        """
        Export detailed trade log to CSV.
        Every trade with entry/exit prices, costs, PnL, swap.
        """
        trades_data = []
        for trade in result.trades:
            trades_data.append({
                "pair": trade.pair,
                "entry_date": trade.entry_date.isoformat(),
                "exit_date": trade.exit_date.isoformat(),
                "direction": trade.direction,
                "units": trade.units,
                "entry_price_raw": f"{trade.entry_price_raw:.5f}",
                "entry_price": f"{trade.entry_price:.5f}",
                "entry_cost_pips": f"{trade.entry_cost_pips:.2f}",
                "exit_price_raw": f"{trade.exit_price_raw:.5f}",
                "exit_price": f"{trade.exit_price:.5f}",
                "exit_cost_pips": f"{trade.exit_cost_pips:.2f}",
                "stop_loss": f"{trade.stop_loss:.5f}",
                "exit_reason": trade.exit_reason,
                "pnl_pips": f"{trade.pnl_pips:.2f}",
                "pnl_currency": f"{trade.pnl_currency:.2f}",
                "holding_days": trade.holding_days,
                "swap_paid": f"{trade.swap_paid:.2f}",
            })

        df = pd.DataFrame(trades_data)
        path = self.output_dir / f"{prefix}_trades.csv"
        df.to_csv(path, index=False)
        print(f"✓ Trade log: {path}")
        return path

    def chart_equity_curve(self, result: BacktestResult, prefix: str = "backtest") -> Path:
        """
        Plot equity curve over time.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(result.equity_curve.index, result.equity_curve.values, linewidth=2, label="Equity")
        ax.axhline(result.equity_curve.iloc[0], color="gray", linestyle="--", alpha=0.5, label="Starting Balance")
        ax.fill_between(result.equity_curve.index, result.equity_curve.iloc[0], result.equity_curve.values,
                       alpha=0.2, color="blue")

        ax.set_xlabel("Date")
        ax.set_ylabel("Account Equity (USD)")
        ax.set_title(f"Equity Curve\n{result.params}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()

        path = self.output_dir / f"{prefix}_equity_curve.png"
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"✓ Equity curve: {path}")
        return path

    def chart_drawdown(self, result: BacktestResult, prefix: str = "backtest") -> Path:
        """
        Plot underwater (drawdown) chart.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        cummax = result.equity_curve.cummax()
        drawdown = (result.equity_curve - cummax) / cummax * 100

        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red", label="Drawdown")
        ax.plot(drawdown.index, drawdown.values, linewidth=1.5, color="red")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(f"Drawdown Chart | Max DD: {result.max_drawdown:.2%}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()

        path = self.output_dir / f"{prefix}_drawdown.png"
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"✓ Drawdown chart: {path}")
        return path

    def chart_monthly_heatmap(self, result: BacktestResult, prefix: str = "backtest") -> Path:
        """
        Plot monthly returns as heatmap.
        """
        if len(result.daily_returns) < 2:
            # Not enough data
            return self.output_dir / f"{prefix}_monthly_heatmap.png"

        monthly = MetricsCalculator.monthly_returns(result.equity_curve)
        if len(monthly) < 2:
            # Not enough data
            return self.output_dir / f"{prefix}_monthly_heatmap.png"

        # Create heatmap data (years × months)
        monthly_idx = monthly.index
        years = monthly_idx.year.unique()
        months_range = range(1, 13)

        data = np.zeros((len(years), 12))
        data[:] = np.nan

        for i, year in enumerate(years):
            year_data = monthly[monthly_idx.year == year]
            for j in range(12):
                month_key = pd.Timestamp(year=year, month=j+1, day=1)
                if month_key in year_data.index:
                    data[i, j] = year_data[month_key] * 100  # Convert to %

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)

        ax.set_title("Monthly Returns (%)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Return %")

        # Add text annotations
        for i in range(len(years)):
            for j in range(12):
                if not np.isnan(data[i, j]):
                    text = ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center",
                                  color="black" if abs(data[i, j]) < 5 else "white", fontsize=8)

        plt.tight_layout()
        path = self.output_dir / f"{prefix}_monthly_heatmap.png"
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"✓ Monthly heatmap: {path}")
        return path

    def export_metrics_summary(self, result: BacktestResult, prefix: str = "backtest") -> Path:
        """
        Export summary metrics to text file.
        """
        output = []
        output.append("=" * 60)
        output.append("BACKTEST RESULTS SUMMARY")
        output.append("=" * 60)
        output.append("")

        output.append("STRATEGY PARAMETERS")
        output.append("-" * 60)
        for key, val in result.params.items():
            output.append(f"  {key}: {val}")
        output.append("")

        output.append("PERFORMANCE METRICS")
        output.append("-" * 60)
        output.append(f"  Total Trades:              {result.trade_count}")
        output.append(f"  Winning Trades:            {result.win_count} ({result.win_count/max(1, result.trade_count)*100:.1f}%)")
        output.append(f"  Losing Trades:             {result.loss_count} ({result.loss_count/max(1, result.trade_count)*100:.1f}%)")
        output.append(f"  Win Rate:                  {result.win_count/max(1, result.trade_count)*100:.1f}%")
        output.append(f"  Profit Factor:             {result.profit_factor:.2f}")
        output.append(f"  Average Win (pips):        {result.avg_win_pips:.2f}")
        output.append(f"  Average Loss (pips):       {result.avg_loss_pips:.2f}")
        output.append(f"  Payoff Ratio (W/L):        {result.avg_win_pips / max(0.01, abs(result.avg_loss_pips)):.2f}")
        output.append("")

        output.append("RISK METRICS")
        output.append("-" * 60)
        output.append(f"  Max Drawdown:              {result.max_drawdown:.2%}")
        output.append(f"  Sharpe Ratio:              {result.sharpe_ratio:.3f}")
        output.append(f"  Annual Return:             {MetricsCalculator.annual_return(result.equity_curve):.2%}")
        output.append("")

        output.append("COST ASSUMPTIONS")
        output.append("-" * 60)
        output.append(f"  Spread Scenario:           {result.cost_config.scenario.value}")
        output.append("")

        output.append("TRADE LOG AVAILABLE")
        output.append("-" * 60)
        output.append(f"  See: {prefix}_trades.csv")
        output.append("")

        text = "\n".join(output)
        path = self.output_dir / f"{prefix}_metrics.txt"
        with open(path, "w") as f:
            f.write(text)

        print(f"✓ Metrics summary: {path}")
        print(text)
        return path

    def export_sensitivity_table(
        self,
        results: dict,  # dict[param_combo] → BacktestResult
        param_names: list[str],
        metric: str = "sharpe_ratio",
    ) -> Path:
        """
        Export sensitivity analysis as CSV.
        Shows how metrics vary with parameter changes.
        """
        rows = []
        for params, result in results.items():
            row = dict(params)
            row[metric] = getattr(result, metric)
            rows.append(row)

        df = pd.DataFrame(rows)
        path = self.output_dir / f"sensitivity_{metric}.csv"
        df.to_csv(path, index=False)
        print(f"✓ Sensitivity table: {path}")
        return path

    def export_per_pair_stats(self, result: BacktestResult) -> Path:
        """
        Export per-pair breakdown.
        """
        if len(result.trades) == 0:
            return self.output_dir / "per_pair_stats.csv"

        trades_df = pd.DataFrame([
            {
                "pair": t.pair,
                "direction": t.direction,
                "pnl_pips": t.pnl_pips,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
            }
            for t in result.trades
        ])

        stats = MetricsCalculator.per_pair_stats(trades_df)
        rows = [{"pair": pair, **metrics} for pair, metrics in stats.items()]
        df = pd.DataFrame(rows)

        path = self.output_dir / "per_pair_stats.csv"
        df.to_csv(path, index=False)
        print(f"✓ Per-pair stats: {path}")
        return path
