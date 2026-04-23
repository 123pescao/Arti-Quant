"""
Phase 3 Backtest Runner — Entry point for full backtesting suite.

Run with:
  python -m backtest.run_backtest

Produces:
  - Trade log (CSV)
  - Equity curve (PNG)
  - Drawdown chart (PNG)
  - Monthly heatmap (PNG)
  - Metrics summary (TXT)
  - Per-pair statistics (CSV)
  - Walk-forward validation (console output)

All outputs saved to outputs/ directory.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backtest.data import load_all_pairs
from backtest.engine import BacktestEngine
from backtest.costs import CostConfig, SpreadScenario, all_scenarios
from backtest.walk_forward import WalkForwardRunner
from backtest.reports import ReportGenerator
from backtest.metrics import MetricsCalculator


def main():
    """Run full Phase 3 backtest suite."""
    print("\n" + "=" * 70)
    print("BULL-RUN PHASE 3: BACKTEST ENGINE")
    print("=" * 70 + "\n")

    # Setup
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]
    initial_balance = 10_000.0
    output_base = Path(__file__).parent.parent / "outputs"

    # Load data (synthetic for Phase 3 demo)
    print("[1/4] Loading data...")
    try:
        data = load_all_pairs(
            pairs,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        print(f"  ✓ Loaded {len(data)} pairs")
        for pair, df in data.items():
            print(f"    {pair}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return

    # Strategy params
    strategy_params = {
        "entry_period": 20,
        "exit_period": 10,
        "atr_period": 14,
        "atr_multiplier": 2.0,
    }

    # --- Test 1: Single backtest with normal spreads ---
    print("\n[2/4] Running single backtest (NORMAL spread scenario)...")
    engine = BacktestEngine(
        pairs=pairs,
        risk_per_trade_pct=0.0025,
        max_positions=2,
        max_usd_exposure=2,
    )
    result_normal = engine.run(
        data,
        initial_balance=initial_balance,
        params=strategy_params,
        cost_config=CostConfig(scenario=SpreadScenario.NORMAL),
    )
    print(f"  ✓ Completed: {result_normal.trade_count} trades")
    print(f"    Sharpe: {result_normal.sharpe_ratio:.3f}")
    print(f"    Max DD: {result_normal.max_drawdown:.2%}")
    print(f"    Profit Factor: {result_normal.profit_factor:.2f}")

    # --- Test 2: Stress scenario ---
    print("\n[3/4] Running stress test (2x SPREAD scenario)...")
    engine_stress = BacktestEngine(
        pairs=pairs,
        risk_per_trade_pct=0.0025,
        max_positions=2,
        max_usd_exposure=2,
    )
    result_stress = engine_stress.run(
        data,
        initial_balance=initial_balance,
        params=strategy_params,
        cost_config=CostConfig(scenario=SpreadScenario.STRESS_2),
    )
    print(f"  ✓ Completed: {result_stress.trade_count} trades")
    print(f"    Sharpe: {result_stress.sharpe_ratio:.3f}")
    print(f"    Max DD: {result_stress.max_drawdown:.2%}")
    print(f"    Profit Factor: {result_stress.profit_factor:.2f}")

    # --- Test 3: Walk-forward validation ---
    print("\n[4/4] Running walk-forward validation...")
    try:
        wf_runner = WalkForwardRunner(
            pairs=pairs,
            initial_balance=initial_balance,
            is_months=24,
            oos_months=6,
            step_months=6,  # Reduced for faster demo
        )
        param_grid = {
            "entry_period": [20],
            "exit_period": [10],
            "atr_period": [14],
            "atr_multiplier": [2.0],
        }
        wf_results = wf_runner.run(
            data,
            param_grid=param_grid,
            cost_scenario=SpreadScenario.NORMAL,
        )
        print(f"  ✓ Walk-forward complete: {len(wf_results.windows)} windows")
        print(f"    Median OOS Sharpe: {wf_results.median_sharpe:.3f}")
        print(f"    Median OOS PF: {wf_results.median_pf:.2f}")
        print(f"    Pass criteria: {'✓ PASS' if wf_results.pass_criteria else '✗ FAIL'}")
        if wf_results.fail_reason:
            print(f"    Reason: {wf_results.fail_reason}")
    except Exception as e:
        print(f"  ✗ Walk-forward error: {e}")
        wf_results = None

    # Generate reports
    print("\n[Reports] Generating output files...")
    reporter = ReportGenerator(output_base)

    print("\n  Normal spread scenario:")
    paths = reporter.generate_all(result_normal, prefix="backtest_normal")
    for name, path in paths.items():
        print(f"    {name}: {path}")

    print("\n  Stressed spread scenario:")
    paths_stress = reporter.generate_all(result_stress, prefix="backtest_stress")
    for name, path in paths_stress.items():
        print(f"    {name}: {path}")

    # Summary
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print("\nKEY FINDINGS (REQUIREMENT: BE SKEPTICAL):\n")

    print("1. IS THIS PROFITABLE?")
    if result_normal.profit_factor > 1.1:
        print(f"   → Possibly. PF={result_normal.profit_factor:.2f} (threshold: 1.1)")
        print("   → CAVEAT: Synthetic data, not real market execution")
    else:
        print(f"   → Unlikely. PF={result_normal.profit_factor:.2f} (threshold: 1.1)")

    print("\n2. SPREAD IMPACT")
    pf_impact = ((result_normal.profit_factor - result_stress.profit_factor)
                 / (result_normal.profit_factor + 0.001) * 100)
    print(f"   → 2x spreads reduce PF by {pf_impact:.1f}%")
    print(f"   → Normal: {result_normal.profit_factor:.2f} → Stress: {result_stress.profit_factor:.2f}")

    print("\n3. DRAWDOWN RISK")
    print(f"   → Max DD: {result_normal.max_drawdown:.2%}")
    print(f"   → Circuit breaker at -15% → margin: {15 - abs(result_normal.max_drawdown*100):.1f}%")

    print("\n4. TRADE STATISTICS")
    wr = result_normal.win_count / max(1, result_normal.trade_count)
    print(f"   → Win rate: {wr*100:.1f}% (target: ~38%)")
    print(f"   → Avg winner: {result_normal.avg_win_pips:.1f} pips")
    print(f"   → Avg loser: {result_normal.avg_loss_pips:.1f} pips")

    print("\n5. WALK-FORWARD VALIDATION")
    if wf_results:
        print(f"   → OOS median Sharpe: {wf_results.median_sharpe:.3f}")
        print(f"   → Degradation (IS→OOS): {(wf_results.median_sharpe - result_normal.sharpe_ratio) / max(0.1, result_normal.sharpe_ratio) * 100:.0f}%")
        print(f"   → Pass/fail: {'PASS' if wf_results.pass_criteria else 'FAIL'}")
    else:
        print("   → Walk-forward did not complete (see above)")

    print("\n6. RECOMMENDATION FOR PHASE 4")
    if wf_results and wf_results.pass_criteria and result_normal.profit_factor > 1.0:
        print("   → PROCEED to Phase 4 (Strategy Implementation)")
        print("   → Caveat: Still demo-only, validate with real data before live")
    elif result_normal.profit_factor > 1.0:
        print("   → CONDITIONAL: PF > 1.0 but walk-forward concerns")
        print("   → Recommendation: Extend walk-forward window sizes for confidence")
    else:
        print("   → HOLD: Insufficient profitability in backtest")
        print("   → Recommendation: Review parameters or consider alternative strategies")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n✗ FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
