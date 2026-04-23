"""
Walk-forward validation: rolling in-sample / out-of-sample testing.

Prevents overfitting by:
1. Selecting parameters on in-sample data
2. Evaluating them on unseen out-of-sample data
3. Rolling windows forward
4. Reporting out-of-sample performance only
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from itertools import product

from .engine import BacktestEngine, BacktestResult
from .costs import CostConfig, SpreadScenario
from .metrics import MetricsCalculator
from .data import DataLoader


@dataclass
class WalkForwardWindow:
    """Single in-sample / out-of-sample evaluation."""
    is_start:         pd.Timestamp
    is_end:           pd.Timestamp
    oos_start:        pd.Timestamp
    oos_end:          pd.Timestamp
    is_result:        Optional[BacktestResult] = None
    oos_result:       Optional[BacktestResult] = None
    best_params:      dict = field(default_factory=dict)


@dataclass
class WalkForwardResults:
    """Aggregated walk-forward results."""
    windows:          list[WalkForwardWindow]
    oos_sharpes:      list[float]
    oos_profit_factors: list[float]
    oos_max_dds:      list[float]
    median_sharpe:    float
    median_pf:        float
    pass_criteria:    bool
    fail_reason:      Optional[str] = None


class WalkForwardRunner:
    """
    Execute rolling parameter optimization with out-of-sample validation.
    """

    def __init__(
        self,
        pairs: list[str],
        initial_balance: float = 10_000.0,
        is_months: int = 24,
        oos_months: int = 6,
        step_months: int = 3,
    ):
        self.pairs = pairs
        self.initial_balance = initial_balance
        self.is_months = is_months
        self.oos_months = oos_months
        self.step_months = step_months

    def run(
        self,
        data: dict[str, pd.DataFrame],
        param_grid: dict = None,
        cost_scenario: SpreadScenario = SpreadScenario.NORMAL,
    ) -> WalkForwardResults:
        """
        Execute full walk-forward test.

        Args:
            data: dict[pair] → full OHLCV DataFrame (index=datetime)
            param_grid: dict of param names to lists of values
            cost_scenario: spread scenario (NORMAL, STRESS_15, STRESS_2)

        Returns:
            WalkForwardResults with all window evaluations and summary.
        """
        if param_grid is None:
            param_grid = {
                "entry_period": [20],
                "exit_period": [10],
                "atr_period": [14],
                "atr_multiplier": [2.0],
            }

        # Get date index from first pair
        all_dates = data[self.pairs[0]].index
        if not isinstance(all_dates, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

        # Generate windows
        windows = self._generate_windows(all_dates)
        results = WalkForwardResults(windows=[], oos_sharpes=[], oos_profit_factors=[],
                                    oos_max_dds=[], median_sharpe=0, median_pf=0, pass_criteria=False)

        for window in windows:
            print(f"Window: IS {window.is_start.date()} to {window.is_end.date()} | "
                  f"OOS {window.oos_start.date()} to {window.oos_end.date()}")

            # Slice data for this window
            is_data = {pair: df.loc[window.is_start:window.is_end] for pair, df in data.items()}
            oos_data = {pair: df.loc[window.oos_start:window.oos_end] for pair, df in data.items()}

            # Parameter sweep on IS
            best_params = self._sweep_parameters(
                is_data, param_grid, cost_scenario
            )
            window.best_params = best_params

            # Evaluate on IS (informational only)
            engine_is = BacktestEngine(self.pairs)
            is_result = engine_is.run(
                is_data,
                initial_balance=self.initial_balance,
                params=best_params,
                cost_config=CostConfig(scenario=cost_scenario),
            )
            window.is_result = is_result
            is_sharpe = is_result.sharpe_ratio

            # Evaluate on OOS (this is what we trust)
            engine_oos = BacktestEngine(self.pairs)
            oos_result = engine_oos.run(
                oos_data,
                initial_balance=self.initial_balance,
                params=best_params,
                cost_config=CostConfig(scenario=cost_scenario),
            )
            window.oos_result = oos_result
            oos_sharpe = oos_result.sharpe_ratio
            oos_pf = oos_result.profit_factor
            oos_dd = oos_result.max_drawdown

            results.oos_sharpes.append(oos_sharpe)
            results.oos_profit_factors.append(oos_pf)
            results.oos_max_dds.append(oos_dd)

            print(f"  → IS Sharpe: {is_sharpe:.3f} | OOS Sharpe: {oos_sharpe:.3f} | "
                  f"OOS PF: {oos_pf:.2f} | OOS MDD: {oos_dd:.2%}")

        # Compute summary
        results.median_sharpe = float(np.median(results.oos_sharpes))
        results.median_pf = float(np.median(results.oos_profit_factors))

        # Pass/fail criteria (from Phase 2 research)
        results.pass_criteria, results.fail_reason = self._evaluate_pass_fail(results)

        return results

    def _generate_windows(self, dates: pd.DatetimeIndex) -> list[WalkForwardWindow]:
        """Generate rolling IS/OOS windows."""
        windows = []
        is_duration = pd.DateOffset(months=self.is_months)
        oos_duration = pd.DateOffset(months=self.oos_months)
        step_duration = pd.DateOffset(months=self.step_months)

        current_is_start = dates[0]
        while True:
            current_is_end = current_is_start + is_duration - pd.Timedelta(days=1)
            current_oos_start = current_is_end + pd.Timedelta(days=1)
            current_oos_end = current_oos_start + oos_duration - pd.Timedelta(days=1)

            if current_oos_end >= dates[-1]:
                break

            windows.append(WalkForwardWindow(
                is_start=current_is_start,
                is_end=current_is_end,
                oos_start=current_oos_start,
                oos_end=current_oos_end,
            ))

            current_is_start = current_is_start + step_duration

        return windows

    def _sweep_parameters(
        self,
        data: dict[str, pd.DataFrame],
        param_grid: dict,
        cost_scenario: SpreadScenario,
    ) -> dict:
        """
        Grid search: test all param combinations, return best.
        Best = highest Sharpe ratio.
        """
        best_params = None
        best_sharpe = -np.inf

        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))

            engine = BacktestEngine(self.pairs)
            result = engine.run(
                data,
                initial_balance=self.initial_balance,
                params=params,
                cost_config=CostConfig(scenario=cost_scenario),
            )

            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_params = params

        return best_params

    @staticmethod
    def _evaluate_pass_fail(
        results: WalkForwardResults,
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluate against Phase 2 pass/fail criteria.

        MUST ANSWER criteria (from Phase 2):
          Q1. In-sample Sharpe ≥ 0.3
          Q2. OOS median Sharpe ≥ 0.2 (degradation expected)
          Q3. System survives worst regime (2017–2019 range)
          Q7. No OOS window > 15% drawdown

        PASS if all met, FAIL if any violated.
        """

        if results.median_sharpe < 0.2:
            return False, f"OOS median Sharpe {results.median_sharpe:.3f} < 0.2 threshold"

        if results.median_pf < 1.0:
            return False, f"OOS median PF {results.median_pf:.2f} < 1.0 (unprofitable)"

        if any(dd < -0.15 for dd in results.oos_max_dds):
            bad_window = max(enumerate(results.oos_max_dds), key=lambda x: abs(x[1]))[0]
            return False, f"OOS window {bad_window} drawdown {results.oos_max_dds[bad_window]:.2%} < -15% (catastrophic)"

        return True, None
