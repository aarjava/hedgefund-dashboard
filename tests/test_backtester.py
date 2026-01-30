import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modules import backtester


class TestBacktester(unittest.TestCase):

    def setUp(self):
        """Create test dataframes with known properties."""
        dates = pd.date_range(start="2020-01-01", periods=100)
        self.df = pd.DataFrame(
            {
                "Close": np.linspace(100, 200, 100),
                "Daily_Return": np.full(100, 0.01),  # Constant 1% return
                "Signal": np.full(100, 1),  # Always Long
            },
            index=dates,
        )

        # Create larger dataset for walk-forward tests
        dates_long = pd.date_range(start="2018-01-01", periods=1000)
        self.df_long = pd.DataFrame(
            {
                "Close": np.cumprod(1 + np.random.randn(1000) * 0.01) * 100,
                "Daily_Return": np.random.randn(1000) * 0.01,
                "Signal": np.where(np.random.rand(1000) > 0.5, 1, 0),
            },
            index=dates_long,
        )

    def test_run_backtest_daily(self):
        """Test basic daily rebalancing backtest."""
        data = self.df.copy()
        res = backtester.run_backtest(data, "Signal", cost_bps=0.0, rebalance_freq="D")

        self.assertIn("Strategy_Return", res.columns)
        self.assertIn("Equity_Strategy", res.columns)
        self.assertIn("DD_Strategy", res.columns)

        # Since return is positive and we are long, equity should grow
        self.assertTrue(res["Equity_Strategy"].iloc[-1] > 1.0)

    def test_run_backtest_weekly(self):
        """Test weekly rebalancing backtest."""
        data = self.df.copy()
        res = backtester.run_backtest(data, "Signal", cost_bps=0.0, rebalance_freq="W")

        self.assertIn("Equity_Strategy", res.columns)
        self.assertTrue(res["Equity_Strategy"].iloc[-1] > 1.0)

    def test_run_backtest_monthly(self):
        """Test monthly rebalancing backtest."""
        data = self.df.copy()
        res = backtester.run_backtest(data, "Signal", cost_bps=0.0, rebalance_freq="M")

        self.assertIn("Equity_Strategy", res.columns)

    def test_metrics_basic(self):
        """Test basic performance metrics calculation."""
        equity = pd.Series([1.0, 1.1, 1.21], index=pd.date_range("2020-01-01", periods=3))
        metrics = backtester.calculate_perf_metrics(equity)

        self.assertIn("CAGR", metrics)
        self.assertIn("Sharpe", metrics)
        self.assertIn("MaxDD", metrics)
        self.assertIn("WinRate", metrics)
        self.assertTrue(metrics["CAGR"] > 0)

    def test_metrics_with_bootstrap_ci(self):
        """Test performance metrics with bootstrap CI."""
        # Need more data for reliable CI
        dates = pd.date_range("2020-01-01", periods=100)
        returns = np.random.randn(100) * 0.01 + 0.001  # Slight positive drift
        equity = pd.Series((1 + returns).cumprod(), index=dates)

        metrics = backtester.calculate_perf_metrics(
            equity, include_bootstrap_ci=True, n_bootstrap=100
        )

        self.assertIn("Sharpe_CI_Lower", metrics)
        self.assertIn("Sharpe_CI_Upper", metrics)

        # CI bounds should exist and be ordered
        if not np.isnan(metrics["Sharpe_CI_Lower"]):
            self.assertLessEqual(metrics["Sharpe_CI_Lower"], metrics["Sharpe_CI_Upper"])

    def test_drawdown_duration(self):
        """Test drawdown duration calculation."""
        # Create equity curve with known drawdown
        equity = pd.Series([1.0, 1.1, 1.0, 0.9, 0.85, 0.9, 1.0, 1.1])

        max_dd, avg_dd = backtester.calculate_drawdown_duration(equity)

        self.assertGreater(max_dd, 0)
        self.assertGreater(avg_dd, 0)

    def test_bootstrap_sharpe_ci(self):
        """Test bootstrap Sharpe CI directly."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)

        lower, upper = backtester.bootstrap_sharpe_ci(
            returns, n_bootstrap=500, confidence_level=0.95, random_state=42
        )

        self.assertFalse(np.isnan(lower))
        self.assertFalse(np.isnan(upper))
        self.assertLess(lower, upper)

    def test_bootstrap_sharpe_ci_insufficient_data(self):
        """Test bootstrap CI with insufficient data."""
        returns = pd.Series([0.01, 0.02, 0.01])  # Only 3 points

        lower, upper = backtester.bootstrap_sharpe_ci(returns)

        # Should return NaN due to insufficient data
        self.assertTrue(np.isnan(lower))
        self.assertTrue(np.isnan(upper))

    def test_conditional_stats(self):
        """Test conditional statistics calculation."""
        df = pd.DataFrame(
            {
                "Strategy_Net_Return": np.random.randn(100) * 0.01,
                "Vol_Regime": ["High"] * 30 + ["Normal"] * 40 + ["Low"] * 30,
            }
        )

        stats = backtester.calculate_conditional_stats(df, "Strategy_Net_Return", "Vol_Regime")

        self.assertIn("High", stats.index)
        self.assertIn("Normal", stats.index)
        self.assertIn("Low", stats.index)
        self.assertIn("Sharpe", stats.columns)

    def test_walk_forward_backtest(self):
        """Test walk-forward validation."""
        # Use the longer dataset
        result = backtester.walk_forward_backtest(
            self.df_long,
            "Signal",
            train_months=12,
            test_months=3,
            cost_bps=0.001,
            rebalance_freq="M",
        )

        self.assertIn("summary", result)
        self.assertIn("periods", result)
        self.assertIn("n_periods", result)
        self.assertGreater(result["n_periods"], 0)

    def test_walk_forward_insufficient_data(self):
        """Test walk-forward with insufficient data."""
        result = backtester.walk_forward_backtest(
            self.df, "Signal", train_months=24, test_months=6  # Only 100 days
        )

        # Should return empty dict due to insufficient data
        self.assertEqual(result, {})

    def test_empty_dataframe_handling(self):
        """Test that functions handle empty dataframes gracefully."""
        empty_df = pd.DataFrame()

        result = backtester.run_backtest(empty_df, "Signal")
        self.assertTrue(result.empty)

        metrics = backtester.calculate_perf_metrics(pd.Series(dtype=float))
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
