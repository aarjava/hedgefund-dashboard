"""Tests for advanced signal generation module."""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modules import signals_advanced


class TestBollingerBands(unittest.TestCase):
    """Tests for Bollinger Bands calculation."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=100)
        # Random walk price series
        returns = np.random.randn(100) * 0.02
        prices = 100 * np.cumprod(1 + returns)
        self.df = pd.DataFrame({"Close": prices}, index=dates)

    def test_bollinger_bands_columns(self):
        """Test that all BB columns are created."""
        result = signals_advanced.calculate_bollinger_bands(self.df)

        self.assertIn("BB_Middle", result.columns)
        self.assertIn("BB_Upper", result.columns)
        self.assertIn("BB_Lower", result.columns)
        self.assertIn("BB_Width", result.columns)
        self.assertIn("BB_Position", result.columns)

    def test_bollinger_bands_order(self):
        """Test that upper > middle > lower."""
        result = signals_advanced.calculate_bollinger_bands(self.df)

        # After warmup period
        valid_data = result.iloc[25:]
        self.assertTrue((valid_data["BB_Upper"] > valid_data["BB_Middle"]).all())
        self.assertTrue((valid_data["BB_Middle"] > valid_data["BB_Lower"]).all())

    def test_bollinger_bands_width_positive(self):
        """Test that band width is always positive."""
        result = signals_advanced.calculate_bollinger_bands(self.df)
        valid_width = result["BB_Width"].dropna()
        self.assertTrue((valid_width > 0).all())


class TestMeanReversionSignal(unittest.TestCase):
    """Tests for mean reversion signal generation."""

    def setUp(self):
        """Create test data with RSI and BB columns."""
        dates = pd.date_range(start="2020-01-01", periods=100)
        self.df = pd.DataFrame(
            {
                "RSI_14": np.linspace(20, 80, 100),  # RSI from oversold to overbought
                "BB_Position": np.linspace(-1.5, 1.5, 100),  # BB from below to above
            },
            index=dates,
        )

    def test_signal_values(self):
        """Test that signals are in valid range."""
        signal = signals_advanced.generate_mean_reversion_signal(self.df)

        self.assertTrue(signal.isin([-1, 0, 1]).all())

    def test_oversold_buy_signal(self):
        """Test that oversold conditions generate buy signals."""
        # Create oversold data
        df = pd.DataFrame({"RSI_14": [25, 28, 29], "BB_Position": [-0.9, -0.85, -0.95]})

        signal = signals_advanced.generate_mean_reversion_signal(df, oversold=30, overbought=70)

        # Should have buy signals
        self.assertTrue((signal == 1).any())

    def test_overbought_sell_signal(self):
        """Test that overbought conditions generate sell signals."""
        df = pd.DataFrame({"RSI_14": [75, 78, 80], "BB_Position": [0.9, 0.85, 0.95]})

        signal = signals_advanced.generate_mean_reversion_signal(df, oversold=30, overbought=70)

        # Should have sell signals
        self.assertTrue((signal == -1).any())


class TestVolatilityBreakoutSignal(unittest.TestCase):
    """Tests for volatility breakout signal."""

    def setUp(self):
        """Create test data with volatility."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=200)
        self.df = pd.DataFrame(
            {
                "Close": np.cumprod(1 + np.random.randn(200) * 0.01) * 100,
                "Vol_21d": np.abs(np.random.randn(200) * 0.1) + 0.1,
            },
            index=dates,
        )

        # Spike volatility at end
        self.df.loc[self.df.index[-20:], "Vol_21d"] = 0.5

    def test_signal_generation(self):
        """Test that signal is generated."""
        signal = signals_advanced.generate_volatility_breakout_signal(self.df)

        self.assertEqual(len(signal), len(self.df))
        self.assertTrue(signal.isin([-1, 0, 1]).all())


class TestDualMomentumSignal(unittest.TestCase):
    """Tests for dual momentum signal."""

    def setUp(self):
        """Create test data with momentum."""
        dates = pd.date_range(start="2020-01-01", periods=100)
        self.df = pd.DataFrame(
            {"Momentum_12M_1M": np.linspace(-0.2, 0.3, 100)}, index=dates  # -20% to +30%
        )

    def test_positive_momentum_long(self):
        """Test that positive momentum generates long signal."""
        signal = signals_advanced.generate_dual_momentum_signal(self.df)

        # Positive momentum should have some long signals
        positive_mom_mask = self.df["Momentum_12M_1M"] > 0
        self.assertTrue((signal[positive_mom_mask] == 1).any())

    def test_negative_momentum_cash(self):
        """Test that negative momentum is cash."""
        signal = signals_advanced.generate_dual_momentum_signal(self.df)

        # Negative momentum should be cash (0)
        negative_mom_mask = self.df["Momentum_12M_1M"] < 0
        self.assertTrue((signal[negative_mom_mask] == 0).all())


class TestCompositeSignal(unittest.TestCase):
    """Tests for composite signal generation."""

    def setUp(self):
        """Create test signals."""
        dates = pd.date_range(start="2020-01-01", periods=10)
        self.df = pd.DataFrame(index=dates)

        self.signals = {
            "trend": pd.Series([1, 1, 1, 0, -1, -1, 1, 1, 0, 0], index=dates),
            "momentum": pd.Series([1, 1, 0, 0, 0, -1, 1, 0, 0, 1], index=dates),
        }

    def test_equal_weight_combination(self):
        """Test equal weight signal combination."""
        signal = signals_advanced.generate_composite_signal(self.df, self.signals, threshold=0.5)

        self.assertEqual(len(signal), len(self.df))
        self.assertTrue(signal.isin([-1, 0, 1]).all())

    def test_weighted_combination(self):
        """Test weighted signal combination."""
        weights = {"trend": 0.7, "momentum": 0.3}
        signal = signals_advanced.generate_composite_signal(
            self.df, self.signals, weights=weights, threshold=0.5
        )

        self.assertTrue(signal.isin([-1, 0, 1]).all())


class TestATR(unittest.TestCase):
    """Tests for ATR calculation."""

    def setUp(self):
        """Create OHLC test data."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=50)
        close = np.cumprod(1 + np.random.randn(50) * 0.01) * 100

        self.df = pd.DataFrame(
            {
                "Open": close * (1 + np.random.randn(50) * 0.005),
                "High": close * (1 + np.abs(np.random.randn(50) * 0.01)),
                "Low": close * (1 - np.abs(np.random.randn(50) * 0.01)),
                "Close": close,
            },
            index=dates,
        )

    def test_atr_positive(self):
        """Test that ATR is always positive."""
        atr = signals_advanced.calculate_atr(self.df)

        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0).all())

    def test_atr_window(self):
        """Test custom ATR window."""
        atr_14 = signals_advanced.calculate_atr(self.df, window=14)
        atr_7 = signals_advanced.calculate_atr(self.df, window=7)

        # Shorter window should have values earlier
        self.assertTrue(atr_7.first_valid_index() <= atr_14.first_valid_index())


class TestPositionSizing(unittest.TestCase):
    """Tests for position sizing."""

    def test_basic_position_size(self):
        """Test basic position sizing calculation."""
        shares, stop = signals_advanced.calculate_position_size(
            account_value=100000,
            risk_per_trade=0.01,  # 1% risk
            atr=2.0,
            atr_multiplier=2.0,
            price=50.0,
        )

        self.assertGreater(shares, 0)
        self.assertEqual(stop, 4.0)  # 2 * 2 ATR

    def test_zero_atr_handling(self):
        """Test handling of zero ATR."""
        shares, stop = signals_advanced.calculate_position_size(
            account_value=100000, risk_per_trade=0.01, atr=0, price=50.0
        )

        self.assertEqual(shares, 0)
        self.assertEqual(stop, 0.0)


if __name__ == "__main__":
    unittest.main()
