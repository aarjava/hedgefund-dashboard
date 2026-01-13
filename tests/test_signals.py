import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules import signals


class TestSignals(unittest.TestCase):
    
    def setUp(self):
        """Create test dataframes with known properties."""
        # Create a dummy dataframe with 300 days of upward trending data
        dates = pd.date_range(start='2020-01-01', periods=300)
        prices = np.linspace(100, 200, 300)
        self.df = pd.DataFrame({'Close': prices}, index=dates)
        
    def test_sma_calculation(self):
        """Test that SMA is calculated correctly."""
        result = signals.add_technical_indicators(self.df, sma_window=50, mom_window=12)
        
        self.assertIn('SMA_50', result.columns)
        # SMA should not be nan at the end
        self.assertFalse(np.isnan(result['SMA_50'].iloc[-1]))
        
        # Check logic: In a perfect linear uptrend, Price > SMA
        self.assertTrue(result['Close'].iloc[-1] > result['SMA_50'].iloc[-1])

    def test_sma_200_always_calculated(self):
        """Test that 200-day SMA is always calculated as benchmark."""
        result = signals.add_technical_indicators(self.df, sma_window=50)
        self.assertIn('SMA_200', result.columns)

    def test_momentum_calculation(self):
        """Test momentum calculation for various lookback windows."""
        result = signals.add_technical_indicators(self.df, sma_window=50, mom_window=12)
        col_name = 'Momentum_12M_1M'
        self.assertIn(col_name, result.columns)
        
        # Momentum should be positive for uptrend
        self.assertTrue(result[col_name].iloc[-1] > 0)

    def test_rsi_bounds(self):
        """Test that RSI stays within 0-100 bounds."""
        result = signals.add_technical_indicators(self.df, sma_window=50, mom_window=12)
        self.assertIn('RSI_14', result.columns)
        
        valid_rsi = result['RSI_14'].dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

    def test_volatility_regime_in_sample(self):
        """Test in-sample regime detection (full-sample quantiles)."""
        # Create a df with varying volatility
        dates = pd.date_range('2020-01-01', periods=100)
        df = pd.DataFrame({'Vol_21d': np.random.rand(100)}, index=dates)
        
        # Force some high and low values
        df.iloc[0:10, 0] = 0.01  # Low
        df.iloc[90:100, 0] = 1.0  # High
        
        res = signals.detect_volatility_regime(
            df, 'Vol_21d', 0.8, 0.2, use_expanding=False
        )
        
        self.assertIn('Vol_Regime', res.columns)
        # Check that we have High, Low, and Normal labels
        unique_regimes = res['Vol_Regime'].unique()
        self.assertIn('High', unique_regimes)
        self.assertIn('Low', unique_regimes)

    def test_volatility_regime_out_of_sample(self):
        """Test out-of-sample regime detection (expanding-window quantiles)."""
        dates = pd.date_range('2020-01-01', periods=100)
        df = pd.DataFrame({'Vol_21d': np.random.rand(100)}, index=dates)
        
        # Force some high and low values
        df.iloc[0:10, 0] = 0.01  # Low
        df.iloc[90:100, 0] = 1.0  # High
        
        res = signals.detect_volatility_regime(
            df, 'Vol_21d', 0.8, 0.2, use_expanding=True, min_periods=20
        )
        
        self.assertIn('Vol_Regime', res.columns)
        # Early periods should be 'Unknown' due to insufficient data
        self.assertIn('Unknown', res['Vol_Regime'].iloc[:20].values)

    def test_volatility_regime_oos_wrapper(self):
        """Test the convenience wrapper for out-of-sample regime detection."""
        dates = pd.date_range('2020-01-01', periods=100)
        df = pd.DataFrame({'Vol_21d': np.random.rand(100)}, index=dates)
        
        res = signals.detect_volatility_regime_oos(
            df, 'Vol_21d', min_periods=20
        )
        
        self.assertIn('Vol_Regime', res.columns)
        # Verify it uses expanding window (early periods should be 'Unknown')
        self.assertIn('Unknown', res['Vol_Regime'].iloc[:20].values)

    def test_empty_dataframe_handling(self):
        """Test that functions handle empty dataframes gracefully."""
        empty_df = pd.DataFrame()
        
        result = signals.add_technical_indicators(empty_df)
        self.assertTrue(result.empty)
        
        result = signals.detect_volatility_regime(empty_df)
        self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()

