import unittest
import pandas as pd
import numpy as np
from src.modules import signals

class TestSignals(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy dataframe
        dates = pd.date_range(start='2020-01-01', periods=300)
        # Create a simple upward trend
        prices = np.linspace(100, 200, 300)
        self.df = pd.DataFrame({'Close': prices}, index=dates)
        
    def test_sma_calculation(self):
        result = signals.add_technical_indicators(self.df, sma_window=50, mom_window=12)
        self.assertIn('SMA_50', result.columns)
        # SMA should not be nan at the end
        self.assertFalse(np.isnan(result['SMA_50'].iloc[-1]))
        
        # Check logic: In a perfect linear uptrend, Price > SMA
        self.assertTrue(result['Close'].iloc[-1] > result['SMA_50'].iloc[-1])

    def test_momentum_calculation(self):
        result = signals.add_technical_indicators(self.df, sma_window=50, mom_window=12)
        # Momentum column name depends on window
        col_name = 'Momentum_12M_1M'
        self.assertIn(col_name, result.columns)
        
        # Momentum should be positive for uptrend
        # Note: Depending on lag, it might be NaN at start, but valid at end
        self.assertTrue(result[col_name].iloc[-1] > 0)

if __name__ == '__main__':
    unittest.main()
