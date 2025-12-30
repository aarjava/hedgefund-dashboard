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
        # Momentum column name depends on column name
        col_name = 'Momentum_12M_1M'
        self.assertIn(col_name, result.columns)
        
        # Momentum should be positive for uptrend
        # Note: Depending on lag, it might be NaN at start, but valid at end
        self.assertTrue(result[col_name].iloc[-1] > 0)

    def test_volatility_regime(self):
        # Create a df with varying volatility
        dates = pd.date_range('2020-01-01', periods=100)
        df = pd.DataFrame({'Vol_21d': np.random.rand(100)}, index=dates)
        
        # Force some high and low values
        df.iloc[0:10, 0] = 0.01 # Low
        df.iloc[90:100, 0] = 1.0 # High
        
        res = signals.detect_volatility_regime(df, 'Vol_21d', 0.8, 0.2)
        
        self.assertIn('Vol_Regime', res.columns)
        # Check that we have High, Low, and Normal labels
        unique_regimes = res['Vol_Regime'].unique()
        self.assertIn('High', unique_regimes)
        self.assertIn('Low', unique_regimes)

if __name__ == '__main__':
    unittest.main()
