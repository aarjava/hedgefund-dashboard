import unittest
import pandas as pd
import numpy as np
from src.modules import backtester

class TestBacktester(unittest.TestCase):
    
    def setUp(self):
        dates = pd.date_range(start='2020-01-01', periods=100)
        self.df = pd.DataFrame({
            'Close': np.linspace(100, 200, 100),
            'Daily_Return': np.full(100, 0.01), # Constant 1% return
            'Signal': np.full(100, 1) # Always Long
        }, index=dates)
        
    def test_run_backtest(self):
        # Test basic flow
        data = self.df.copy()
        # Ensure we have enough data for resampling if monthly
        res = backtester.run_backtest(data, 'Signal', cost_bps=0.0, rebalance_freq='D')
        
        self.assertIn('Strategy_Return', res.columns)
        self.assertIn('Equity_Strategy', res.columns)
        
        # Since return is positive and we are long, equity should grow
        self.assertTrue(res['Equity_Strategy'].iloc[-1] > 1.0)
        
    def test_metrics(self):
        equity = pd.Series([1.0, 1.1, 1.21], index=pd.date_range('2020-01-01', periods=3))
        metrics = backtester.calculate_perf_metrics(equity)
        
        self.assertIn('CAGR', metrics)
        self.assertIn('Sharpe', metrics)
        # CAGR calculation check
        # It's a very short period (3 days), so CAGR will be huge
        self.assertTrue(metrics['CAGR'] > 0)

if __name__ == '__main__':
    unittest.main()
