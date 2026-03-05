import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modules import sweep


class TestSweep(unittest.TestCase):

    def test_sma_regime_sweep(self):
        dates = pd.date_range("2020-01-01", periods=200)
        close = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
        df = pd.DataFrame({"Close": close})
        df["Daily_Return"] = df["Close"].pct_change()

        out = sweep.run_sma_regime_sweep(df, [20, 50], 12, 0.75, False)
        self.assertFalse(out.empty)
        self.assertIn("Sharpe", out.columns)
        self.assertIn("CAGR", out.columns)


if __name__ == "__main__":
    unittest.main()
