import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modules import factors


class TestFactors(unittest.TestCase):

    def test_factor_betas(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=120)
        f1 = np.random.normal(0, 0.01, size=len(dates))
        f2 = np.random.normal(0, 0.01, size=len(dates))
        factor_returns = pd.DataFrame({"F1": f1, "F2": f2}, index=dates)
        noise = np.random.normal(0, 0.001, size=len(dates))
        returns = 2.0 * f1 + 0.5 * f2 + noise
        returns = pd.Series(returns, index=dates)

        betas = factors.compute_factor_betas(returns, factor_returns, window=60)
        self.assertFalse(betas.empty)
        last = betas.iloc[-1]
        self.assertAlmostEqual(last["F1"], 2.0, delta=0.3)
        self.assertAlmostEqual(last["F2"], 0.5, delta=0.3)


if __name__ == "__main__":
    unittest.main()
