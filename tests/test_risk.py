import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modules import risk


class TestRisk(unittest.TestCase):

    def test_var_cvar(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.00])
        out = risk.compute_var_cvar(returns, level=0.95)
        expected_var = np.quantile(returns, 0.05)
        self.assertAlmostEqual(out["VaR"], expected_var)
        self.assertTrue(out["CVaR"] <= out["VaR"])

    def test_beta(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])
        bench = pd.Series([0.01, 0.02, 0.03, 0.04])
        beta = risk.compute_beta(returns, bench)
        self.assertAlmostEqual(beta, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
