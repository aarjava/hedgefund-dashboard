import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modules import regime_analysis


class TestRegimeAnalysis(unittest.TestCase):

    def test_transition_matrix(self):
        regimes = pd.Series(["Low", "Normal", "High", "Normal", "High"])
        tm = regime_analysis.compute_transition_matrix(regimes)
        self.assertFalse(tm.empty)
        self.assertIn("Low", tm.index)
        self.assertIn("High", tm.columns)
        row_sums = tm.sum(axis=1)
        for val in row_sums:
            self.assertAlmostEqual(val, 1.0, places=6)

    def test_transition_stats(self):
        regimes = pd.Series(["Low", "Normal", "High", "Normal", "High"])
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        stats = regime_analysis.compute_transition_stats(returns, regimes)
        self.assertFalse(stats.empty)
        self.assertIn("Normal→High", stats.index)
        self.assertIn("Sharpe", stats.columns)

    def test_bootstrap_regime_diff(self):
        regimes = pd.Series(["Normal"] * 50 + ["High"] * 50)
        returns = pd.Series([0.001] * 50 + [0.005] * 50)
        out = regime_analysis.bootstrap_regime_diff(returns, regimes, metric="Mean", n_boot=100)
        self.assertFalse(np.isnan(out["diff"]))
        self.assertGreaterEqual(out["p_value"], 0.0)
        self.assertLessEqual(out["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
