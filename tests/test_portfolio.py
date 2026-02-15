import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modules import portfolio


class TestPortfolio(unittest.TestCase):

    def test_normalize_weights(self):
        tickers = ["A", "B"]
        weights = [2, 1]
        norm = portfolio.normalize_weights(tickers, weights)
        self.assertAlmostEqual(norm.sum(), 1.0)
        self.assertAlmostEqual(norm["A"], 2 / 3)
        self.assertAlmostEqual(norm["B"], 1 / 3)

    def test_compute_portfolio_returns(self):
        dates = pd.date_range("2020-01-01", periods=3)
        price_df = pd.DataFrame({"A": [100, 110, 121], "B": [200, 220, 242]}, index=dates)
        weights = pd.Series({"A": 0.5, "B": 0.5})
        returns = portfolio.compute_portfolio_returns(price_df, weights)
        # Both assets have 10% daily returns
        self.assertTrue(np.allclose(returns.values, [0.10, 0.10]))


if __name__ == "__main__":
    unittest.main()
