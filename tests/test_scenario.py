import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules import scenario


class TestScenario(unittest.TestCase):

    def test_run_scenario_shocks(self):
        betas = pd.Series({"Rates": 0.5, "USD": -0.2})
        shocks = {"Rates": 0.01, "USD": -0.02}
        impact = scenario.run_scenario_shocks(betas, shocks)
        self.assertAlmostEqual(impact["Rates"], 0.005)
        self.assertAlmostEqual(impact["USD"], 0.004)
        self.assertAlmostEqual(impact["Total"], 0.009)


if __name__ == "__main__":
    unittest.main()
