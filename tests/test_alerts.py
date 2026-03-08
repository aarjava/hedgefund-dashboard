import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules import alerts


class TestAlerts(unittest.TestCase):

    def test_evaluate_alerts(self):
        metrics = {"Volatility": 0.4, "MaxDrawdown": -0.3}
        thresholds = {
            "Volatility": {"type": ">", "value": 0.3, "severity": "High"},
            "MaxDrawdown": {"type": "<", "value": -0.2, "severity": "High"},
        }
        df = alerts.evaluate_alerts(metrics, thresholds)
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
