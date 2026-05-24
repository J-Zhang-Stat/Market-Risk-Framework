import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_risk.backtesting import backtest_var, basel_traffic_light


class TestBacktesting(unittest.TestCase):
    def test_exception_count(self):
        returns = pd.Series([-0.02, -0.01, 0.00, 0.01])
        var = pd.Series([0.015, 0.015, 0.015, 0.015])
        result = backtest_var(returns, var, confidence=0.99, method="test")
        self.assertEqual(result.exceptions, 1)
        self.assertEqual(result.observations, 4)

    def test_basel_traffic_light(self):
        self.assertEqual(basel_traffic_light(4), "Green")
        self.assertEqual(basel_traffic_light(5), "Yellow")
        self.assertEqual(basel_traffic_light(10), "Red")


if __name__ == "__main__":
    unittest.main()

