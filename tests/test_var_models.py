import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_risk.var import expected_shortfall, historical_var, parametric_var, rolling_var


class TestVarModels(unittest.TestCase):
    def test_historical_var_positive_loss(self):
        returns = pd.Series([-0.10, -0.04, -0.02, 0.00, 0.01, 0.03])
        self.assertGreater(historical_var(returns, 0.95), 0)
        self.assertGreater(expected_shortfall(returns, 0.95), 0)

    def test_parametric_var_matches_formula_sign(self):
        returns = pd.Series(np.linspace(-0.03, 0.03, 200))
        var = parametric_var(returns, 0.99)
        self.assertGreater(var, 0)

    def test_rolling_parametric_var_uses_prior_window(self):
        returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300))
        var = rolling_var(returns, 0.99, method="parametric", window=250)
        self.assertTrue(var.iloc[:250].isna().all())
        self.assertGreater(var.dropna().iloc[0], 0)


if __name__ == "__main__":
    unittest.main()

