from __future__ import annotations

import pandas as pd

from market_risk.exposure import FACTOR_COLUMNS


def scenario_asset_returns(
    asset_exposures: pd.DataFrame,
    shock: dict[str, float],
) -> pd.Series:
    shock_vector = pd.Series(shock, dtype=float).reindex(FACTOR_COLUMNS).fillna(0.0)
    return asset_exposures[FACTOR_COLUMNS].dot(shock_vector)


def run_stress_tests(
    weights: pd.Series,
    asset_exposures: pd.DataFrame,
    scenarios: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    contribution_rows = []
    common = [asset for asset in weights.index if asset in asset_exposures.index]
    for scenario_name, shock in scenarios.items():
        asset_shocks = scenario_asset_returns(asset_exposures.loc[common], shock)
        contributions = weights.loc[common] * asset_shocks
        summary_rows.append(
            {
                "scenario": scenario_name,
                "portfolio_return": float(contributions.sum()),
                "portfolio_loss": float(-contributions.sum()),
            }
        )
        for asset, contribution in contributions.items():
            contribution_rows.append(
                {
                    "scenario": scenario_name,
                    "asset": asset,
                    "weighted_contribution": float(contribution),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(contribution_rows)

