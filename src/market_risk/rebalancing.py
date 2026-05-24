from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from market_risk.config import DEFAULT_CLASS_BANDS, AssetUniverse


@dataclass(frozen=True)
class RebalanceResult:
    base_var: float
    optimized_var: float
    achieved_reduction: float
    success: bool
    message: str
    weights: pd.DataFrame


def parametric_portfolio_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
) -> float:
    portfolio = returns.to_numpy() @ weights
    return float(-(portfolio.mean() + norm.ppf(1 - confidence) * portfolio.std(ddof=1)))


def rebalance_to_var_target(
    returns: pd.DataFrame,
    base_weights: pd.Series,
    universe: AssetUniverse,
    confidence: float = 0.99,
    target_reduction: float = 0.20,
    max_weight: float = 0.25,
    class_bands: dict[str, tuple[float, float]] | None = None,
) -> RebalanceResult:
    assets = [asset for asset in base_weights.index if asset in returns.columns]
    clean_returns = returns[assets].fillna(0.0)
    w0 = base_weights.loc[assets].to_numpy(dtype=float)
    base_var = parametric_portfolio_var(clean_returns, w0, confidence=confidence)
    target_var = base_var * (1 - target_reduction)
    class_bands = class_bands or DEFAULT_CLASS_BANDS

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    constraints.extend(_class_band_constraints(assets, universe, class_bands))
    constraints.append(
        {
            "type": "ineq",
            "fun": lambda w: target_var - parametric_portfolio_var(clean_returns, w, confidence),
        }
    )
    bounds = [(0.0, max_weight) for _ in assets]

    result = minimize(
        lambda w: np.sum(np.abs(w - w0)),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not result.success:
        fallback_constraints = constraints[:-1]
        result = minimize(
            lambda w: parametric_portfolio_var(clean_returns, w, confidence),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=fallback_constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

    w_opt = pd.Series(result.x, index=assets).clip(lower=0.0)
    w_opt = w_opt / w_opt.sum()
    opt_var = parametric_portfolio_var(clean_returns, w_opt.to_numpy(), confidence=confidence)
    reduction = (base_var - opt_var) / base_var if base_var else np.nan
    weights = pd.DataFrame({"base_weight": base_weights.loc[assets], "optimized_weight": w_opt})
    weights["change"] = weights["optimized_weight"] - weights["base_weight"]

    return RebalanceResult(
        base_var=base_var,
        optimized_var=opt_var,
        achieved_reduction=float(reduction),
        success=bool(result.success),
        message=str(result.message),
        weights=weights,
    )


def _class_band_constraints(
    assets: list[str],
    universe: AssetUniverse,
    class_bands: dict[str, tuple[float, float]],
) -> list[dict[str, object]]:
    buckets = {
        "equities": set(universe.equities),
        "bonds": set(universe.bonds),
        "commodities": set(universe.commodities),
        "fx": set(universe.fx),
    }
    constraints: list[dict[str, object]] = []
    for name, members in buckets.items():
        if name not in class_bands:
            continue
        idx = [i for i, asset in enumerate(assets) if asset in members]
        if not idx:
            continue
        lower, upper = class_bands[name]
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx, lower=lower: np.sum(w[idx]) - lower})
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx, upper=upper: upper - np.sum(w[idx])})
    return constraints

