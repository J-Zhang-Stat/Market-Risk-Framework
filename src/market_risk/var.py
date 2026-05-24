from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    common = [asset for asset in weights.index if asset in returns.columns]
    return returns[common].dot(weights.loc[common]).dropna()


def historical_var(x: pd.Series, confidence: float = 0.99) -> float:
    return float(-x.dropna().quantile(1 - confidence))


def parametric_var(x: pd.Series, confidence: float = 0.99) -> float:
    clean = x.dropna()
    return float(-(clean.mean() + norm.ppf(1 - confidence) * clean.std(ddof=1)))


def monte_carlo_var(
    x: pd.Series,
    confidence: float = 0.99,
    simulations: int = 100_000,
    seed: int = 42,
) -> float:
    clean = x.dropna()
    rng = np.random.default_rng(seed)
    simulated = rng.normal(clean.mean(), clean.std(ddof=1), simulations)
    return float(-np.quantile(simulated, 1 - confidence))


def expected_shortfall(x: pd.Series, confidence: float = 0.99) -> float:
    clean = x.dropna()
    threshold = clean.quantile(1 - confidence)
    tail = clean[clean <= threshold]
    return float(-tail.mean())


def rolling_var(
    x: pd.Series,
    confidence: float = 0.99,
    method: str = "historical",
    window: int = 250,
    min_periods: int | None = None,
) -> pd.Series:
    clean = x.dropna()
    min_periods = min_periods or window
    if method == "historical":
        return -clean.rolling(window, min_periods=min_periods).quantile(1 - confidence).shift(1)
    if method == "parametric":
        mu = clean.rolling(window, min_periods=min_periods).mean().shift(1)
        sigma = clean.rolling(window, min_periods=min_periods).std(ddof=1).shift(1)
        return -(mu + norm.ppf(1 - confidence) * sigma)
    raise ValueError("method must be 'historical' or 'parametric'")


def ewma_volatility(x: pd.Series, lam: float = 0.94) -> pd.Series:
    clean = x.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    sigma2 = np.empty(len(clean))
    sigma2[0] = clean.var(ddof=1)
    for i in range(1, len(clean)):
        sigma2[i] = lam * sigma2[i - 1] + (1 - lam) * clean.iloc[i - 1] ** 2
    return pd.Series(np.sqrt(sigma2), index=clean.index)


def ewma_var(
    x: pd.Series,
    confidence: float = 0.99,
    lam: float = 0.94,
    distribution: str = "normal",
    df: int = 5,
) -> pd.Series:
    sigma = ewma_volatility(x, lam=lam)
    if distribution == "normal":
        q = norm.ppf(confidence)
    elif distribution == "t":
        q = student_t.ppf(confidence, df=df) / np.sqrt(df / (df - 2))
    else:
        raise ValueError("distribution must be 'normal' or 't'")
    return q * sigma


def var_metric_table(x: pd.Series, confidence_levels: tuple[float, ...] = (0.95, 0.99)) -> pd.DataFrame:
    rows = []
    for confidence in confidence_levels:
        rows.append(
            {
                "confidence": confidence,
                "historical_var": historical_var(x, confidence),
                "parametric_var": parametric_var(x, confidence),
                "monte_carlo_var": monte_carlo_var(x, confidence),
                "expected_shortfall": expected_shortfall(x, confidence),
            }
        )
    return pd.DataFrame(rows)

