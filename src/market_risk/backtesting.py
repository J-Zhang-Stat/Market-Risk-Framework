from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass(frozen=True)
class BacktestResult:
    method: str
    confidence: float
    observations: int
    exceptions: int
    exception_rate: float
    expected_rate: float
    kupiec_lr: float
    kupiec_pvalue: float
    independence_lr: float
    independence_pvalue: float
    conditional_lr: float
    conditional_pvalue: float
    last_250_exceptions: int | None
    basel_traffic_light: str | None
    aligned: pd.DataFrame

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "method": self.method,
            "confidence": self.confidence,
            "observations": self.observations,
            "exceptions": self.exceptions,
            "exception_rate": self.exception_rate,
            "expected_rate": self.expected_rate,
            "kupiec_pvalue": self.kupiec_pvalue,
            "independence_pvalue": self.independence_pvalue,
            "conditional_pvalue": self.conditional_pvalue,
            "last_250_exceptions": self.last_250_exceptions,
            "basel_traffic_light": self.basel_traffic_light,
        }


def backtest_var(
    returns: pd.Series,
    var_forecast: pd.Series,
    confidence: float = 0.99,
    method: str = "VaR",
) -> BacktestResult:
    aligned = pd.DataFrame({"return": returns, "var": var_forecast}).dropna()
    aligned["exception"] = (aligned["return"] < -aligned["var"]).astype(int)
    observations = int(len(aligned))
    exceptions = int(aligned["exception"].sum())
    expected_rate = 1 - confidence
    exception_rate = exceptions / observations if observations else np.nan

    kupiec_lr, kupiec_p = kupiec_test(exceptions, observations, expected_rate)
    ind_lr, ind_p = christoffersen_independence_test(aligned["exception"])
    cc_lr = kupiec_lr + ind_lr
    cc_p = float(1 - chi2.cdf(cc_lr, df=2))
    last_250 = int(aligned["exception"].tail(250).sum()) if observations >= 250 else None
    traffic = basel_traffic_light(last_250) if confidence >= 0.99 and last_250 is not None else None

    return BacktestResult(
        method=method,
        confidence=confidence,
        observations=observations,
        exceptions=exceptions,
        exception_rate=exception_rate,
        expected_rate=expected_rate,
        kupiec_lr=kupiec_lr,
        kupiec_pvalue=kupiec_p,
        independence_lr=ind_lr,
        independence_pvalue=ind_p,
        conditional_lr=cc_lr,
        conditional_pvalue=cc_p,
        last_250_exceptions=last_250,
        basel_traffic_light=traffic,
        aligned=aligned,
    )


def kupiec_test(exceptions: int, observations: int, expected_rate: float) -> tuple[float, float]:
    if observations <= 0:
        return np.nan, np.nan
    phat = exceptions / observations
    ll_null = _binomial_log_likelihood(exceptions, observations, expected_rate)
    ll_alt = _binomial_log_likelihood(exceptions, observations, phat)
    lr = max(0.0, -2 * (ll_null - ll_alt))
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def christoffersen_independence_test(exceptions: pd.Series) -> tuple[float, float]:
    e = exceptions.astype(int).to_numpy()
    if len(e) < 2:
        return np.nan, np.nan

    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(e[:-1], e[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_ind = _transition_log_likelihood(n00, n01, n10, n11, pi01, pi11)
    ll_iid = _transition_log_likelihood(n00, n01, n10, n11, pi, pi)
    lr = max(0.0, -2 * (ll_iid - ll_ind))
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def basel_traffic_light(exceptions_250: int | None) -> str | None:
    if exceptions_250 is None:
        return None
    if exceptions_250 <= 4:
        return "Green"
    if exceptions_250 <= 9:
        return "Yellow"
    return "Red"


def summarize_backtests(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_dict() for result in results])


def _binomial_log_likelihood(exceptions: int, observations: int, probability: float) -> float:
    probability = min(max(probability, 1e-12), 1 - 1e-12)
    return exceptions * np.log(probability) + (observations - exceptions) * np.log(1 - probability)


def _transition_log_likelihood(
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    p01: float,
    p11: float,
) -> float:
    p01 = min(max(p01, 1e-12), 1 - 1e-12)
    p11 = min(max(p11, 1e-12), 1 - 1e-12)
    p00 = 1 - p01
    p10 = 1 - p11
    return (
        n00 * np.log(p00)
        + n01 * np.log(p01)
        + n10 * np.log(p10)
        + n11 * np.log(p11)
    )

