from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_risk.config import DEFAULT_CLASS_WEIGHTS, AssetUniverse


@dataclass(frozen=True)
class MarketDataset:
    prices: pd.DataFrame
    returns: pd.DataFrame
    factors: pd.DataFrame


def fetch_yfinance_prices(
    universe: AssetUniverse,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    prices = _yf_download(universe.all_tickers, start, end)
    factor_prices = _yf_download(list(universe.factors.values()), start, end)
    factor_prices = factor_prices.rename(
        columns={ticker: name for name, ticker in universe.factors.items()}
    )
    return prices.sort_index(), factor_prices.sort_index()


def _yf_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]
    elif "Adj Close" in df:
        df = df["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df


def generate_synthetic_prices(
    universe: AssetUniverse,
    start: str = "2018-01-02",
    end: str = "2024-12-31",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic market data with crisis-like volatility regimes."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    factor_cols = ["MKT_SPY", "RATE_TNX", "COM_DBC", "USD_UUP", "VIX_CHG"]
    corr = np.array(
        [
            [1.00, -0.18, 0.45, -0.25, -0.68],
            [-0.18, 1.00, 0.10, 0.15, 0.12],
            [0.45, 0.10, 1.00, -0.22, -0.25],
            [-0.25, 0.15, -0.22, 1.00, 0.34],
            [-0.68, 0.12, -0.25, 0.34, 1.00],
        ]
    )
    base_vol = np.array([0.009, 0.00045, 0.010, 0.004, 0.055])
    regime = np.ones(n)
    regime[(dates >= "2020-02-18") & (dates <= "2020-04-30")] = 2.7
    regime[(dates >= "2022-01-03") & (dates <= "2022-10-31")] = 1.6
    regime[(dates >= "2018-10-01") & (dates <= "2018-12-31")] = 1.5

    shocks = rng.multivariate_normal(np.zeros(5), corr, size=n)
    factors = shocks * base_vol * regime[:, None]
    factor_df = pd.DataFrame(factors, index=dates, columns=factor_cols)

    crisis_days = {
        "2020-03-12": [-0.095, -0.0018, -0.045, 0.014, 0.70],
        "2020-03-16": [-0.110, -0.0015, -0.060, 0.018, 0.95],
        "2022-06-13": [-0.040, 0.0016, -0.030, 0.010, 0.28],
    }
    for date, shock in crisis_days.items():
        ts = pd.Timestamp(date)
        if ts in factor_df.index:
            factor_df.loc[ts] += shock

    beta_map = {
        "AAPL": [1.25, -0.60, 0.05, -0.15, -0.08],
        "MSFT": [1.15, -0.45, 0.02, -0.10, -0.07],
        "JPM": [1.10, 0.85, 0.10, 0.05, -0.06],
        "XOM": [0.85, 0.25, 0.75, 0.02, -0.05],
        "PG": [0.55, -0.20, 0.05, 0.08, -0.03],
        "JNJ": [0.60, -0.25, 0.03, 0.06, -0.03],
        "TLT": [-0.35, -5.50, -0.05, 0.03, 0.04],
        "LQD": [0.25, -2.20, 0.02, 0.02, -0.02],
        "GLD": [0.05, -1.20, 0.25, -0.22, 0.03],
        "DBC": [0.20, 0.20, 1.00, -0.18, -0.02],
        "UUP": [-0.12, 0.30, -0.20, 1.00, 0.03],
        "SPY": [1.00, 0.00, 0.00, 0.00, 0.00],
    }
    idio_vol = {
        "AAPL": 0.010,
        "MSFT": 0.009,
        "JPM": 0.011,
        "XOM": 0.012,
        "PG": 0.006,
        "JNJ": 0.006,
        "TLT": 0.006,
        "LQD": 0.004,
        "GLD": 0.007,
        "DBC": 0.008,
        "UUP": 0.003,
        "SPY": 0.002,
    }

    returns = pd.DataFrame(index=dates)
    for asset in universe.all_tickers:
        beta = np.array(beta_map[asset])
        noise = rng.normal(0.0, idio_vol[asset], size=n) * np.sqrt(regime)
        returns[asset] = 0.00015 + factor_df.values @ beta + noise

    prices = 100.0 * (1.0 + returns).clip(lower=0.70).cumprod()
    factor_prices = _factor_returns_to_prices(factor_df)
    return prices, factor_prices


def _factor_returns_to_prices(factors: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=factors.index)
    out["MKT_SPY"] = 100 * (1 + factors["MKT_SPY"]).cumprod()
    out["RATE_TNX"] = (0.025 + factors["RATE_TNX"].cumsum()).clip(lower=0.001) * 1000
    out["COM_DBC"] = 20 * (1 + factors["COM_DBC"]).cumprod()
    out["USD_UUP"] = 25 * (1 + factors["USD_UUP"]).cumprod()
    out["VIX"] = (18 * (1 + factors["VIX_CHG"]).clip(lower=0.10).cumprod()).clip(
        lower=8,
        upper=90,
    )
    return out


def build_market_dataset(prices_raw: pd.DataFrame, factor_prices_raw: pd.DataFrame) -> MarketDataset:
    idx = prices_raw.index.union(factor_prices_raw.index).sort_values()
    prices = prices_raw.reindex(idx).ffill().dropna(how="all")
    factor_prices = factor_prices_raw.reindex(idx).ffill().dropna(how="all")
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

    factors = pd.DataFrame(index=idx)
    factors["MKT_SPY"] = _pct_or_zero(factor_prices, "MKT_SPY", idx)
    if "RATE_TNX" in factor_prices:
        factors["RATE_TNX"] = (factor_prices["RATE_TNX"] / 1000.0).diff().fillna(0.0)
    else:
        factors["RATE_TNX"] = 0.0
    factors["COM_DBC"] = _pct_or_zero(factor_prices, "COM_DBC", idx)
    factors["USD_UUP"] = _pct_or_zero(factor_prices, "USD_UUP", idx)
    factors["VIX_CHG"] = _pct_or_zero(factor_prices, "VIX", idx)
    factors = factors.reindex(returns.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return MarketDataset(prices=prices.loc[returns.index], returns=returns, factors=factors)


def _pct_or_zero(df: pd.DataFrame, column: str, index: pd.Index) -> pd.Series:
    if column in df:
        return df[column].pct_change()
    return pd.Series(0.0, index=index)


def build_default_weights(
    universe: AssetUniverse,
    class_weights: dict[str, float] | None = None,
) -> pd.Series:
    class_weights = class_weights or DEFAULT_CLASS_WEIGHTS
    buckets = {
        "equities": universe.equities,
        "bonds": universe.bonds,
        "commodities": universe.commodities,
        "fx": universe.fx,
    }
    weights: dict[str, float] = {}
    for bucket, assets in buckets.items():
        if assets:
            per_asset = class_weights[bucket] / len(assets)
            weights.update({asset: per_asset for asset in assets})
    out = pd.Series(weights, dtype=float)
    return out / out.sum()
