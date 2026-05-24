from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_COLUMNS = ["MKT_SPY", "RATE_TNX", "COM_DBC", "USD_UUP", "VIX_CHG"]
EXPOSURE_COLUMNS = ["const", *FACTOR_COLUMNS]


def ols_beta(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    aligned = pd.concat([y.rename("y"), x], axis=1).dropna()
    if aligned.empty:
        return pd.Series(np.nan, index=EXPOSURE_COLUMNS)
    x_mat = np.column_stack([np.ones(len(aligned)), aligned[x.columns].to_numpy()])
    beta = np.linalg.lstsq(x_mat, aligned["y"].to_numpy(), rcond=None)[0]
    return pd.Series(beta, index=["const", *list(x.columns)])


def rolling_betas(
    y: pd.Series,
    x: pd.DataFrame,
    window: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    out = np.full((len(y), len(x.columns) + 1), np.nan)
    for i in range(len(y)):
        start = max(0, i + 1 - window)
        if i + 1 - start < min_periods:
            continue
        y_win = y.iloc[start : i + 1]
        x_win = x.iloc[start : i + 1]
        out[i, :] = ols_beta(y_win, x_win).to_numpy()
    return pd.DataFrame(out, index=y.index, columns=["const", *list(x.columns)])


def estimate_asset_exposures(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    assets: list[str],
    window: int = 252,
    min_periods: int = 126,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = factors[FACTOR_COLUMNS].fillna(0.0)
    full_rows = []
    rolling = {}
    for asset in assets:
        y = returns[asset].fillna(0.0)
        full_rows.append(ols_beta(y, x).rename(asset))
        rolling[asset] = rolling_betas(y, x, window=window, min_periods=min_periods)
    full = pd.DataFrame(full_rows)
    roll_panel = pd.concat(rolling, axis=1)
    return full, roll_panel


def portfolio_exposure(asset_exposures: pd.DataFrame, weights: pd.Series) -> pd.Series:
    common = [asset for asset in weights.index if asset in asset_exposures.index]
    weighted = asset_exposures.loc[common].multiply(weights.loc[common], axis=0)
    return weighted.sum(axis=0).rename("Portfolio")


def rolling_portfolio_exposure(rolling_panel: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    exposures = pd.DataFrame(index=rolling_panel.index, columns=EXPOSURE_COLUMNS, dtype=float)
    common = [asset for asset in weights.index if asset in rolling_panel.columns.get_level_values(0)]
    for factor in EXPOSURE_COLUMNS:
        block = rolling_panel.loc[:, pd.IndexSlice[common, factor]]
        exposures[factor] = block.dot(weights.loc[block.columns.get_level_values(0)].to_numpy())
    return exposures


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    clean = returns.fillna(0.0)
    cov = np.zeros((clean.shape[1], clean.shape[1]))
    for row in clean.to_numpy():
        x = row.reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (x @ x.T)
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)

