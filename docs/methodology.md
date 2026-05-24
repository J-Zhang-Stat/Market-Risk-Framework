# Methodology

This project converts a research notebook into a production-style market-risk workflow. The objective is not to forecast returns, but to quantify portfolio risk exposure, tail loss, and resilience under shocks.

## Data and Factors

For each asset, daily simple returns are computed from adjusted close prices:

```text
r_t = P_t / P_{t-1} - 1
```

The factor matrix contains:

- `MKT_SPY`: SPY daily return, used as broad equity market risk.
- `RATE_TNX`: daily change in 10-year Treasury yield, scaled to decimal yield units.
- `COM_DBC`: DBC daily return, used as broad commodity risk.
- `USD_UUP`: UUP daily return, used as USD strength risk.
- `VIX_CHG`: VIX daily percentage change, used as volatility regime risk.

## Factor Exposure

Asset-level exposures are estimated with OLS:

```text
r_i,t = alpha_i + beta_i,MKT MKT_t + beta_i,RATE RATE_t
        + beta_i,COM COM_t + beta_i,USD USD_t + beta_i,VIX VIX_t + epsilon_i,t
```

Portfolio exposure is the weighted sum of asset exposures:

```text
beta_portfolio = sum_i w_i * beta_i
```

Rolling exposures use a one-year trading window by default.

## VaR and ES

Historical VaR uses the empirical left-tail quantile:

```text
VaR_c = -Quantile(r, 1 - c)
```

Parametric VaR assumes normally distributed returns:

```text
VaR_c = -(mu + z_{1-c} * sigma)
```

Expected Shortfall averages losses beyond the VaR threshold:

```text
ES_c = -E[r | r <= Quantile(r, 1 - c)]
```

EWMA volatility follows the RiskMetrics recursion:

```text
sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
```

The EWMA-t model replaces the normal quantile with a Student-t quantile to better capture heavy tails.

## Backtesting

An exception occurs when realized return is lower than the negative VaR forecast:

```text
exception_t = 1 if r_t < -VaR_t else 0
```

The project reports:

- Kupiec unconditional coverage test: whether the exception frequency is statistically consistent with the VaR confidence level.
- Christoffersen independence test: whether exceptions cluster through time.
- Joint conditional coverage test: combines frequency and independence.
- Basel traffic-light classification: based on the last 250 observations for 99% VaR.

## Stress Testing

Scenario loss uses the factor model:

```text
Delta r_i ~= beta_i * Delta factors
Delta r_portfolio = sum_i w_i * Delta r_i
```

This makes it possible to compare crisis-style losses and identify which positions contribute most to downside risk.

## Rebalancing

The optimizer searches for a portfolio that meets a target VaR reduction while preserving realistic constraints:

- weights sum to 1;
- no short positions by default;
- individual asset caps;
- asset-class allocation bands;
- minimal turnover objective when the target is feasible.

If the exact target is infeasible under the constraints, the optimizer returns the lowest-risk feasible portfolio and reports the achieved reduction.

