# Market Risk Analysis Summary

Data source: `synthetic`

## VaR and ES

- 99% Historical VaR: 2.05%
- 99% Parametric VaR: 2.05%
- 99% Monte Carlo VaR: 2.07%
- 99% Expected Shortfall: 3.32%

## Backtesting

Best conditional coverage p-value in this run:

- Method: EWMA t(df=5) 99%
- Exceptions: 15 / 1825
- Conditional coverage p-value: 0.647
- Last 250-day Basel traffic light: Green

See `var_backtests.csv` for the full comparison across VaR models.

## Stress Testing

Largest scenario loss:

- Scenario: 2008 Crisis
- Estimated portfolio loss: 27.64%

## Rebalancing

- Base 99% one-day VaR: 2.05%
- Optimized 99% one-day VaR: 1.64%
- Achieved VaR reduction: 20.0%
- Optimizer status: Optimization terminated successfully

