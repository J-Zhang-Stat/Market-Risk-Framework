# Recruiter Walkthrough

This file is written for a fast GitHub review. It highlights what to open first and what each component proves.

## 2-Minute Review Path

1. Open `README.md` for the project summary, screenshots, and sample results.
2. Open `scripts/run_analysis.py` to see the end-to-end workflow.
3. Open `src/market_risk/var.py` and `src/market_risk/backtesting.py` for the core risk model implementation.
4. Open `src/market_risk/rebalancing.py` for the portfolio optimization component.
5. Open `reports/summary.md` and `reports/var_backtests.csv` for generated results.

## Interview Talking Points

- I converted a research notebook into a modular Python package with reproducible outputs and tests.
- I corrected notebook-level pitfalls such as look-ahead alignment and VaR sign conventions before packaging the code.
- I implemented multiple VaR models and evaluated them with statistical backtests instead of relying on visual inspection.
- I used factor betas to translate macro shock assumptions into asset-level and portfolio-level stress losses.
- I framed rebalancing as a constrained optimization problem: reduce VaR while preserving realistic asset-class bands.

## Files That Best Show Engineering Quality

- `src/market_risk/backtesting.py`: Kupiec and Christoffersen tests with edge-case handling.
- `src/market_risk/var.py`: static, rolling, Monte Carlo, and EWMA VaR implementations.
- `src/market_risk/rebalancing.py`: SLSQP optimizer with target-risk and asset-class constraints.
- `tests/`: simple unit tests around sign conventions, exception counts, and Basel traffic-light rules.

## Suggested GitHub Description

Multi-asset market risk framework in Python with factor exposures, VaR/ES models, statistical VaR backtesting, stress testing, and VaR-constrained portfolio rebalancing.

