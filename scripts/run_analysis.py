from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_risk.backtesting import backtest_var, summarize_backtests
from market_risk.config import DEFAULT_SCENARIOS, default_universe
from market_risk.data import (
    build_default_weights,
    build_market_dataset,
    fetch_yfinance_prices,
    generate_synthetic_prices,
)
from market_risk.exposure import (
    estimate_asset_exposures,
    portfolio_exposure,
    rolling_portfolio_exposure,
)
from market_risk.rebalancing import rebalance_to_var_target
from market_risk.stress import run_stress_tests
from market_risk.var import ewma_var, portfolio_returns, rolling_var, var_metric_table


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    universe = default_universe()
    if args.source == "yfinance":
        prices_raw, factor_prices = fetch_yfinance_prices(universe, args.start, args.end)
    else:
        prices_raw, factor_prices = generate_synthetic_prices(universe, args.start, args.end, seed=args.seed)

    dataset = build_market_dataset(prices_raw, factor_prices)
    weights = build_default_weights(universe)
    assets = universe.portfolio_assets
    asset_exposures, rolling_panel = estimate_asset_exposures(dataset.returns, dataset.factors, assets)
    port_exposure = portfolio_exposure(asset_exposures, weights)
    rolling_port = rolling_portfolio_exposure(rolling_panel, weights)
    rp = portfolio_returns(dataset.returns, weights)

    var_metrics = var_metric_table(rp)
    backtests = [
        backtest_var(rp, rolling_var(rp, 0.99, method="historical"), 0.99, "Historical 99%"),
        backtest_var(rp, rolling_var(rp, 0.99, method="parametric"), 0.99, "Parametric Normal 99%"),
        backtest_var(rp, ewma_var(rp, 0.99, distribution="normal"), 0.99, "EWMA Normal 99%"),
        backtest_var(rp, ewma_var(rp, 0.99, distribution="t", df=5), 0.99, "EWMA t(df=5) 99%"),
        backtest_var(rp, ewma_var(rp, 0.95, distribution="normal"), 0.95, "EWMA Normal 95%"),
    ]
    backtest_table = summarize_backtests(backtests)
    stress_summary, stress_contrib = run_stress_tests(weights, asset_exposures, DEFAULT_SCENARIOS)
    rebalance = rebalance_to_var_target(
        dataset.returns,
        weights,
        universe,
        confidence=0.99,
        target_reduction=args.target_reduction,
    )

    asset_exposures.to_csv(output_dir / "factor_exposures.csv")
    port_exposure.to_frame().T.to_csv(output_dir / "portfolio_exposures.csv", index=False)
    var_metrics.to_csv(output_dir / "var_metrics.csv", index=False)
    backtest_table.to_csv(output_dir / "var_backtests.csv", index=False)
    stress_summary.to_csv(output_dir / "stress_summary.csv", index=False)
    stress_contrib.to_csv(output_dir / "stress_contributions.csv", index=False)
    rebalance.weights.to_csv(output_dir / "rebalanced_weights.csv", index_label="asset")

    plot_rolling_exposure(rolling_port, figure_dir / "rolling_portfolio_exposure.png")
    plot_var_backtest(backtests[3].aligned, figure_dir / "ewma_t_var_backtest.png")
    plot_stress(stress_summary, figure_dir / "stress_losses.png")
    plot_weights(rebalance.weights, figure_dir / "rebalanced_weights.png")

    write_summary(output_dir / "summary.md", args.source, var_metrics, backtest_table, stress_summary, rebalance)
    print(f"Analysis complete. Reports written to {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the market risk framework end to end.")
    parser.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-reduction", type=float, default=0.20)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def plot_rolling_exposure(rolling_port, path: Path) -> None:
    ax = rolling_port[["MKT_SPY", "RATE_TNX", "COM_DBC", "USD_UUP", "VIX_CHG"]].plot(figsize=(10, 4))
    ax.set_title("Rolling Portfolio Factor Exposures")
    ax.set_ylabel("Rolling beta")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_var_backtest(aligned, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(aligned.index, aligned["return"], label="Portfolio return", linewidth=0.8)
    ax.plot(aligned.index, -aligned["var"], label="-VaR forecast", linewidth=1.0)
    exceptions = aligned[aligned["exception"] == 1]
    ax.scatter(exceptions.index, exceptions["return"], s=12, color="crimson", label="Exception", zorder=3)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("VaR Backtest: Realized Return vs Forecast Loss Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_stress(stress_summary, path: Path) -> None:
    ax = stress_summary.set_index("scenario")["portfolio_loss"].mul(100).plot(kind="bar", figsize=(8, 4))
    ax.set_title("Stress Test Portfolio Loss")
    ax.set_ylabel("Loss (%)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_weights(weights, path: Path) -> None:
    ax = weights[["base_weight", "optimized_weight"]].plot(kind="bar", figsize=(10, 4))
    ax.set_title("Base vs Risk-Aware Rebalanced Weights")
    ax.set_ylabel("Portfolio weight")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_summary(path: Path, source: str, var_metrics, backtests, stress_summary, rebalance) -> None:
    var99 = var_metrics.loc[var_metrics["confidence"] == 0.99].iloc[0]
    best = backtests.sort_values("conditional_pvalue", ascending=False).iloc[0]
    worst_stress = stress_summary.sort_values("portfolio_loss", ascending=False).iloc[0]
    content = f"""# Market Risk Analysis Summary

Data source: `{source}`

## VaR and ES

- 99% Historical VaR: {var99["historical_var"]:.2%}
- 99% Parametric VaR: {var99["parametric_var"]:.2%}
- 99% Monte Carlo VaR: {var99["monte_carlo_var"]:.2%}
- 99% Expected Shortfall: {var99["expected_shortfall"]:.2%}

## Backtesting

Best conditional coverage p-value in this run:

- Method: {best["method"]}
- Exceptions: {int(best["exceptions"])} / {int(best["observations"])}
- Conditional coverage p-value: {best["conditional_pvalue"]:.3f}
- Last 250-day Basel traffic light: {best["basel_traffic_light"]}

See `var_backtests.csv` for the full comparison across VaR models.

## Stress Testing

Largest scenario loss:

- Scenario: {worst_stress["scenario"]}
- Estimated portfolio loss: {worst_stress["portfolio_loss"]:.2%}

## Rebalancing

- Base 99% one-day VaR: {rebalance.base_var:.2%}
- Optimized 99% one-day VaR: {rebalance.optimized_var:.2%}
- Achieved VaR reduction: {rebalance.achieved_reduction:.1%}
- Optimizer status: {rebalance.message}

"""
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
