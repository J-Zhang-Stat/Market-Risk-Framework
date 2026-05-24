from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetUniverse:
    equities: tuple[str, ...]
    bonds: tuple[str, ...]
    commodities: tuple[str, ...]
    fx: tuple[str, ...]
    benchmark: tuple[str, ...]
    factors: dict[str, str]

    @property
    def portfolio_assets(self) -> list[str]:
        return list(self.equities + self.bonds + self.commodities + self.fx)

    @property
    def all_tickers(self) -> list[str]:
        return list(dict.fromkeys(self.portfolio_assets + list(self.benchmark)))


def default_universe() -> AssetUniverse:
    return AssetUniverse(
        equities=("AAPL", "MSFT", "JPM", "XOM", "PG", "JNJ"),
        bonds=("TLT", "LQD"),
        commodities=("GLD", "DBC"),
        fx=("UUP",),
        benchmark=("SPY",),
        factors={
            "MKT_SPY": "SPY",
            "RATE_TNX": "^TNX",
            "COM_DBC": "DBC",
            "USD_UUP": "UUP",
            "VIX": "^VIX",
        },
    )


DEFAULT_CLASS_WEIGHTS = {
    "equities": 0.60,
    "bonds": 0.25,
    "commodities": 0.10,
    "fx": 0.05,
}


DEFAULT_SCENARIOS = {
    "2008 Crisis": {
        "MKT_SPY": -0.35,
        "RATE_TNX": 0.015,
        "COM_DBC": -0.10,
        "USD_UUP": 0.03,
        "VIX_CHG": 2.00,
    },
    "COVID Shock": {
        "MKT_SPY": -0.25,
        "RATE_TNX": -0.010,
        "COM_DBC": -0.08,
        "USD_UUP": 0.02,
        "VIX_CHG": 1.50,
    },
    "Inflation Rate Shock": {
        "MKT_SPY": -0.12,
        "RATE_TNX": 0.020,
        "COM_DBC": 0.12,
        "USD_UUP": 0.04,
        "VIX_CHG": 0.60,
    },
    "Bear 15%": {
        "MKT_SPY": -0.15,
        "RATE_TNX": 0.005,
        "COM_DBC": -0.05,
        "USD_UUP": 0.02,
        "VIX_CHG": 0.80,
    },
}


DEFAULT_CLASS_BANDS = {
    "equities": (0.35, 0.65),
    "bonds": (0.15, 0.45),
    "commodities": (0.05, 0.25),
    "fx": (0.00, 0.12),
}

