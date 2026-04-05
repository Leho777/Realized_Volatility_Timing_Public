from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from investment_lab import option_strategies

from realized_vol_timing.benchmarks import (
    add_realized_volatility_benchmarks,
    build_realized_volatility_signal_panel,
)
from realized_vol_timing.config import AllocationConfig, HestonUKFConfig, RollingWindowConfig
from realized_vol_timing.data import CourseOptionMarketDataRepository
from realized_vol_timing.heston_ukf import HestonUKF
from realized_vol_timing.signals import DynamicAllocationPolicy, SpreadSignalEngine
from realized_vol_timing.strategy import TimedCarryStrategyRunner, TradeAllocator, summarize_backtests


@dataclass
class CarryExperimentResult:
    ticker: str
    market_panel: pd.DataFrame
    signal_frame: pd.DataFrame
    parameter_frame: pd.DataFrame
    rv_signal_frame: pd.DataFrame
    comparison_frame: pd.DataFrame
    base_trades: pd.DataFrame
    ukf_timed_trades: pd.DataFrame
    rv_timed_trades: pd.DataFrame
    base_backtest: object
    ukf_timed_backtest: object
    rv_timed_backtest: object


_DEFAULT_DATE_RANGES: dict[str, dict[str, tuple[datetime, datetime]]] = {
    "SPY": {
        "demo": (datetime(2020, 1, 2), datetime(2020, 9, 30)),
        "full": (datetime(2020, 1, 2), datetime(2022, 12, 30)),
    },
    "AAPL": {
        "demo": (datetime(2020, 1, 2), datetime(2020, 9, 30)),
        "full": (datetime(2016, 1, 4), datetime(2023, 3, 31)),
    },
}


def resolve_experiment_dates(
    ticker: str,
    *,
    full_sample: bool = False,
) -> tuple[datetime, datetime]:
    normalized_ticker = ticker.upper()
    if normalized_ticker not in _DEFAULT_DATE_RANGES:
        raise ValueError(
            f"Unsupported ticker for notebook presets: {ticker}. "
            f"Supported presets: {sorted(_DEFAULT_DATE_RANGES)}"
        )

    preset = "full" if full_sample else "demo"
    return _DEFAULT_DATE_RANGES[normalized_ticker][preset]


def compute_proxy_metrics(signal_frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "proxy": "UKF sigma_hat",
            "corr_vs_rv_21d": signal_frame["estimated_realized_vol"].corr(signal_frame["rv_21d"]),
            "corr_vs_rv_63d": signal_frame["estimated_realized_vol"].corr(signal_frame["rv_63d"]),
            "mae_vs_rv_21d": (
                signal_frame["estimated_realized_vol"] - signal_frame["rv_21d"]
            ).abs().mean(),
            "mae_vs_rv_63d": (
                signal_frame["estimated_realized_vol"] - signal_frame["rv_63d"]
            ).abs().mean(),
        },
        {
            "proxy": "IV ATM",
            "corr_vs_rv_21d": signal_frame["atm_implied_vol"].corr(signal_frame["rv_21d"]),
            "corr_vs_rv_63d": signal_frame["atm_implied_vol"].corr(signal_frame["rv_63d"]),
            "mae_vs_rv_21d": (
                signal_frame["atm_implied_vol"] - signal_frame["rv_21d"]
            ).abs().mean(),
            "mae_vs_rv_63d": (
                signal_frame["atm_implied_vol"] - signal_frame["rv_63d"]
            ).abs().mean(),
        },
    ]
    return pd.DataFrame(rows).set_index("proxy")


def run_carry_experiment(
    *,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    implied_vol_maturity_days: int = 30,
    ukf_config: HestonUKFConfig | None = None,
    rolling_config: RollingWindowConfig | None = None,
    allocation_config: AllocationConfig | None = None,
) -> CarryExperimentResult:
    resolved_ukf_config = ukf_config or HestonUKFConfig()
    resolved_rolling_config = rolling_config or RollingWindowConfig()
    resolved_allocation_config = allocation_config or AllocationConfig()

    repository = CourseOptionMarketDataRepository(
        implied_vol_maturity_days=implied_vol_maturity_days,
    )
    market_panel = repository.load_market_panel(start_date, end_date, ticker)
    market_panel = add_realized_volatility_benchmarks(market_panel, windows=(21, 63))

    signal_engine = SpreadSignalEngine(
        ukf=HestonUKF(config=resolved_ukf_config),
        rolling_config=resolved_rolling_config,
        allocation_policy=DynamicAllocationPolicy(config=resolved_allocation_config),
    )
    runner = TimedCarryStrategyRunner(
        market_data_repository=repository,
        signal_engine=signal_engine,
    )
    strategy_result = runner.run_backtest(
        start_date=start_date,
        end_date=end_date,
        tickers=ticker,
        legs=option_strategies.SHORT_1W_STRANGLE_95_105,
        market_panel=market_panel,
    )

    rv_signal_frame = build_realized_volatility_signal_panel(
        market_panel,
        rv_column="rv_21d",
        allocation_config=resolved_allocation_config,
    )
    rv_timed_trades = TradeAllocator().apply(strategy_result.base_trades, rv_signal_frame)
    rv_timed_backtest = runner.backtester_class(rv_timed_trades).compute_backtest()

    comparison_frame = summarize_backtests(
        {
            "Base carry": strategy_result.base_backtest,
            "Timed carry UKF": strategy_result.timed_backtest,
            "Timed carry RV_21d": rv_timed_backtest,
        }
    )

    signal_frame = strategy_result.signal_result.signal_frame.dropna(subset=["spread"]).copy()
    parameter_frame = strategy_result.signal_result.parameter_frame.copy()
    rv_signal_frame = rv_signal_frame.dropna(subset=["spread"]).copy()

    return CarryExperimentResult(
        ticker=ticker,
        market_panel=market_panel,
        signal_frame=signal_frame,
        parameter_frame=parameter_frame,
        rv_signal_frame=rv_signal_frame,
        comparison_frame=comparison_frame,
        base_trades=strategy_result.base_trades,
        ukf_timed_trades=strategy_result.timed_trades,
        rv_timed_trades=rv_timed_trades,
        base_backtest=strategy_result.base_backtest,
        ukf_timed_backtest=strategy_result.timed_backtest,
        rv_timed_backtest=rv_timed_backtest,
    )
