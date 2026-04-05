from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from investment_lab.backtest import StrategyBacktester
from investment_lab.metrics.performance import calmar_ratio, max_drawdown, sharpe_ratio
from investment_lab.option_trade import OptionTrade

from realized_vol_timing.data import CourseOptionMarketDataRepository, MarketDataRepository
from realized_vol_timing.signals import DynamicAllocationPolicy, SignalResult, SpreadSignalEngine


@dataclass
class TimedCarryResult:
    signal_result: SignalResult
    base_trades: pd.DataFrame
    timed_trades: pd.DataFrame
    base_backtest: StrategyBacktester | None = None
    timed_backtest: StrategyBacktester | None = None
    comparison: pd.DataFrame | None = None


class TradeAllocator:
    def __init__(self, signal_column: str = "allocation") -> None:
        self.signal_column = signal_column

    def apply(self, trades: pd.DataFrame, signal_frame: pd.DataFrame) -> pd.DataFrame:
        required_trade_columns = {"entry_date", "ticker", "weight"}
        required_signal_columns = {"date", "ticker", self.signal_column}
        missing_trade_columns = required_trade_columns.difference(trades.columns)
        missing_signal_columns = required_signal_columns.difference(signal_frame.columns)
        if missing_trade_columns:
            raise ValueError(f"Trades are missing required columns: {missing_trade_columns}")
        if missing_signal_columns:
            raise ValueError(f"Signal frame is missing required columns: {missing_signal_columns}")

        df_signal = signal_frame[["date", "ticker", self.signal_column]].rename(
            columns={"date": "entry_date", self.signal_column: "allocation_multiplier"}
        )
        df_trades = trades.copy()
        df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])
        df_signal["entry_date"] = pd.to_datetime(df_signal["entry_date"])
        df_trades = df_trades.merge(df_signal, on=["entry_date", "ticker"], how="left")
        df_trades["allocation_multiplier"] = df_trades["allocation_multiplier"].fillna(1.0)
        df_trades["weight"] = df_trades["weight"] * df_trades["allocation_multiplier"]
        return df_trades


class BacktestAnalyzer:
    @classmethod
    def compare(
        cls,
        base_backtest: StrategyBacktester,
        timed_backtest: StrategyBacktester,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                cls._summarize("Base carry", base_backtest),
                cls._summarize("Timed carry", timed_backtest),
            ]
        ).set_index("strategy")

    @classmethod
    def summarize_many(
        cls,
        backtests: dict[str, StrategyBacktester],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [cls._summarize(name, backtest) for name, backtest in backtests.items()]
        ).set_index("strategy")

    @staticmethod
    def _summarize(name: str, backtest: StrategyBacktester) -> dict[str, float]:
        returns = backtest.nav["NAV"].pct_change().dropna()
        if returns.empty:
            return {
                "strategy": name,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
            }
        return {
            "strategy": name,
            "total_return": float(backtest.nav["NAV"].iloc[-1] - 1.0),
            "sharpe_ratio": float(sharpe_ratio(returns)),
            "calmar_ratio": float(calmar_ratio(returns)),
            "max_drawdown": float(max_drawdown(returns)),
        }


class TimedCarryStrategyRunner:
    def __init__(
        self,
        market_data_repository: MarketDataRepository | None = None,
        signal_engine: SpreadSignalEngine | None = None,
        trade_allocator: TradeAllocator | None = None,
        trade_class=OptionTrade,
        backtester_class=StrategyBacktester,
    ) -> None:
        self.market_data_repository = market_data_repository or CourseOptionMarketDataRepository()
        self.signal_engine = signal_engine or SpreadSignalEngine(
            allocation_policy=DynamicAllocationPolicy(),
        )
        self.trade_allocator = trade_allocator or TradeAllocator()
        self.trade_class = trade_class
        self.backtester_class = backtester_class

    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: str | list[str],
        legs: list[dict[str, Any]],
        *,
        market_panel: pd.DataFrame | None = None,
        cost_neutral: bool = False,
        hedging_args: dict[str, Any] | None = None,
        tcost_args: dict[str, Any] | None = None,
    ) -> TimedCarryResult:
        panel = market_panel
        if panel is None:
            panel = self.market_data_repository.load_market_panel(start_date, end_date, tickers)

        signal_result = self.signal_engine.build_signal_panel(panel)
        base_trades = self.trade_class.generate_trades(
            start_date,
            end_date,
            tickers=tickers,
            legs=legs,
            cost_neutral=cost_neutral,
            hedging_args=hedging_args,
        )
        timed_trades = self.trade_allocator.apply(base_trades, signal_result.signal_frame)
        base_backtest = self.backtester_class(base_trades).compute_backtest(tcost_args=tcost_args)
        timed_backtest = self.backtester_class(timed_trades).compute_backtest(tcost_args=tcost_args)
        comparison = BacktestAnalyzer.compare(base_backtest, timed_backtest)
        return TimedCarryResult(
            signal_result=signal_result,
            base_trades=base_trades,
            timed_trades=timed_trades,
            base_backtest=base_backtest,
            timed_backtest=timed_backtest,
            comparison=comparison,
        )


def apply_allocation_to_trades(
    trades: pd.DataFrame,
    signal_frame: pd.DataFrame,
    signal_column: str = "allocation",
) -> pd.DataFrame:
    return TradeAllocator(signal_column=signal_column).apply(trades, signal_frame)


def compare_backtests(
    base_backtest: StrategyBacktester,
    timed_backtest: StrategyBacktester,
) -> pd.DataFrame:
    return BacktestAnalyzer.compare(base_backtest, timed_backtest)


def summarize_backtests(backtests: dict[str, StrategyBacktester]) -> pd.DataFrame:
    return BacktestAnalyzer.summarize_many(backtests)


def run_timed_carry_backtest(
    start_date: datetime,
    end_date: datetime,
    tickers: str | list[str],
    legs: list[dict[str, Any]],
    **kwargs,
) -> TimedCarryResult:
    return TimedCarryStrategyRunner().run_backtest(
        start_date,
        end_date,
        tickers,
        legs,
        **kwargs,
    )
