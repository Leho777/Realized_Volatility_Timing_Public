from __future__ import annotations

import pandas as pd

from investment_lab.metrics.volatility import rolling_realized_volatility

from realized_vol_timing.config import AllocationConfig
from realized_vol_timing.signals import DynamicAllocationPolicy


def add_realized_volatility_benchmarks(
    market_panel: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (21, 63),
    return_column: str = "log_return",
) -> pd.DataFrame:
    required_columns = {"date", "ticker", return_column}
    missing_columns = required_columns.difference(market_panel.columns)
    if missing_columns:
        raise ValueError(
            f"Market panel is missing required columns for RV benchmarks: {missing_columns}"
        )

    panel = market_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    for window in windows:
        rv_column = f"rv_{window}d"
        panel[rv_column] = panel.groupby("ticker")[return_column].transform(
            lambda returns: rolling_realized_volatility(
                returns,
                window=window,
                volatility_type="std",
            )
        )

    return panel


def build_realized_volatility_signal_panel(
    market_panel: pd.DataFrame,
    *,
    rv_column: str = "rv_21d",
    iv_column: str = "atm_implied_vol",
    allocation_config: AllocationConfig | None = None,
) -> pd.DataFrame:
    required_columns = {"date", "ticker", iv_column, rv_column}
    missing_columns = required_columns.difference(market_panel.columns)
    if missing_columns:
        raise ValueError(
            f"Market panel is missing required columns for the RV signal: {missing_columns}"
        )

    policy = DynamicAllocationPolicy(config=allocation_config)
    signal_frames: list[pd.DataFrame] = []

    for ticker, df_ticker in market_panel.sort_values(["ticker", "date"]).groupby("ticker"):
        ticker_signal = df_ticker.copy().set_index("date").sort_index()
        ticker_signal["benchmark_realized_vol"] = ticker_signal[rv_column]
        ticker_signal["spread"] = ticker_signal[iv_column] - ticker_signal[rv_column]
        allocation_frame = policy.transform(ticker_signal["spread"])
        ticker_signal = ticker_signal.join(allocation_frame, how="left")
        ticker_signal["allocation"] = ticker_signal["allocation"].fillna(
            policy.config.base_allocation
        )
        ticker_signal["benchmark_name"] = rv_column
        ticker_signal["ticker"] = ticker
        signal_frames.append(ticker_signal.reset_index())

    return pd.concat(signal_frames, ignore_index=True)
