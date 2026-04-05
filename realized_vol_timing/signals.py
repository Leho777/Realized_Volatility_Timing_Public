from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from realized_vol_timing.config import AllocationConfig, HestonUKFConfig, RollingWindowConfig
from realized_vol_timing.heston_ukf import HestonUKF


@dataclass
class SignalResult:
    signal_frame: pd.DataFrame
    parameter_frame: pd.DataFrame


class DynamicAllocationPolicy:
    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig()

    def transform(self, spread: pd.Series) -> pd.DataFrame:
        signal = pd.Series(spread).astype(float)
        smoothed_spread = (
            signal.ewm(
                span=self.config.ewm_span,
                adjust=False,
                min_periods=1,
            ).mean()
            if self.config.ewm_span
            else signal.copy()
        )
        rolling_mean = smoothed_spread.rolling(self.config.zscore_window, min_periods=1).mean()
        rolling_std = smoothed_spread.rolling(
            self.config.zscore_window,
            min_periods=5,
        ).std()
        zscore = ((smoothed_spread - rolling_mean) / rolling_std.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        zscore = zscore.fillna(0.0).clip(-self.config.zscore_clip, self.config.zscore_clip)
        allocation = (
            self.config.base_allocation + self.config.sensitivity * zscore
        ).clip(self.config.min_allocation, self.config.max_allocation)
        return pd.DataFrame(
            {
                "smoothed_spread": smoothed_spread,
                "spread_zscore": zscore,
                "allocation": allocation,
            },
            index=signal.index,
        )


class SpreadSignalEngine:
    REQUIRED_COLUMNS = {"date", "ticker", "log_return", "atm_implied_vol"}

    def __init__(
        self,
        ukf: HestonUKF | None = None,
        ukf_config: HestonUKFConfig | None = None,
        rolling_config: RollingWindowConfig | None = None,
        allocation_policy: DynamicAllocationPolicy | None = None,
    ) -> None:
        self.ukf = ukf or HestonUKF(config=ukf_config)
        self.rolling_config = rolling_config or RollingWindowConfig()
        self.allocation_policy = allocation_policy or DynamicAllocationPolicy()

    def build_signal_panel(self, market_panel: pd.DataFrame) -> SignalResult:
        missing_columns = self.REQUIRED_COLUMNS.difference(market_panel.columns)
        if missing_columns:
            raise ValueError(f"Market panel is missing required columns: {missing_columns}")

        signal_frames: list[pd.DataFrame] = []
        parameter_frames: list[pd.DataFrame] = []

        for ticker, df_ticker in market_panel.sort_values(["ticker", "date"]).groupby("ticker"):
            ticker_result = self._build_ticker_signal(df_ticker.copy(), ticker)
            signal_frames.append(ticker_result.signal_frame)
            parameter_frames.append(ticker_result.parameter_frame)

        return SignalResult(
            signal_frame=pd.concat(signal_frames, ignore_index=True),
            parameter_frame=pd.concat(parameter_frames, ignore_index=True),
        )

    def _build_ticker_signal(self, df_ticker: pd.DataFrame, ticker: str) -> SignalResult:
        returns = df_ticker.set_index("date")["log_return"].dropna()
        rolling_fit = self.ukf.rolling_fit(returns, rolling_config=self.rolling_config)
        ticker_signal = (
            df_ticker.set_index("date")
            .join(rolling_fit.signal_frame, how="left")
            .sort_index()
        )
        ticker_signal["spread"] = ticker_signal["atm_implied_vol"] - ticker_signal["estimated_realized_vol"]
        allocation_frame = self.allocation_policy.transform(ticker_signal["spread"])
        ticker_signal = ticker_signal.join(allocation_frame, how="left")
        ticker_signal["allocation"] = ticker_signal["allocation"].fillna(
            self.allocation_policy.config.base_allocation
        )
        ticker_signal["ticker"] = ticker

        ticker_params = rolling_fit.parameter_frame.copy()
        ticker_params["ticker"] = ticker
        return SignalResult(
            signal_frame=ticker_signal.reset_index(),
            parameter_frame=ticker_params.reset_index(),
        )


def allocation_from_spread(
    spread: pd.Series,
    config: AllocationConfig | None = None,
) -> pd.DataFrame:
    return DynamicAllocationPolicy(config=config).transform(spread)


def build_signal_panel(
    market_panel: pd.DataFrame,
    ukf_config: HestonUKFConfig | None = None,
    rolling_config: RollingWindowConfig | None = None,
    allocation_config: AllocationConfig | None = None,
) -> SignalResult:
    engine = SpreadSignalEngine(
        ukf_config=ukf_config,
        rolling_config=rolling_config,
        allocation_policy=DynamicAllocationPolicy(config=allocation_config),
    )
    return engine.build_signal_panel(market_panel)
