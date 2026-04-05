from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from investment_lab.data.option_db import AAPLOptionLoader, OptionLoader, SPYOptionLoader
from investment_lab.option_selection import select_options


class MarketDataRepository(ABC):
    @abstractmethod
    def load_market_panel(self, start_date, end_date, tickers: str | list[str]) -> pd.DataFrame:
        raise NotImplementedError


class CourseOptionMarketDataRepository(MarketDataRepository):
    def __init__(
        self,
        option_loader: type[OptionLoader] | None = None,
        implied_vol_maturity_days: int = 30,
    ) -> None:
        self.option_loader = option_loader
        self.implied_vol_maturity_days = implied_vol_maturity_days

    def load_market_panel(self, start_date, end_date, tickers: str | list[str]) -> pd.DataFrame:
        loader = self._resolve_option_loader(tickers)
        df_options = loader.load_data(
            start_date,
            end_date,
            process_kwargs={"ticker": tickers},
        )
        return self.build_market_panel_from_options(df_options)

    def _resolve_option_loader(self, tickers: str | list[str]) -> type[OptionLoader]:
        if self.option_loader is not None:
            return self.option_loader

        if isinstance(tickers, str):
            normalized_tickers = [tickers]
        else:
            normalized_tickers = list(tickers)

        unique_tickers = {ticker.upper() for ticker in normalized_tickers}
        if unique_tickers == {"SPY"}:
            return SPYOptionLoader
        if unique_tickers == {"AAPL"}:
            return AAPLOptionLoader
        return OptionLoader

    def build_market_panel_from_options(self, df_options: pd.DataFrame) -> pd.DataFrame:
        df_options = df_options.copy()
        df_options["date"] = pd.to_datetime(df_options["date"])

        df_spot = (
            df_options.groupby(["date", "ticker"])[["spot"]]
            .first()
            .sort_index()
            .reset_index()
        )
        df_spot["log_return"] = np.log(df_spot["spot"]).groupby(df_spot["ticker"]).diff()

        df_call = select_options(
            df_options.copy(),
            call_or_put="C",
            strike_col="moneyness",
            strike_target=1.0,
            day_to_expiry_target=self.implied_vol_maturity_days,
        )
        df_put = select_options(
            df_options.copy(),
            call_or_put="P",
            strike_col="moneyness",
            strike_target=1.0,
            day_to_expiry_target=self.implied_vol_maturity_days,
        )
        df_implied_vol = (
            pd.concat(
                [
                    df_call[["date", "ticker", "implied_volatility"]],
                    df_put[["date", "ticker", "implied_volatility"]],
                ],
                ignore_index=True,
            )
            .groupby(["date", "ticker"])[["implied_volatility"]]
            .mean()
            .rename(columns={"implied_volatility": "atm_implied_vol"})
            .reset_index()
        )

        return (
            df_spot.merge(df_implied_vol, on=["date", "ticker"], how="left")
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )


def load_market_panel(
    start_date,
    end_date,
    tickers: str | list[str],
    implied_vol_maturity_days: int = 30,
) -> pd.DataFrame:
    repository = CourseOptionMarketDataRepository(
        implied_vol_maturity_days=implied_vol_maturity_days,
    )
    return repository.load_market_panel(start_date, end_date, tickers)


def build_market_panel_from_options(
    df_options: pd.DataFrame,
    implied_vol_maturity_days: int = 30,
) -> pd.DataFrame:
    repository = CourseOptionMarketDataRepository(
        implied_vol_maturity_days=implied_vol_maturity_days,
    )
    return repository.build_market_panel_from_options(df_options)
