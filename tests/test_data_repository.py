import unittest
from datetime import datetime, timedelta

import pandas as pd

from investment_lab.data.option_db import AAPLOptionLoader, OptionLoader, SPYOptionLoader
from realized_vol_timing.data import CourseOptionMarketDataRepository


class DataRepositoryTestCase(unittest.TestCase):
    def test_repository_prefers_specialized_local_loaders(self) -> None:
        repository = CourseOptionMarketDataRepository()
        self.assertIs(repository._resolve_option_loader("SPY"), SPYOptionLoader)
        self.assertIs(repository._resolve_option_loader(["AAPL"]), AAPLOptionLoader)
        self.assertIs(repository._resolve_option_loader(["SPY", "AAPL"]), OptionLoader)

    def test_specialized_loader_reinjects_missing_ticker_column(self) -> None:
        df = pd.DataFrame(
            {
                "date": [datetime(2020, 1, 1)],
                "expiration": [datetime(2020, 1, 31)],
                "strike": [100.0],
                "spot": [100.0],
                "mid": [1.0],
                "bid": [0.9],
                "ask": [1.1],
                "volume": [10.0],
                "call_put": ["C"],
                "option_id": ["SPY 20200131C100"],
            }
        )
        processed = SPYOptionLoader._process_loaded_data(df, ticker="SPY")
        self.assertEqual(processed["ticker"].iloc[0], "SPY")

    def test_build_market_panel_from_options_selects_atm_implied_vol(self) -> None:
        base_date = datetime(2020, 1, 1)
        rows = []
        for offset, spot in enumerate([100.0, 102.0, 101.0]):
            current_date = base_date + timedelta(days=offset)
            expiration = current_date + timedelta(days=30)
            rows.extend(
                [
                    {
                        "date": current_date,
                        "ticker": "SYNTH",
                        "spot": spot,
                        "call_put": "C",
                        "day_to_expiration": 30,
                        "moneyness": 1.0,
                        "strike": spot,
                        "implied_volatility": 0.20 + 0.01 * offset,
                        "expiration": expiration,
                        "option_id": f"C{offset}",
                    },
                    {
                        "date": current_date,
                        "ticker": "SYNTH",
                        "spot": spot,
                        "call_put": "P",
                        "day_to_expiration": 30,
                        "moneyness": 1.0,
                        "strike": spot,
                        "implied_volatility": 0.22 + 0.01 * offset,
                        "expiration": expiration,
                        "option_id": f"P{offset}",
                    },
                    {
                        "date": current_date,
                        "ticker": "SYNTH",
                        "spot": spot,
                        "call_put": "C",
                        "day_to_expiration": 10,
                        "moneyness": 1.1,
                        "strike": spot * 1.1,
                        "implied_volatility": 0.50,
                        "expiration": current_date + timedelta(days=10),
                        "option_id": f"NOISE{offset}",
                    },
                ]
            )

        repository = CourseOptionMarketDataRepository(implied_vol_maturity_days=30)
        panel = repository.build_market_panel_from_options(pd.DataFrame(rows))

        self.assertEqual(len(panel), 3)
        self.assertIn("log_return", panel.columns)
        self.assertIn("atm_implied_vol", panel.columns)
        self.assertAlmostEqual(panel.loc[0, "atm_implied_vol"], 0.21)
        self.assertAlmostEqual(panel.loc[1, "atm_implied_vol"], 0.22)
        self.assertAlmostEqual(panel.loc[2, "atm_implied_vol"], 0.23)
        self.assertTrue(panel["date"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
