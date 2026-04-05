import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from investment_lab.option_trade import OptionTradeABC


class _DummyOptionTrade(OptionTradeABC):
    @classmethod
    def load_data(cls, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        current_date = datetime(2019, 1, 2)
        expiration = current_date + timedelta(days=7)
        return pd.DataFrame(
            {
                "date": [current_date, current_date],
                "ticker": ["AAPL", "AAPL"],
                "option_id": ["AAPL_P", "AAPL_C"],
                "expiration": [expiration, expiration],
                "delta": [-0.2, 0.2],
                "strike": [95.0, 105.0],
                "moneyness": [0.95, 1.05],
                "call_put": ["P", "C"],
                "spot": [100.0, 100.0],
                "mid": [1.0, 1.2],
                "day_to_expiration": [7, 7],
                "implied_volatility": [0.25, 0.24],
            }
        )


class OptionTradeFallbackTestCase(unittest.TestCase):
    def test_generate_trades_skips_forward_enrichment_when_rates_are_unavailable(self) -> None:
        with patch(
            "investment_lab.option_trade.USRatesLoader.load_data",
            side_effect=ValueError("Data is only available between 2020-01-02 and 2023-12-30"),
        ):
            trades = _DummyOptionTrade._generate_trades(
                start_date=datetime(2019, 1, 2),
                end_date=datetime(2019, 1, 4),
                tickers="AAPL",
                legs=[
                    {
                        "call_or_put": "P",
                        "strike_col": "moneyness",
                        "strike_target": 0.95,
                        "day_to_expiry_target": 7,
                        "weight": -1.0,
                        "leg_name": "Short Put",
                        "rebal_week_day": [2],
                    }
                ],
            )

        self.assertGreater(len(trades), 0)
        self.assertNotIn("risk_free_rate", trades.columns)
        self.assertNotIn("forward", trades.columns)


if __name__ == "__main__":
    unittest.main()
