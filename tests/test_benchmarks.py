import unittest
from datetime import datetime

import pandas as pd

from realized_vol_timing import (
    AllocationConfig,
    add_realized_volatility_benchmarks,
    build_realized_volatility_signal_panel,
)


class BenchmarkHelpersTestCase(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2020-01-01", periods=8, freq="B")
        self.market_panel = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["SYNTH"] * len(dates),
                "spot": [100, 101, 100, 102, 101, 103, 102, 104],
                "log_return": [None, 0.01, -0.009, 0.018, -0.009, 0.019, -0.01, 0.019],
                "atm_implied_vol": [0.20, 0.205, 0.21, 0.208, 0.212, 0.215, 0.214, 0.216],
            }
        )

    def test_add_realized_volatility_benchmarks_adds_requested_columns(self) -> None:
        panel = add_realized_volatility_benchmarks(
            self.market_panel,
            windows=(3, 5),
        )

        self.assertIn("rv_3d", panel.columns)
        self.assertIn("rv_5d", panel.columns)
        self.assertGreater(panel["rv_3d"].notna().sum(), 0)
        self.assertGreater(panel["rv_5d"].notna().sum(), 0)

    def test_build_realized_volatility_signal_panel_produces_spread_and_allocation(self) -> None:
        panel = add_realized_volatility_benchmarks(self.market_panel, windows=(3,))
        signal_panel = build_realized_volatility_signal_panel(
            panel,
            rv_column="rv_3d",
            allocation_config=AllocationConfig(base_allocation=1.0, sensitivity=0.25),
        )

        self.assertIn("benchmark_realized_vol", signal_panel.columns)
        self.assertIn("spread", signal_panel.columns)
        self.assertIn("allocation", signal_panel.columns)
        self.assertIn("benchmark_name", signal_panel.columns)
        non_null_spread = signal_panel["spread"].dropna()
        self.assertGreater(len(non_null_spread), 0)


if __name__ == "__main__":
    unittest.main()
