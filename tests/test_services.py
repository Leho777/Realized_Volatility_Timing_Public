import unittest

import pandas as pd

from realized_vol_timing import (
    DynamicAllocationPolicy,
    HestonParams,
    HestonStateSpaceModel,
    HestonUKF,
    SpreadSignalEngine,
    TimedCarryStrategyRunner,
    TradeAllocator,
    summarize_backtests,
)
from realized_vol_timing.config import AllocationConfig, HestonUKFConfig, RollingWindowConfig


class ServiceLayerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fast_config = HestonUKFConfig(optimizer_maxiter=30)
        model = HestonStateSpaceModel(
            params=HestonParams(
                mu=0.04,
                kappa=3.0,
                theta=0.05,
                xi=0.45,
                rho=-0.5,
                v0=0.05,
            ),
            config=self.fast_config,
        )
        self.simulated = model.simulate(periods=60, seed=13)
        self.simulated["ticker"] = "SYNTH"
        self.simulated["atm_implied_vol"] = self.simulated["latent_volatility"] + 0.025

    def test_allocation_policy_is_clipped(self) -> None:
        policy = DynamicAllocationPolicy(
            config=AllocationConfig(min_allocation=0.25, max_allocation=1.5, sensitivity=0.8),
        )
        allocation = policy.transform(pd.Series([-1.0, -0.2, 0.0, 0.2, 1.0]))
        self.assertTrue((allocation["allocation"] >= 0.25).all())
        self.assertTrue((allocation["allocation"] <= 1.5).all())

    def test_signal_engine_builds_expected_columns(self) -> None:
        signal_engine = SpreadSignalEngine(
            ukf=HestonUKF(config=self.fast_config),
            rolling_config=RollingWindowConfig(window=20, refit_every=10),
        )
        signal_result = signal_engine.build_signal_panel(
            self.simulated[["date", "ticker", "spot", "log_return", "atm_implied_vol"]]
        )
        self.assertIn("spread", signal_result.signal_frame.columns)
        self.assertIn("allocation", signal_result.signal_frame.columns)

    def test_trade_allocator_scales_weights(self) -> None:
        signal_engine = SpreadSignalEngine(
            ukf=HestonUKF(config=self.fast_config),
            rolling_config=RollingWindowConfig(window=20, refit_every=10),
        )
        signal_result = signal_engine.build_signal_panel(
            self.simulated[["date", "ticker", "spot", "log_return", "atm_implied_vol"]]
        )
        trades = pd.DataFrame(
            {
                "date": self.simulated["date"].iloc[30:35].tolist(),
                "entry_date": self.simulated["date"].iloc[30:35].tolist(),
                "option_id": [f"OPT{i}" for i in range(5)],
                "leg_name": ["TEST"] * 5,
                "weight": [1.0] * 5,
                "ticker": ["SYNTH"] * 5,
            }
        )
        scaled_trades = TradeAllocator().apply(trades, signal_result.signal_frame)
        self.assertIn("allocation_multiplier", scaled_trades.columns)
        self.assertTrue((scaled_trades["weight"] >= 0.0).all())

    def test_runner_accepts_prebuilt_market_panel(self) -> None:
        class DummyTradeClass:
            @staticmethod
            def generate_trades(*args, **kwargs):
                return pd.DataFrame(
                    {
                        "date": self.simulated["date"].iloc[30:35].tolist(),
                        "entry_date": self.simulated["date"].iloc[30:35].tolist(),
                        "option_id": [f"OPT{i}" for i in range(5)],
                        "leg_name": ["TEST"] * 5,
                        "weight": [1.0] * 5,
                        "ticker": ["SYNTH"] * 5,
                    }
                )

        class DummyBacktester:
            def __init__(self, positions):
                self.positions = positions
                nav = pd.Series([1.0, 1.02, 1.01, 1.03], name="NAV")
                self.nav = nav.to_frame()

            def compute_backtest(self, tcost_args=None):
                return self

        runner = TimedCarryStrategyRunner(
            signal_engine=SpreadSignalEngine(
                ukf=HestonUKF(config=self.fast_config),
                rolling_config=RollingWindowConfig(window=20, refit_every=10),
            ),
            trade_class=DummyTradeClass,
            backtester_class=DummyBacktester,
        )
        result = runner.run_backtest(
            start_date=self.simulated["date"].iloc[0].to_pydatetime(),
            end_date=self.simulated["date"].iloc[-1].to_pydatetime(),
            tickers="SYNTH",
            legs=[{}],
            market_panel=self.simulated[["date", "ticker", "spot", "log_return", "atm_implied_vol"]],
        )
        self.assertIn("total_return", result.comparison.columns)

    def test_summarize_backtests_accepts_multiple_series(self) -> None:
        class DummyBacktester:
            def __init__(self, values):
                self.nav = pd.DataFrame({"NAV": values}, index=pd.date_range("2020-01-01", periods=4))

        summary = summarize_backtests(
            {
                "Base carry": DummyBacktester([1.0, 1.01, 1.02, 1.01]),
                "Timed carry UKF": DummyBacktester([1.0, 1.02, 1.03, 1.04]),
                "Timed carry RV_21d": DummyBacktester([1.0, 1.01, 1.015, 1.02]),
            }
        )

        self.assertEqual(list(summary.index), ["Base carry", "Timed carry UKF", "Timed carry RV_21d"])
        self.assertIn("sharpe_ratio", summary.columns)


if __name__ == "__main__":
    unittest.main()
