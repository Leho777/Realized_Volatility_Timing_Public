import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from realized_vol_timing import (
    OutputPaths,
    SignalResult,
    TimedCarryResult,
    export_carry_experiment_outputs,
    export_timed_carry_outputs,
    prepare_output_dirs,
)


class _DummyBacktest:
    def __init__(self) -> None:
        self.nav = pd.DataFrame(
            {"NAV": [1.0, 1.01, 1.02]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        )


class ReportingTestCase(unittest.TestCase):
    def test_prepare_output_dirs_can_ensure_unique_run_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = prepare_output_dirs(base_dir=tmpdir, run_name="test_run", ensure_unique=True)
            second = prepare_output_dirs(base_dir=tmpdir, run_name="test_run", ensure_unique=True)

            self.assertNotEqual(first.root, second.root)
            self.assertTrue(first.root.exists())
            self.assertTrue(second.root.exists())

    def test_export_timed_carry_outputs_creates_expected_files(self) -> None:
        signal_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "ticker": ["SYNTH"] * 3,
                "atm_implied_vol": [0.20, 0.21, 0.19],
                "estimated_realized_vol": [0.18, 0.20, 0.17],
                "spread": [0.02, 0.01, 0.02],
                "allocation": [1.1, 1.0, 1.2],
            }
        )
        parameter_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "ticker": ["SYNTH"] * 3,
                "kappa": [2.0, 2.1, 2.2],
                "theta": [0.04, 0.04, 0.05],
                "xi": [0.5, 0.5, 0.6],
                "rho": [-0.4, -0.4, -0.3],
                "mu": [0.03, 0.03, 0.04],
                "v0": [0.04, 0.04, 0.05],
            }
        )
        result = TimedCarryResult(
            signal_result=SignalResult(signal_frame=signal_frame, parameter_frame=parameter_frame),
            base_trades=pd.DataFrame({"weight": [1.0]}),
            timed_trades=pd.DataFrame({"weight": [1.1]}),
            base_backtest=_DummyBacktest(),
            timed_backtest=_DummyBacktest(),
            comparison=pd.DataFrame(
                {
                    "total_return": [0.01, 0.02],
                    "sharpe_ratio": [0.5, 0.6],
                },
                index=["Base carry", "Timed carry"],
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_timed_carry_outputs(
                result,
                base_dir=tmpdir,
                run_name="test_run",
                metadata={"ticker": "SYNTH"},
            )

            self.assertIsInstance(paths, OutputPaths)
            self.assertTrue((paths.figures / "signal_dashboard.png").exists())
            self.assertTrue((paths.figures / "heston_parameters.png").exists())
            self.assertTrue((paths.figures / "nav_comparison.png").exists())
            self.assertTrue((paths.tables / "signal_frame.csv").exists())
            self.assertTrue((paths.tables / "parameter_frame.csv").exists())
            self.assertTrue((paths.tables / "backtest_comparison.csv").exists())
            self.assertTrue((paths.tables / "nav_comparison.csv").exists())
            self.assertTrue((paths.root / "run_metadata.json").exists())
            self.assertTrue((paths.root / "run_summary.txt").exists())

    def test_export_carry_experiment_outputs_creates_expected_files(self) -> None:
        signal_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "ticker": ["SPY"] * 3,
                "atm_implied_vol": [0.20, 0.21, 0.19],
                "estimated_realized_vol": [0.18, 0.20, 0.17],
                "rv_21d": [0.19, 0.20, 0.18],
                "rv_63d": [0.18, 0.19, 0.18],
                "spread": [0.02, 0.01, 0.02],
                "allocation": [1.1, 1.0, 1.2],
            }
        )
        rv_signal_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "ticker": ["SPY"] * 3,
                "benchmark_realized_vol": [0.19, 0.20, 0.18],
                "spread": [0.01, 0.01, 0.01],
                "allocation": [1.05, 1.0, 1.05],
            }
        )
        parameter_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "ticker": ["SPY"] * 3,
                "kappa": [2.0, 2.1, 2.2],
                "theta": [0.04, 0.04, 0.05],
                "xi": [0.5, 0.5, 0.6],
                "rho": [-0.4, -0.4, -0.3],
                "mu": [0.03, 0.03, 0.04],
                "v0": [0.04, 0.04, 0.05],
            }
        )
        experiment = SimpleNamespace(
            signal_frame=signal_frame,
            parameter_frame=parameter_frame,
            rv_signal_frame=rv_signal_frame,
            comparison_frame=pd.DataFrame(
                {
                    "total_return": [0.01, 0.02, 0.015],
                    "sharpe_ratio": [0.5, 0.6, 0.55],
                },
                index=["Base carry", "Timed carry UKF", "Timed carry RV_21d"],
            ),
            base_trades=pd.DataFrame({"weight": [1.0]}),
            ukf_timed_trades=pd.DataFrame({"weight": [1.1]}),
            rv_timed_trades=pd.DataFrame({"weight": [1.05]}),
            base_backtest=_DummyBacktest(),
            ukf_timed_backtest=_DummyBacktest(),
            rv_timed_backtest=_DummyBacktest(),
        )
        proxy_metrics = pd.DataFrame(
            {
                "corr_vs_rv_21d": [0.8, 0.7],
                "corr_vs_rv_63d": [0.75, 0.65],
            },
            index=["UKF sigma_hat", "IV ATM"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_carry_experiment_outputs(
                experiment,
                base_dir=tmpdir,
                run_name="experiment_run",
                proxy_metrics=proxy_metrics,
                metadata={"ticker": "SPY"},
            )

            self.assertIsInstance(paths, OutputPaths)
            self.assertTrue((paths.figures / "signal_dashboard.png").exists())
            self.assertTrue((paths.figures / "heston_parameters.png").exists())
            self.assertTrue((paths.figures / "nav_comparison.png").exists())
            self.assertTrue((paths.tables / "signal_frame_ukf.csv").exists())
            self.assertTrue((paths.tables / "signal_frame_rv_21d.csv").exists())
            self.assertTrue((paths.tables / "parameter_frame.csv").exists())
            self.assertTrue((paths.tables / "backtest_comparison.csv").exists())
            self.assertTrue((paths.tables / "nav_comparison.csv").exists())
            self.assertTrue((paths.tables / "proxy_metrics.csv").exists())
            self.assertTrue((paths.root / "run_metadata.json").exists())
            self.assertTrue((paths.root / "run_summary.txt").exists())


if __name__ == "__main__":
    unittest.main()
