import unittest

from realized_vol_timing import HestonParams, HestonStateSpaceModel, HestonUKF
from realized_vol_timing.config import HestonUKFConfig, RollingWindowConfig


class HestonUKFTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.params = HestonParams(
            mu=0.05,
            kappa=3.5,
            theta=0.04,
            xi=0.55,
            rho=-0.4,
            v0=0.04,
        )
        self.fast_config = HestonUKFConfig(optimizer_maxiter=30)
        self.model = HestonStateSpaceModel(params=self.params, config=self.fast_config)
        self.df_simulated = self.model.simulate(periods=90, seed=7)
        self.ukf = HestonUKF(config=self.fast_config)

    def test_filter_keeps_positive_variance(self) -> None:
        result = self.ukf.filter(self.df_simulated["log_return"], self.params)
        self.assertTrue((result.history["filtered_variance"] > 0).all())
        self.assertEqual(len(result.history), len(self.df_simulated))

    def test_model_calibration_updates_parameters(self) -> None:
        fit = self.model.calibrate(self.df_simulated["log_return"].iloc[:60])
        self.assertTrue(fit.optimization_result.success or fit.optimization_result.fun < 1e6)
        self.assertGreater(self.model.params.theta, 0.0)
        self.assertGreater(self.model.params.xi, 0.0)
        self.assertGreater(self.model.params.v0, 0.0)

    def test_rolling_fit_outputs_signal_frame(self) -> None:
        rolling = self.ukf.rolling_fit(
            self.df_simulated["log_return"],
            rolling_config=RollingWindowConfig(window=25, refit_every=10),
        )
        self.assertIn("estimated_realized_vol", rolling.signal_frame.columns)
        self.assertIn("theta", rolling.parameter_frame.columns)
        self.assertTrue((rolling.signal_frame["estimated_variance"] > 0).all())


if __name__ == "__main__":
    unittest.main()
