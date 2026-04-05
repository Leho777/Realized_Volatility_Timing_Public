from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from investment_lab.metrics.distance import mse
from investment_lab.surface.base import VolSmoother
from investment_lab.util import check_is_true


class SVISmoother(VolSmoother):
    def __init__(
        self, initial_params: tuple[float, float, float, float, float]
    ) -> None:
        check_is_true(
            len(initial_params) == 5,
            "Initial parameters must be a tuple of (a, b, rho, m, sigma).",
        )
        super().__init__(initial_params)

    def _fit(
        self,
        forward: float | pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray | float,
        market_implied_vols: pd.Series | np.ndarray,
        **kwargs,
    ) -> Self:
        def objective(params: tuple[float, float, float, float, float]) -> float:
            self._params = params
            model_vol = self.transform(
                forward=forward, strike=strike, time_to_maturities=time_to_maturities
            )
            return mse(market_implied_vols, model_vol)

        bounds = [(1e-6, None), (0, None), (-0.999, 0.999), (None, None), (1e-6, None)]

        def no_arbitrage_constraint(
            params: tuple[float, float, float, float, float],
        ) -> float:
            a, b, rho, m, sigma = params
            forward_array = np.asarray(forward)
            strike_array = np.asarray(strike)
            forward_log_moneyness = np.log(strike_array / forward_array)
            total_variance = a + b * (
                rho * (forward_log_moneyness - m)
                + np.sqrt((forward_log_moneyness - m) ** 2 + sigma**2)
            )
            return np.min(total_variance)

        result = minimize(
            objective,
            self._params,
            method="L-BFGS-B",
            bounds=bounds,
            constraints={"type": "ineq", "fun": no_arbitrage_constraint},
            options={"maxiter": 1000, "disp": False},
        )
        self._params = result.x
        return self

    def _transform(
        self,
        forward: pd.Series | np.ndarray,
        strike: pd.Series | np.ndarray,
        time_to_maturities: pd.Series | np.ndarray,
        **kwargs,
    ) -> pd.Series | np.ndarray:
        a, b, rho, m, sigma = self._params
        forward = np.asarray(forward)
        strike = np.asarray(strike)
        forward_log_moneyness = np.log(strike / forward)
        time_to_maturities = np.asarray(time_to_maturities)

        total_variance = a + b * (
            rho * (forward_log_moneyness - m)
            + np.sqrt((forward_log_moneyness - m) ** 2 + sigma**2)
        )
        total_variance = np.maximum(total_variance, 0)
        return np.sqrt(total_variance / time_to_maturities)
