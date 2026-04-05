from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from investment_lab.stochastic.base import StochasticProcess

from realized_vol_timing.config import HestonUKFConfig, RollingWindowConfig


@dataclass(frozen=True)
class HestonParams:
    mu: float
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.mu, self.kappa, self.theta, self.xi, self.rho, self.v0],
            dtype=float,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "mu": self.mu,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "v0": self.v0,
        }

    @classmethod
    def from_array(cls, values: Iterable[float]) -> "HestonParams":
        mu, kappa, theta, xi, rho, v0 = [float(v) for v in values]
        return cls(mu=mu, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)


@dataclass
class HestonFilterResult:
    history: pd.DataFrame
    total_log_likelihood: float


@dataclass
class HestonMLEFit:
    params: HestonParams
    filter_result: HestonFilterResult
    optimization_result: OptimizeResult


@dataclass
class RollingCalibrationResult:
    signal_frame: pd.DataFrame
    parameter_frame: pd.DataFrame


class HestonUKF:
    _DEFAULT_BOUNDS = (
        (-1.0, 1.0),
        (1e-4, 15.0),
        (1e-6, 4.0),
        (1e-4, 5.0),
        (-0.999, 0.999),
        (1e-6, 4.0),
    )

    def __init__(self, config: HestonUKFConfig | None = None) -> None:
        self.config = config or HestonUKFConfig()

    def filter(self, returns: pd.Series, params: HestonParams) -> HestonFilterResult:
        series = pd.Series(returns).dropna().astype(float)
        mean = max(params.v0, self.config.variance_floor)
        covariance = max(self.config.initial_covariance, self.config.covariance_floor)
        rows: list[dict[str, float]] = []
        total_log_likelihood = 0.0

        for timestamp, observed_return in series.items():
            predicted = self._predict_joint(mean, covariance, params)
            innovation = observed_return - predicted["predicted_return"]
            kalman_gain = predicted["cross_covariance"] / predicted["innovation_variance"]
            mean = max(
                predicted["predicted_variance"] + kalman_gain * innovation,
                self.config.variance_floor,
            )
            covariance = max(
                predicted["predicted_covariance"]
                - (kalman_gain**2) * predicted["innovation_variance"],
                self.config.covariance_floor,
            )
            log_likelihood = -0.5 * (
                np.log(2.0 * np.pi * predicted["innovation_variance"])
                + (innovation**2) / predicted["innovation_variance"]
            )
            total_log_likelihood += log_likelihood
            rows.append(
                {
                    "filtered_variance": mean,
                    "filtered_volatility": np.sqrt(mean),
                    "predicted_variance": predicted["predicted_variance"],
                    "predicted_return": predicted["predicted_return"],
                    "innovation": innovation,
                    "innovation_variance": predicted["innovation_variance"],
                    "log_likelihood": log_likelihood,
                }
            )

        history = pd.DataFrame(rows, index=series.index)
        return HestonFilterResult(history=history, total_log_likelihood=total_log_likelihood)

    def negative_log_likelihood(self, values: np.ndarray, returns: pd.Series) -> float:
        params = HestonParams.from_array(values)
        if params.theta <= 0 or params.xi <= 0 or params.v0 <= 0 or abs(params.rho) >= 1:
            return 1e12

        penalty = max(0.0, params.xi**2 - 2.0 * params.kappa * params.theta)
        try:
            result = self.filter(returns, params)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return 1e12

        objective = -result.total_log_likelihood + 10.0 * penalty
        if not np.isfinite(objective):
            return 1e12
        return float(objective)

    def fit_mle(
        self,
        returns: pd.Series,
        initial_guess: HestonParams | np.ndarray | None = None,
        bounds: tuple[tuple[float, float], ...] | None = None,
    ) -> HestonMLEFit:
        series = pd.Series(returns).dropna().astype(float)
        if len(series) < 5:
            raise ValueError("At least five returns are required to calibrate the Heston UKF.")

        if initial_guess is None:
            x0 = self._default_initial_guess(series)
        elif isinstance(initial_guess, HestonParams):
            x0 = initial_guess.to_array()
        else:
            x0 = np.asarray(initial_guess, dtype=float)

        current_bounds = bounds or self._DEFAULT_BOUNDS
        clipped_x0 = np.array(
            [np.clip(value, lower, upper) for value, (lower, upper) in zip(x0, current_bounds)],
            dtype=float,
        )
        optimization_result = minimize(
            self.negative_log_likelihood,
            x0=clipped_x0,
            args=(series,),
            method=self.config.optimizer_method,
            bounds=current_bounds,
            options={
                "maxiter": self.config.optimizer_maxiter,
                "ftol": self.config.optimizer_ftol,
            },
        )
        params = HestonParams.from_array(optimization_result.x)
        filter_result = self.filter(series, params)
        return HestonMLEFit(
            params=params,
            filter_result=filter_result,
            optimization_result=optimization_result,
        )

    def rolling_fit(
        self,
        returns: pd.Series,
        rolling_config: RollingWindowConfig | None = None,
    ) -> RollingCalibrationResult:
        series = pd.Series(returns).dropna().astype(float)
        cfg = rolling_config or RollingWindowConfig()
        min_periods = cfg.min_periods or cfg.window
        if len(series) < min_periods:
            raise ValueError("The return series is too short for the requested rolling window.")

        signal_rows: list[dict[str, float]] = []
        parameter_rows: list[dict[str, float]] = []
        warm_start: HestonParams | None = None

        for end in range(min_periods, len(series) + 1):
            window_series = series.iloc[max(0, end - cfg.window) : end]
            should_refit = (
                warm_start is None
                or (end == min_periods)
                or ((end - min_periods) % max(cfg.refit_every, 1) == 0)
            )

            if should_refit:
                fit = self.fit_mle(
                    window_series,
                    initial_guess=warm_start if cfg.warm_start else None,
                )
                warm_start = fit.params
                filter_result = fit.filter_result
                optimization_result = fit.optimization_result
            else:
                filter_result = self.filter(window_series, warm_start)
                optimization_result = OptimizeResult(
                    success=True,
                    message="Skipped optimization, reused warm-start parameters.",
                    fun=-filter_result.total_log_likelihood,
                    nit=0,
                )

            timestamp = window_series.index[-1]
            latest = filter_result.history.iloc[-1]
            signal_rows.append(
                {
                    "date": timestamp,
                    "estimated_variance": latest["filtered_variance"],
                    "estimated_realized_vol": latest["filtered_volatility"],
                    "predicted_return": latest["predicted_return"],
                    "innovation": latest["innovation"],
                    "window_log_likelihood": filter_result.total_log_likelihood,
                }
            )
            parameter_rows.append(
                {
                    "date": timestamp,
                    **warm_start.to_dict(),
                    "optimization_success": bool(optimization_result.success),
                    "objective": float(optimization_result.fun),
                    "iterations": int(getattr(optimization_result, "nit", 0)),
                }
            )

        signal_frame = pd.DataFrame(signal_rows).set_index("date")
        parameter_frame = pd.DataFrame(parameter_rows).set_index("date")
        return RollingCalibrationResult(
            signal_frame=signal_frame,
            parameter_frame=parameter_frame,
        )

    def _predict_joint(
        self,
        mean: float,
        covariance: float,
        params: HestonParams,
    ) -> dict[str, float]:
        augmented_mean = np.array([mean, 0.0, 0.0], dtype=float)
        augmented_covariance = np.array(
            [
                [max(covariance, self.config.covariance_floor), 0.0, 0.0],
                [0.0, 1.0, params.rho],
                [0.0, params.rho, 1.0],
            ],
            dtype=float,
        )
        sigma_points, weights_mean, weights_cov = self._sigma_points(
            augmented_mean,
            augmented_covariance,
        )
        next_variances = np.empty(len(sigma_points))
        next_returns = np.empty(len(sigma_points))

        for idx, sigma_point in enumerate(sigma_points):
            variance = max(float(sigma_point[0]), self.config.variance_floor)
            epsilon_price = float(sigma_point[1])
            epsilon_variance = float(sigma_point[2])
            variance_sqrt = np.sqrt(variance * self.config.dt)
            next_variances[idx] = max(
                variance
                + params.kappa * (params.theta - variance) * self.config.dt
                + params.xi * variance_sqrt * epsilon_variance,
                self.config.variance_floor,
            )
            next_returns[idx] = (
                (params.mu - 0.5 * variance) * self.config.dt
                + variance_sqrt * epsilon_price
            )

        predicted_variance = float(np.sum(weights_mean * next_variances))
        predicted_return = float(np.sum(weights_mean * next_returns))
        diff_variance = next_variances - predicted_variance
        diff_return = next_returns - predicted_return
        predicted_covariance = float(
            np.sum(weights_cov * diff_variance * diff_variance)
            + self.config.covariance_floor
        )
        innovation_variance = float(
            np.sum(weights_cov * diff_return * diff_return)
            + self.config.covariance_floor
        )
        cross_covariance = float(np.sum(weights_cov * diff_variance * diff_return))
        return {
            "predicted_variance": max(predicted_variance, self.config.variance_floor),
            "predicted_covariance": predicted_covariance,
            "predicted_return": predicted_return,
            "innovation_variance": innovation_variance,
            "cross_covariance": cross_covariance,
        }

    def _sigma_points(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = mean.shape[0]
        lambda_value = (
            self.config.sigma_point_alpha**2
            * (n + self.config.sigma_point_kappa)
            - n
        )
        scaling = n + lambda_value
        chol = np.linalg.cholesky(self._stabilize_covariance(covariance) * scaling)
        sigma_points = [mean]
        for idx in range(n):
            sigma_points.append(mean + chol[:, idx])
            sigma_points.append(mean - chol[:, idx])

        weights_mean = np.full(2 * n + 1, 1.0 / (2.0 * scaling))
        weights_cov = np.full(2 * n + 1, 1.0 / (2.0 * scaling))
        weights_mean[0] = lambda_value / scaling
        weights_cov[0] = (
            lambda_value / scaling
            + 1.0
            - self.config.sigma_point_alpha**2
            + self.config.sigma_point_beta
        )
        return np.asarray(sigma_points), weights_mean, weights_cov

    def _default_initial_guess(self, returns: pd.Series) -> np.ndarray:
        annualized_variance = max(returns.var() / self.config.dt, 0.04)
        annualized_mean = returns.mean() / self.config.dt
        return np.array(
            [
                annualized_mean,
                3.0,
                annualized_variance,
                0.7,
                -0.5,
                annualized_variance,
            ],
            dtype=float,
        )

    def _stabilize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        stabilized = np.array(covariance, dtype=float, copy=True)
        jitter = self.config.covariance_floor
        for _ in range(6):
            try:
                np.linalg.cholesky(stabilized)
                return stabilized
            except np.linalg.LinAlgError:
                stabilized = stabilized + np.eye(stabilized.shape[0]) * jitter
                jitter *= 10.0
        raise np.linalg.LinAlgError("Unable to stabilize covariance matrix for UKF sigma points.")


class HestonStateSpaceModel(StochasticProcess):
    def __init__(
        self,
        params: HestonParams | None = None,
        config: HestonUKFConfig | None = None,
        filter_engine: HestonUKF | None = None,
    ) -> None:
        self.config = config or HestonUKFConfig()
        self.filter_engine = filter_engine or HestonUKF(config=self.config)
        self.params = params

    def simulate(
        self,
        periods: int = 252,
        start_date: str = "2020-01-01",
        spot0: float = 100.0,
        seed: int = 0,
    ) -> pd.DataFrame:
        if self.params is None:
            raise ValueError("Model parameters must be set before simulation.")

        from realized_vol_timing.synthetic import simulate_heston_series

        return simulate_heston_series(
            params=self.params,
            periods=periods,
            start_date=start_date,
            spot0=spot0,
            seed=seed,
            config=self.config,
        )

    def calibrate(
        self,
        returns: pd.Series,
        initial_guess: HestonParams | np.ndarray | None = None,
    ) -> HestonMLEFit:
        fit = self.filter_engine.fit_mle(returns, initial_guess=initial_guess)
        self.params = fit.params
        return fit

    def filter(self, returns: pd.Series) -> HestonFilterResult:
        if self.params is None:
            raise ValueError("Model parameters must be calibrated or provided before filtering.")
        return self.filter_engine.filter(returns, self.params)

    def rolling_calibrate(
        self,
        returns: pd.Series,
        rolling_config: RollingWindowConfig | None = None,
    ) -> RollingCalibrationResult:
        return self.filter_engine.rolling_fit(returns, rolling_config=rolling_config)
