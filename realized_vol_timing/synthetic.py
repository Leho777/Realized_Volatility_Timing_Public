from __future__ import annotations

import numpy as np
import pandas as pd

from realized_vol_timing.config import HestonUKFConfig
from realized_vol_timing.heston_ukf import HestonParams


def simulate_heston_series(
    params: HestonParams,
    periods: int = 252,
    start_date: str = "2020-01-01",
    spot0: float = 100.0,
    seed: int = 0,
    config: HestonUKFConfig | None = None,
) -> pd.DataFrame:
    cfg = config or HestonUKFConfig()
    rng = np.random.default_rng(seed)
    correlated_normals = rng.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[1.0, params.rho], [params.rho, 1.0]],
        size=periods,
    )
    dates = pd.bdate_range(start=start_date, periods=periods)
    variances = np.empty(periods)
    volatilities = np.empty(periods)
    log_returns = np.empty(periods)
    spots = np.empty(periods)

    variance = max(params.v0, cfg.variance_floor)
    spot = spot0
    for idx in range(periods):
        epsilon_price, epsilon_variance = correlated_normals[idx]
        current_variance = max(variance, cfg.variance_floor)
        variance_sqrt = np.sqrt(current_variance * cfg.dt)
        log_returns[idx] = (
            (params.mu - 0.5 * current_variance) * cfg.dt
            + variance_sqrt * epsilon_price
        )
        variance = max(
            current_variance
            + params.kappa * (params.theta - current_variance) * cfg.dt
            + params.xi * variance_sqrt * epsilon_variance,
            cfg.variance_floor,
        )
        spot = spot * np.exp(log_returns[idx])
        variances[idx] = variance
        volatilities[idx] = np.sqrt(variance)
        spots[idx] = spot

    return pd.DataFrame(
        {
            "date": dates,
            "spot": spots,
            "log_return": log_returns,
            "latent_variance": variances,
            "latent_volatility": volatilities,
        }
    )
