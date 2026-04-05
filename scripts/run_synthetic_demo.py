from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realized_vol_timing import (
    DynamicAllocationPolicy,
    HestonParams,
    HestonStateSpaceModel,
    HestonUKF,
    SpreadSignalEngine,
)
from realized_vol_timing.config import AllocationConfig, HestonUKFConfig, RollingWindowConfig


def main() -> None:
    fast_config = HestonUKFConfig(optimizer_maxiter=40)
    params = HestonParams(
        mu=0.06,
        kappa=4.0,
        theta=0.05,
        xi=0.65,
        rho=-0.6,
        v0=0.05,
    )
    model = HestonStateSpaceModel(params=params, config=fast_config)
    df_simulated = model.simulate(periods=140, seed=42)
    df_simulated["ticker"] = "SYNTH"
    df_simulated["atm_implied_vol"] = (
        df_simulated["latent_volatility"]
        + 0.03
        + 0.01 * pd.Series(df_simulated["latent_volatility"]).rolling(10, min_periods=1).mean().values
    )

    signal_engine = SpreadSignalEngine(
        ukf=HestonUKF(config=fast_config),
        rolling_config=RollingWindowConfig(window=40, refit_every=20),
        allocation_policy=DynamicAllocationPolicy(config=AllocationConfig()),
    )
    signal_result = signal_engine.build_signal_panel(
        df_simulated[["date", "ticker", "spot", "log_return", "atm_implied_vol"]]
    )
    columns = [
        "date",
        "ticker",
        "atm_implied_vol",
        "estimated_realized_vol",
        "spread",
        "spread_zscore",
        "allocation",
    ]
    print(signal_result.signal_frame[columns].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
