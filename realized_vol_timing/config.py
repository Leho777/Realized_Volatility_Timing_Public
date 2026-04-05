from dataclasses import dataclass


@dataclass(frozen=True)
class HestonUKFConfig:
    dt: float = 1.0 / 252.0
    sigma_point_alpha: float = 0.1
    sigma_point_beta: float = 2.0
    sigma_point_kappa: float = 0.0
    variance_floor: float = 1e-8
    covariance_floor: float = 1e-8
    initial_covariance: float = 0.05
    optimizer_method: str = "L-BFGS-B"
    optimizer_maxiter: int = 200
    optimizer_ftol: float = 1e-9


@dataclass(frozen=True)
class RollingWindowConfig:
    window: int = 63
    refit_every: int = 5
    min_periods: int | None = None
    warm_start: bool = True


@dataclass(frozen=True)
class AllocationConfig:
    zscore_window: int = 21
    base_allocation: float = 1.0
    sensitivity: float = 0.35
    min_allocation: float = 0.0
    max_allocation: float = 2.0
    zscore_clip: float = 3.0
    ewm_span: int | None = 5
