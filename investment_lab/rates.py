import numpy as np
import pandas as pd

from investment_lab.constants import DAYS_PER_YEAR, TENOR_TO_PERIOD
from investment_lab.util import check_is_true


def compute_forward(df_options: pd.DataFrame, df_rates: pd.DataFrame) -> pd.DataFrame:
    """Compute risk free rates and forward price given daily options and daily rate curve."""
    missing_cols = set(TENOR_TO_PERIOD.keys()).difference(df_rates.columns)
    check_is_true(
        len(missing_cols) == 0, f"df_rates is missing columns: {missing_cols}"
    )
    missing_options_cols = {
        "date",
        "spot",
        "day_to_expiration",
        "option_id",
    }.difference(df_options.columns)
    check_is_true(
        len(missing_options_cols) == 0,
        f"df_options is missing columns: {missing_options_cols}",
    )

    rate_cols = list(TENOR_TO_PERIOD.keys())
    df = df_options.merge(df_rates, on="date", how="left")
    df_groups = (
        df.groupby(["date", "expiration"], as_index=False)
        .first()
        .loc[:, ["date", "expiration", "day_to_expiration", *rate_cols]]
    )
    tenors = np.array([TENOR_TO_PERIOD[col] for col in rate_cols], dtype=float)
    df_groups["risk_free_rate"] = df_groups.apply(
        lambda row: interpolate_rates(
            row["day_to_expiration"] / DAYS_PER_YEAR,
            tenors=tenors,
            rate_curve=row[rate_cols].to_numpy(dtype=float),
        ),
        axis=1,
    )
    df = df.merge(
        df_groups[["date", "expiration", "risk_free_rate"]],
        on=["date", "expiration"],
        how="left",
    )
    df["forward"] = df["spot"] * np.exp(
        df["risk_free_rate"] * df["day_to_expiration"] / DAYS_PER_YEAR
    )
    df_forward = (
        df.groupby(["ticker", "date", "expiration"])[["forward"]]
        .first()
        .ffill()
        .reset_index()
    )
    return df.drop(columns=rate_cols + ["forward"]).merge(
        df_forward, how="left", on=["ticker", "date", "expiration"]
    )


def interpolate_rates(
    eval_tenor: float,
    tenors: pd.Series | np.ndarray,
    rate_curve: pd.Series | np.ndarray,
) -> float:
    """Interpolate rates linearly."""
    tenors = np.asarray(tenors)
    rate_curve = np.asarray(rate_curve)
    check_is_true(
        len(tenors) == len(rate_curve),
        "Tenors and rate curve must have the same length.",
    )
    if eval_tenor <= tenors.min():
        return rate_curve[tenors.argmin()]
    if eval_tenor >= tenors.max():
        return rate_curve[tenors.argmax()]

    idx_above = tenors[tenors >= eval_tenor].argmin()
    idx_below = tenors[tenors <= eval_tenor].argmax()

    tenor_above, tenor_below = tenors[idx_above], tenors[idx_below]
    rate_above, rate_below = rate_curve[idx_above], rate_curve[idx_below]

    weight_above = (eval_tenor - tenor_below) / (tenor_above - tenor_below)
    weight_below = 1 - weight_above

    return weight_below * rate_below + weight_above * rate_above
