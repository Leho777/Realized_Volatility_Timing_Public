import pandas as pd


def levels_to_returns(levels: pd.Series) -> pd.Series:
    """Convert a series of price levels to returns."""
    return levels.pct_change()


def returns_to_levels(returns: pd.Series) -> pd.Series:
    """Convert a series of returns to price levels."""
    return (1 + returns).cumprod()
