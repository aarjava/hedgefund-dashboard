"""
Factor attribution utilities using proxy ETFs.
"""

from typing import Dict
import numpy as np
import pandas as pd


def compute_factor_returns(factor_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns for factor proxy prices.
    """
    if factor_prices.empty:
        return pd.DataFrame()
    return factor_prices.pct_change().dropna()


def compute_factor_betas(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 63
) -> pd.DataFrame:
    """
    Rolling OLS betas of returns on factor returns.

    Args:
        returns: Portfolio returns series.
        factor_returns: DataFrame of factor returns.
        window: Rolling window length.

    Returns:
        DataFrame of betas with same index as returns.
    """
    if returns.empty or factor_returns.empty:
        return pd.DataFrame()

    df = pd.concat([returns, factor_returns], axis=1).dropna()
    if df.shape[0] < window:
        return pd.DataFrame()

    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    betas = []
    idx = []
    cols = list(X.columns)

    for i in range(window, len(df) + 1):
        y_win = y.iloc[i - window:i].values
        X_win = X.iloc[i - window:i].values
        # Add intercept
        X_mat = np.column_stack([np.ones(len(X_win)), X_win])
        coef, *_ = np.linalg.lstsq(X_mat, y_win, rcond=None)
        # coef[0] is alpha
        betas.append(coef[1:])
        idx.append(df.index[i - 1])

    beta_df = pd.DataFrame(betas, index=idx, columns=cols)
    return beta_df


def compute_factor_contributions(
    beta_df: pd.DataFrame,
    factor_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute factor contribution time series.
    """
    if beta_df.empty or factor_returns.empty:
        return pd.DataFrame()

    aligned = factor_returns.reindex(beta_df.index).dropna()
    beta_aligned = beta_df.reindex(aligned.index)
    contrib = aligned * beta_aligned
    return contrib


def compute_alpha_series(
    returns: pd.Series,
    market_returns: pd.Series,
    window: int = 63
) -> pd.Series:
    """
    Rolling alpha from regression on market returns.
    """
    df = pd.concat([returns, market_returns], axis=1).dropna()
    if df.shape[0] < window:
        return pd.Series(dtype=float)

    y = df.iloc[:, 0]
    x = df.iloc[:, 1]
    alpha = []
    idx = []

    for i in range(window, len(df) + 1):
        y_win = y.iloc[i - window:i].values
        x_win = x.iloc[i - window:i].values
        X_mat = np.column_stack([np.ones(len(x_win)), x_win])
        coef, *_ = np.linalg.lstsq(X_mat, y_win, rcond=None)
        alpha.append(coef[0])
        idx.append(df.index[i - 1])

    return pd.Series(alpha, index=idx, name="Alpha")
