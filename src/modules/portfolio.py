"""
Portfolio construction and analytics utilities.
"""

from typing import Iterable, Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize_weights(tickers: Iterable[str], weights: Iterable[float]) -> pd.Series:
    """
    Normalize weights to sum to 1.0.

    Args:
        tickers: Iterable of ticker symbols.
        weights: Iterable of weights.

    Returns:
        pd.Series with tickers as index and normalized weights.
    """
    tickers = list(tickers)
    weights = pd.Series(list(weights), index=tickers, dtype=float)
    total = weights.sum()
    if total == 0:
        logger.warning("Weights sum to zero, defaulting to equal weights")
        weights = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    else:
        weights = weights / total
    return weights


def compute_portfolio_returns(price_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Compute portfolio daily returns from price dataframe and weights.

    Args:
        price_df: DataFrame of prices with tickers as columns.
        weights: Series of weights indexed by tickers.

    Returns:
        Series of portfolio returns.
    """
    if price_df.empty:
        return pd.Series(dtype=float)

    aligned_weights = weights.reindex(price_df.columns).fillna(0.0)
    returns = price_df.pct_change().dropna()
    portfolio_returns = returns.mul(aligned_weights, axis=1).sum(axis=1)
    return portfolio_returns


def compute_contributions(price_df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Compute daily contribution of each asset to portfolio return.

    Args:
        price_df: Price dataframe.
        weights: Normalized weights.

    Returns:
        DataFrame of contributions by asset.
    """
    if price_df.empty:
        return pd.DataFrame()

    aligned_weights = weights.reindex(price_df.columns).fillna(0.0)
    returns = price_df.pct_change().dropna()
    contributions = returns.mul(aligned_weights, axis=1)
    return contributions


def build_portfolio(price_df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Build a portfolio dataframe with returns and equity curve.

    Args:
        price_df: DataFrame of prices.
        weights: Normalized weights.

    Returns:
        DataFrame with portfolio returns and equity curve.
    """
    if price_df.empty:
        return pd.DataFrame()

    port_rets = compute_portfolio_returns(price_df, weights)
    equity = (1 + port_rets).cumprod()
    out = pd.DataFrame({
        "Portfolio_Return": port_rets,
        "Portfolio_Equity": equity,
    })
    return out
