"""
Risk analytics utilities.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_var_cvar(returns: pd.Series, level: float = 0.95) -> Dict[str, float]:
    """
    Compute historical VaR and CVaR.

    Args:
        returns: Series of returns.
        level: Confidence level (e.g., 0.95).

    Returns:
        Dict with VaR and CVaR.
    """
    if returns.empty:
        return {"VaR": np.nan, "CVaR": np.nan}

    clean = returns.dropna()
    if clean.empty:
        return {"VaR": np.nan, "CVaR": np.nan}

    var = np.quantile(clean, 1 - level)
    cvar = clean[clean <= var].mean() if (clean <= var).any() else np.nan
    return {"VaR": var, "CVaR": cvar}


def compute_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Compute beta to benchmark returns.
    """
    df = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if df.shape[0] < 2:
        return np.nan
    cov = df.iloc[:, 0].cov(df.iloc[:, 1])
    var = df.iloc[:, 1].var()
    return cov / var if var != 0 else np.nan


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Compute drawdown series from equity curve.
    """
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown.
    """
    dd = compute_drawdown_series(equity)
    return dd.min() if not dd.empty else np.nan


def compute_rolling_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Rolling volatility (annualized).
    """
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.rolling(window).std() * np.sqrt(252)


def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    if returns_df.empty:
        return pd.DataFrame()
    return returns_df.corr()


def risk_posture_score(
    ann_vol: float,
    max_dd: float,
    beta: float,
    days_to_liquidate: float,
    vol_target: float = 0.15,
    dd_target: float = -0.15,
    beta_target: float = 1.0,
    dttl_target: float = 5.0,
) -> float:
    """
    Simple 0-100 risk posture score.
    """
    score = 100.0

    if not np.isnan(ann_vol):
        score -= min(40, max(0, (ann_vol / vol_target - 1) * 20))
    if not np.isnan(max_dd):
        score -= min(30, max(0, (abs(max_dd) / abs(dd_target) - 1) * 15))
    if not np.isnan(beta):
        score -= min(20, max(0, (abs(beta) / beta_target - 1) * 10))
    if not np.isnan(days_to_liquidate):
        score -= min(10, max(0, (days_to_liquidate / dttl_target - 1) * 5))

    return float(max(0, min(100, score)))
