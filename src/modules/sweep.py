"""
Parameter sweep utilities for robustness checks.
"""

from typing import Iterable
import pandas as pd
import numpy as np

from . import signals
from . import backtester


def run_sma_regime_sweep(
    df: pd.DataFrame,
    sma_windows: Iterable[int],
    mom_window: int,
    vol_q_high: float,
    use_oos: bool,
    vol_window: int = 21,
) -> pd.DataFrame:
    """
    Sweep SMA windows and compute regime-conditional stats.

    Returns DataFrame with MultiIndex (SMA, Regime) and columns Sharpe/CAGR.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    base = df.copy()
    if "Daily_Return" not in base.columns:
        base["Daily_Return"] = base["Close"].pct_change()

    base[f"Vol_{vol_window}d"] = (
        base["Daily_Return"].rolling(window=vol_window).std() * np.sqrt(252)
    )

    base = signals.detect_volatility_regime(
        base,
        vol_col=f"Vol_{vol_window}d",
        quantile_high=vol_q_high,
        quantile_low=0.25,
        use_expanding=use_oos,
    )

    rows = []
    for sma in sma_windows:
        temp = base.copy()
        temp[f"SMA_{sma}"] = temp["Close"].rolling(window=sma).mean()
        temp["Signal_Trend"] = np.where(temp["Close"] > temp[f"SMA_{sma}"], 1, 0)
        temp = temp.dropna(subset=[f"SMA_{sma}", "Daily_Return", "Vol_Regime"])
        if temp.empty:
            continue

        # Simple strategy returns
        temp["Strategy_Net_Return"] = temp["Signal_Trend"].shift(1).fillna(0) * temp["Daily_Return"]

        stats = backtester.calculate_regime_stats(
            temp, "Strategy_Net_Return", "Vol_Regime"
        )

        if stats.empty:
            continue

        for reg, row in stats.iterrows():
            rows.append({
                "SMA": sma,
                "Regime": reg,
                "Sharpe": row.get("Sharpe", np.nan),
                "CAGR": row.get("CAGR", np.nan),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index(["SMA", "Regime"]).sort_index()
    return out
