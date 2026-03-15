"""
Regime transition and sensitivity analytics.
"""

from typing import Dict
import numpy as np
import pandas as pd

DEFAULT_REGIME_ORDER = ["Low", "Normal", "High", "Unknown"]


def _perf_stats(returns: pd.Series) -> Dict[str, float]:
    if returns is None or returns.empty:
        return {"Mean": np.nan, "Sharpe": np.nan, "WinRate": np.nan, "CAGR": np.nan}

    clean = returns.dropna()
    if clean.empty:
        return {"Mean": np.nan, "Sharpe": np.nan, "WinRate": np.nan, "CAGR": np.nan}

    mean = clean.mean() * 252
    vol = clean.std() * np.sqrt(252)
    sharpe = mean / vol if vol != 0 else np.nan
    win = (clean > 0).mean()

    years = max(len(clean) / 252, 1e-6)
    cagr = (1 + clean).prod() ** (1 / years) - 1

    return {"Mean": mean, "Sharpe": sharpe, "WinRate": win, "CAGR": cagr}


def compute_transition_matrix(regime_series: pd.Series) -> pd.DataFrame:
    """
    Compute regime transition matrix as row-normalized probabilities.
    """
    if regime_series is None or regime_series.empty:
        return pd.DataFrame()

    reg = regime_series.dropna().astype(str)
    prev = reg.shift(1)
    curr = reg
    trans = pd.crosstab(prev, curr)

    order = [r for r in DEFAULT_REGIME_ORDER if r in trans.index or r in trans.columns]
    trans = trans.reindex(index=order, columns=order, fill_value=0)

    row_sum = trans.sum(axis=1).replace(0, np.nan)
    prob = trans.div(row_sum, axis=0).fillna(0)
    return prob


def compute_transition_stats(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """
    Compute performance stats by transition pair (prev->curr).
    """
    if returns is None or returns.empty:
        return pd.DataFrame()

    df = pd.concat([returns, regimes], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    df.columns = ["Return", "Regime"]
    df["Prev_Regime"] = df["Regime"].shift(1)
    df = df.dropna()
    df["Transition"] = df["Prev_Regime"].astype(str) + "→" + df["Regime"].astype(str)

    rows = []
    for trans, grp in df.groupby("Transition"):
        stats = _perf_stats(grp["Return"])
        rows.append({"Transition": trans, **stats, "Count": len(grp)})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("Transition").sort_index()


def compute_regime_sensitivity(stats_by_regime: pd.DataFrame) -> Dict[str, float]:
    """
    Compute regime sensitivity metrics based on High vs Normal.
    """
    if stats_by_regime is None or stats_by_regime.empty:
        return {"Sharpe_Diff": np.nan, "CAGR_Diff": np.nan}

    if "High" not in stats_by_regime.index or "Normal" not in stats_by_regime.index:
        return {"Sharpe_Diff": np.nan, "CAGR_Diff": np.nan}

    sharpe_diff = stats_by_regime.loc["High", "Sharpe"] - stats_by_regime.loc["Normal", "Sharpe"]
    cagr_diff = stats_by_regime.loc["High", "CAGR"] - stats_by_regime.loc["Normal", "CAGR"]

    return {"Sharpe_Diff": sharpe_diff, "CAGR_Diff": cagr_diff}


def bootstrap_regime_diff(
    returns: pd.Series,
    regimes: pd.Series,
    metric: str = "Sharpe",
    n_boot: int = 500,
) -> Dict[str, float]:
    """
    Bootstrap difference between High and Normal regimes for a metric.
    """
    df = pd.concat([returns, regimes], axis=1).dropna()
    if df.empty:
        return {"diff": np.nan, "p_value": np.nan}

    df.columns = ["Return", "Regime"]
    high = df[df["Regime"] == "High"]["Return"]
    normal = df[df["Regime"] == "Normal"]["Return"]
    if len(high) < 5 or len(normal) < 5:
        return {"diff": np.nan, "p_value": np.nan}

    def _metric(x):
        stats = _perf_stats(x)
        return stats.get(metric, np.nan)

    obs = _metric(high) - _metric(normal)

    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        h = high.sample(len(high), replace=True, random_state=rng.integers(0, 1_000_000))
        n = normal.sample(len(normal), replace=True, random_state=rng.integers(0, 1_000_000))
        diffs.append(_metric(h) - _metric(n))

    diffs = np.array(diffs)
    if np.isnan(obs) or diffs.size == 0:
        return {"diff": np.nan, "p_value": np.nan}

    p_value = (np.abs(diffs) >= abs(obs)).mean()
    return {"diff": obs, "p_value": p_value}
