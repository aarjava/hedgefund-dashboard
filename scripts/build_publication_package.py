#!/usr/bin/env python3
"""Build a publication-grade, synchronized paper package.

Outputs are generated from one metric source to keep Markdown/PDF/LaTeX aligned.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import sys

REPO = Path("/Users/aarjavametha/Desktop/Projects/hedgefund-dashboard")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.modules import backtester, regime_analysis, signals, sweep  # noqa: E402
from src.modules.config import (  # noqa: E402
    DEFAULT_BOOTSTRAP_ITER,
    DEFAULT_COST_BPS,
    DEFAULT_REBALANCE_FREQ,
    DEFAULT_SMA_SWEEP,
    DEFAULT_SMA_WINDOW,
    DEFAULT_VOLATILITY_WINDOW,
    DEFAULT_VOL_QUANTILE_HIGH,
    DEFAULT_VOL_QUANTILE_LOW,
)


PALETTE = {
    "bg": "#f5f3ee",
    "panel": "#ffffff",
    "grid": "#d9d3c8",
    "text": "#171a24",
    "muted": "#6b7280",
    "strategy": "#2f6fed",
    "benchmark": "#8b95a5",
    "low": "#2ea972",
    "normal": "#d9a441",
    "high": "#d45454",
}


@dataclass
class AssetResults:
    raw: pd.DataFrame
    df: pd.DataFrame
    bt: pd.DataFrame
    strat_metrics: dict
    bench_metrics: dict
    strat_regime: pd.DataFrame
    bench_regime: pd.DataFrame
    avg_vol: pd.Series
    regime_freq: pd.Series
    transition_matrix: pd.DataFrame
    transition_stats: pd.DataFrame
    sensitivity: dict
    bootstrap_hn: dict
    walk_forward: dict
    sweep_df: pd.DataFrame


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.55,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "savefig.dpi": 300,
            "savefig.facecolor": PALETTE["bg"],
        }
    )


def ann_sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    ann_return = clean.mean() * 252
    ann_vol = clean.std() * np.sqrt(252)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return float(ann_return / ann_vol)


def cagr_from_returns(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    years = max(len(clean) / 252, 1e-6)
    return float((1 + clean).prod() ** (1 / years) - 1)


def bootstrap_diff_ci(
    x: pd.Series,
    y: pd.Series,
    stat_fn,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    x = x.dropna()
    y = y.dropna()
    if len(x) < 10 or len(y) < 10:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    obs = stat_fn(x) - stat_fn(y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        xs = x.sample(len(x), replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
        ys = y.sample(len(y), replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
        diffs[i] = stat_fn(xs) - stat_fn(ys)

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    p_value = float((np.abs(diffs) >= abs(obs)).mean())
    return {"diff": float(obs), "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value}


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{x*100:.2f}%"


def num(x: float, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{x:.{d}f}"


def fetch_or_load_snapshot(ticker: str, snapshot_raw_dir: Path, refresh: bool) -> pd.DataFrame:
    path = snapshot_raw_dir / f"{ticker}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df.index = pd.to_datetime(df.index)
        return df

    df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    snapshot_raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = signals.add_technical_indicators(
        df,
        sma_window=DEFAULT_SMA_WINDOW,
        mom_window=12,
        vol_window=DEFAULT_VOLATILITY_WINDOW,
    )
    out[f"SMA_200"] = out["Close"].rolling(200).mean()
    out = signals.detect_volatility_regime(
        out,
        vol_col=f"Vol_{DEFAULT_VOLATILITY_WINDOW}d",
        quantile_high=DEFAULT_VOL_QUANTILE_HIGH,
        quantile_low=DEFAULT_VOL_QUANTILE_LOW,
        use_expanding=True,
    )
    out["Signal_Trend"] = np.where(out["Close"] > out[f"SMA_{DEFAULT_SMA_WINDOW}"], 1, 0)
    out["Signal_Trend_200"] = np.where(out["Close"] > out["SMA_200"], 1, 0)
    return out


def compute_asset(raw: pd.DataFrame) -> AssetResults:
    df = prep(raw)
    df = df.dropna(subset=["Close", "Daily_Return", f"SMA_{DEFAULT_SMA_WINDOW}"])

    bt = backtester.run_backtest(
        df,
        signal_col="Signal_Trend",
        cost_bps=DEFAULT_COST_BPS / 10000,
        rebalance_freq=DEFAULT_REBALANCE_FREQ,
    )

    strat_metrics = backtester.calculate_perf_metrics(
        bt["Equity_Strategy"],
        include_bootstrap_ci=True,
        n_bootstrap=DEFAULT_BOOTSTRAP_ITER,
    )
    bench_metrics = backtester.calculate_perf_metrics(bt["Equity_Benchmark"])

    valid = bt["Vol_Regime"].isin(["Low", "Normal", "High"])
    bt_valid = bt[valid].copy()

    strat_regime = backtester.calculate_regime_stats(bt_valid, "Strategy_Net_Return", "Vol_Regime")
    bench_regime = backtester.calculate_regime_stats(bt_valid, "Daily_Return", "Vol_Regime")

    avg_vol = (
        df.loc[df["Vol_Regime"].isin(["Low", "Normal", "High"]), [f"Vol_{DEFAULT_VOLATILITY_WINDOW}d", "Vol_Regime"]]
        .groupby("Vol_Regime")[f"Vol_{DEFAULT_VOLATILITY_WINDOW}d"]
        .mean()
    )

    regime_freq = df["Vol_Regime"].value_counts(normalize=True)
    regime_freq = regime_freq.loc[[r for r in ["Low", "Normal", "High"] if r in regime_freq.index]]

    reg_series = bt["Vol_Regime"].where(bt["Vol_Regime"].isin(["Low", "Normal", "High"]))
    transition_matrix = regime_analysis.compute_transition_matrix(reg_series)
    transition_stats = regime_analysis.compute_transition_stats(bt["Strategy_Net_Return"], reg_series)

    sensitivity = regime_analysis.compute_regime_sensitivity(strat_regime)

    high = bt_valid.loc[bt_valid["Vol_Regime"] == "High", "Strategy_Net_Return"]
    normal = bt_valid.loc[bt_valid["Vol_Regime"] == "Normal", "Strategy_Net_Return"]
    bootstrap_hn = bootstrap_diff_ci(high, normal, ann_sharpe, n_boot=2000, seed=42)

    walk_forward = backtester.walk_forward_backtest(
        df,
        "Signal_Trend",
        train_months=24,
        test_months=6,
        cost_bps=DEFAULT_COST_BPS / 10000,
        rebalance_freq=DEFAULT_REBALANCE_FREQ,
    )

    sweep_df = sweep.run_sma_regime_sweep(
        df,
        sma_windows=DEFAULT_SMA_SWEEP,
        mom_window=12,
        vol_q_high=DEFAULT_VOL_QUANTILE_HIGH,
        use_oos=True,
        vol_window=DEFAULT_VOLATILITY_WINDOW,
    )

    return AssetResults(
        raw=raw,
        df=df,
        bt=bt,
        strat_metrics=strat_metrics,
        bench_metrics=bench_metrics,
        strat_regime=strat_regime,
        bench_regime=bench_regime,
        avg_vol=avg_vol,
        regime_freq=regime_freq,
        transition_matrix=transition_matrix,
        transition_stats=transition_stats,
        sensitivity=sensitivity,
        bootstrap_hn=bootstrap_hn,
        walk_forward=walk_forward,
        sweep_df=sweep_df,
    )


def make_figures(spy: AssetResults, qqq: AssetResults, iwm: AssetResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(spy.bt.index, spy.bt["Equity_Benchmark"], color=PALETTE["benchmark"], lw=2.0, label="Buy & Hold")
    ax.plot(spy.bt.index, spy.bt["Equity_Strategy"], color=PALETTE["strategy"], lw=2.6, label="Trend Strategy")
    ax.set_yscale("log")
    ax.set_title("Figure 1. Wealth Trajectory: Strategy vs Benchmark", loc="left", fontsize=16, pad=14)
    ax.set_ylabel("Cumulative growth (log scale)")
    ax.set_xlabel("Date")
    ax.grid(True, axis="y")
    for start, end, label in [
        ("2000-09-01", "2003-03-31", "Dot-com"),
        ("2007-10-01", "2009-03-31", "GFC"),
        ("2020-02-20", "2020-05-01", "COVID"),
    ]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#f8d7d7", alpha=0.30)
        ax.text(pd.Timestamp(start), ax.get_ylim()[1] / 1.25, label, fontsize=9, color=PALETTE["muted"])
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_equity_curves.png")
    plt.close(fig)

    # Figure 2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(spy.bt.index, spy.bt["DD_Benchmark"], 0, color=PALETTE["benchmark"], alpha=0.22)
    ax.plot(spy.bt.index, spy.bt["DD_Benchmark"], color=PALETTE["benchmark"], lw=1.8, label="Buy & Hold")
    ax.fill_between(spy.bt.index, spy.bt["DD_Strategy"], 0, color=PALETTE["strategy"], alpha=0.20)
    ax.plot(spy.bt.index, spy.bt["DD_Strategy"], color=PALETTE["strategy"], lw=2.2, label="Trend Strategy")
    ax.set_title("Figure 2. Drawdown Geometry and Tail Truncation", loc="left", fontsize=16, pad=14)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_drawdowns.png")
    plt.close(fig)

    # Figure 3
    freq = spy.regime_freq.reindex(["Low", "Normal", "High"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(freq.index, freq.values, color=[PALETTE["low"], PALETTE["normal"], PALETTE["high"]], width=0.58)
    ax.set_title("Figure 3. Volatility Regime Occupancy (OOS)", loc="left", fontsize=16, pad=14)
    ax.set_ylabel("Share of observations")
    ax.grid(True, axis="y")
    ax.set_ylim(0, max(freq.values) * 1.3)
    for b, v in zip(bars, freq.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v*100:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_regime_frequency.png")
    plt.close(fig)

    # Figure 4
    regs = ["Low", "Normal", "High"]
    y = np.arange(len(regs))[::-1]
    strat = [spy.strat_regime.loc[r, "Sharpe"] for r in regs]
    bench = [spy.bench_regime.loc[r, "Sharpe"] for r in regs]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(regs)):
        ax.plot([bench[i], strat[i]], [y[i], y[i]], color=PALETTE["grid"], lw=5, solid_capstyle="round")
    ax.scatter(bench, y, s=110, color=PALETTE["benchmark"], label="Buy & Hold")
    ax.scatter(strat, y, s=140, color=PALETTE["strategy"], label="Trend Strategy")
    for i in range(len(regs)):
        ax.text(strat[i] + 0.03, y[i] + 0.04, num(strat[i], 2), color=PALETTE["strategy"], fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(regs)
    ax.set_xlabel("Sharpe ratio")
    ax.set_title("Figure 4. Regime-Conditional Risk-Adjusted Returns", loc="left", fontsize=16, pad=14)
    ax.grid(True, axis="x")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_conditional_sharpe.png")
    plt.close(fig)

    # Figure 5
    tm = spy.transition_matrix.reindex(index=["Low", "Normal", "High"], columns=["Low", "Normal", "High"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 6.1))
    im = ax.imshow(tm.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            val = tm.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=("white" if val > 0.65 else PALETTE["text"]), fontsize=11)
    ax.set_xticks(range(3), tm.columns)
    ax.set_yticks(range(3), tm.index)
    ax.set_xlabel("Regime at t")
    ax.set_ylabel("Regime at t-1")
    ax.set_title("Figure 5. Regime Transition Matrix", loc="left", fontsize=16, pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition probability")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_transition_matrix.png")
    plt.close(fig)

    # Figure 6
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for reg, color in [("Low", PALETTE["low"]), ("Normal", PALETTE["normal"]), ("High", PALETTE["high"] )]:
        if reg in spy.sweep_df.index.get_level_values("Regime"):
            s = spy.sweep_df.xs(reg, level="Regime")["Sharpe"]
            ax.plot(s.index, s.values, marker="o", lw=2.4, color=color, label=f"{reg} volatility")
    ax.set_title("Figure 6. Parameter Robustness: SMA Window vs Regime Sharpe", loc="left", fontsize=16, pad=14)
    ax.set_xlabel("SMA window (days)")
    ax.set_ylabel("Sharpe ratio")
    ax.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sma_sweep.png")
    plt.close(fig)

    # Figure 7
    assets = ["SPY", "QQQ", "IWM"]
    data = {"SPY": spy, "QQQ": qqq, "IWM": iwm}
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for idx, metric in enumerate(["CAGR", "Sharpe"]):
        ax = axs[idx]
        for i, asset in enumerate(assets):
            b = data[asset].bench_metrics.get(metric, np.nan)
            s = data[asset].strat_metrics.get(metric, np.nan)
            color = [PALETTE["strategy"], "#5f8df7", "#8fb1ff"][i]
            ax.plot([0, 1], [b, s], marker="o", lw=2.8, color=color)
            ax.text(-0.03, b, f"{asset} {num(b,2) if metric=='Sharpe' else pct(b)}", ha="right", va="center", color=PALETTE["muted"], fontsize=9)
            ax.text(1.03, s, f"{num(s,2) if metric=='Sharpe' else pct(s)}", ha="left", va="center", color=color, fontsize=9)
        ax.set_xticks([0, 1], ["Buy & Hold", "Trend"])
        ax.set_xlim(-0.45, 1.45)
        ax.grid(True, axis="y")
        ax.set_title(metric)
    axs[0].set_ylabel("Annualized metric")
    fig.suptitle("Figure 7. Cross-Asset Robustness", x=0.04, ha="left", fontsize=16, weight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig_robustness_assets.png")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    if df is None or df.empty:
        return "_No data available._"
    d = df.copy()
    if d.index.name is None:
        d.index.name = "index"
    d = d.reset_index()

    def f(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "N/A"
            return f"{v:.{float_digits}f}"
        return str(v)

    headers = list(d.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(f(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def fmt_num(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "N/A"
    return f"{float(x):.{digits}f}"


def fmt_pct(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "N/A"
    return f"{float(x) * 100:.{digits}f}%"


def build_display_tables(inf_tables: dict[str, pd.DataFrame], rb_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # A1: inference by regime
    a1 = inf_tables["inference_strategy_vs_benchmark_by_regime"].copy()
    a1 = a1.rename(
        columns={
            "Sharpe_Diff": "Sharpe Diff (S-B)",
            "Sharpe_CI_Low": "Sharpe CI Low",
            "Sharpe_CI_High": "Sharpe CI High",
            "Sharpe_p": "Sharpe p-value",
            "CAGR_Diff": "CAGR Diff (pp)",
            "CAGR_CI_Low": "CAGR CI Low (pp)",
            "CAGR_CI_High": "CAGR CI High (pp)",
            "CAGR_p": "CAGR p-value",
        }
    )
    for c in ["Sharpe Diff (S-B)", "Sharpe CI Low", "Sharpe CI High"]:
        a1[c] = a1[c].map(lambda v: fmt_num(v, 3))
    for c in ["Sharpe p-value", "CAGR p-value"]:
        a1[c] = a1[c].map(lambda v: fmt_num(v, 3))
    for c in ["CAGR Diff (pp)", "CAGR CI Low (pp)", "CAGR CI High (pp)"]:
        a1[c] = a1[c].map(lambda v: fmt_pct(v, 2))
    a1.index.name = "Regime"

    # A2: high minus normal
    a2 = inf_tables["inference_high_minus_normal"].copy()
    a2 = a2.rename(columns={"diff": "Estimate", "ci_low": "CI Low", "ci_high": "CI High", "p_value": "p-value"})
    a2 = a2.rename(
        index={
            "Sharpe_High_minus_Normal": "Sharpe (High - Normal)",
            "CAGR_High_minus_Normal": "CAGR (High - Normal)",
        }
    )
    est_vals, low_vals, high_vals, p_vals = [], [], [], []
    for idx, row in a2.iterrows():
        if "CAGR" in str(idx):
            est_vals.append(fmt_pct(row["Estimate"], 2))
            low_vals.append(fmt_pct(row["CI Low"], 2))
            high_vals.append(fmt_pct(row["CI High"], 2))
        else:
            est_vals.append(fmt_num(row["Estimate"], 3))
            low_vals.append(fmt_num(row["CI Low"], 3))
            high_vals.append(fmt_num(row["CI High"], 3))
        p_vals.append(fmt_num(row["p-value"], 3))
    a2["Estimate"] = est_vals
    a2["CI Low"] = low_vals
    a2["CI High"] = high_vals
    a2["p-value"] = p_vals
    a2.index.name = "Metric"

    # B1: cost sensitivity
    b1 = rb_tables["appendix_cost_sensitivity"].copy()
    b1 = b1.rename(
        columns={
            "Strategy_CAGR": "Strategy CAGR",
            "Strategy_Sharpe": "Strategy Sharpe",
            "Strategy_MaxDD": "Strategy MaxDD",
            "Delta_CAGR_vs_BH": "Delta CAGR vs Buy-Hold",
            "Delta_Sharpe_vs_BH": "Delta Sharpe vs Buy-Hold",
        }
    )
    b1.index = [int(v) for v in b1.index]
    b1.index.name = "Cost (bps)"
    for c in ["Strategy CAGR", "Strategy MaxDD", "Delta CAGR vs Buy-Hold"]:
        b1[c] = b1[c].map(lambda v: fmt_pct(v, 2))
    for c in ["Strategy Sharpe", "Delta Sharpe vs Buy-Hold"]:
        b1[c] = b1[c].map(lambda v: fmt_num(v, 3))

    # B2: rebalance sensitivity
    b2 = rb_tables["appendix_rebalance_sensitivity"].copy()
    b2 = b2.rename(
        columns={
            "Strategy_CAGR": "Strategy CAGR",
            "Strategy_Sharpe": "Strategy Sharpe",
            "Strategy_MaxDD": "Strategy MaxDD",
            "Delta_CAGR_vs_BH": "Delta CAGR vs Buy-Hold",
            "Delta_Sharpe_vs_BH": "Delta Sharpe vs Buy-Hold",
        }
    )
    b2 = b2.rename(index={"D": "Daily", "W": "Weekly", "M": "Monthly"})
    b2.index.name = "Rebalance Frequency"
    for c in ["Strategy CAGR", "Strategy MaxDD", "Delta CAGR vs Buy-Hold"]:
        b2[c] = b2[c].map(lambda v: fmt_pct(v, 2))
    for c in ["Strategy Sharpe", "Delta Sharpe vs Buy-Hold"]:
        b2[c] = b2[c].map(lambda v: fmt_num(v, 3))

    # B3: baseline comparison
    b3 = rb_tables["appendix_signal_baseline"].copy()
    b3 = b3.rename(columns={"MaxDD": "Max Drawdown"})
    b3.index.name = "Model"
    b3["CAGR"] = b3["CAGR"].map(lambda v: fmt_pct(v, 2))
    b3["Sharpe"] = b3["Sharpe"].map(lambda v: fmt_num(v, 3))
    b3["Max Drawdown"] = b3["Max Drawdown"].map(lambda v: fmt_pct(v, 2))

    return {"a1": a1, "a2": a2, "b1": b1, "b2": b2, "b3": b3}


def build_robustness_tables(spy: AssetResults) -> dict[str, pd.DataFrame]:
    df = spy.df.copy()

    # A1: transaction cost sensitivity (monthly)
    cost_rows = []
    for bps in [0, 5, 10, 20, 50]:
        bt = backtester.run_backtest(df, "Signal_Trend", cost_bps=bps / 10000, rebalance_freq="M")
        sm = backtester.calculate_perf_metrics(bt["Equity_Strategy"])
        bm = backtester.calculate_perf_metrics(bt["Equity_Benchmark"])
        cost_rows.append(
            {
                "Cost_bps": bps,
                "Strategy_CAGR": sm.get("CAGR"),
                "Strategy_Sharpe": sm.get("Sharpe"),
                "Strategy_MaxDD": sm.get("MaxDD"),
                "Delta_CAGR_vs_BH": sm.get("CAGR") - bm.get("CAGR"),
                "Delta_Sharpe_vs_BH": sm.get("Sharpe") - bm.get("Sharpe"),
            }
        )
    cost_df = pd.DataFrame(cost_rows).set_index("Cost_bps")

    # A2: rebalance frequency sensitivity
    reb_rows = []
    for freq in ["D", "W", "M"]:
        bt = backtester.run_backtest(df, "Signal_Trend", cost_bps=DEFAULT_COST_BPS / 10000, rebalance_freq=freq)
        sm = backtester.calculate_perf_metrics(bt["Equity_Strategy"])
        bm = backtester.calculate_perf_metrics(bt["Equity_Benchmark"])
        reb_rows.append(
            {
                "Rebalance": freq,
                "Strategy_CAGR": sm.get("CAGR"),
                "Strategy_Sharpe": sm.get("Sharpe"),
                "Strategy_MaxDD": sm.get("MaxDD"),
                "Delta_CAGR_vs_BH": sm.get("CAGR") - bm.get("CAGR"),
                "Delta_Sharpe_vs_BH": sm.get("Sharpe") - bm.get("Sharpe"),
            }
        )
    reb_df = pd.DataFrame(reb_rows).set_index("Rebalance")

    # A3: baseline signal comparison on common sample (require SMA200 present)
    common = df.dropna(subset=["SMA_200", "Daily_Return", f"SMA_{DEFAULT_SMA_WINDOW}"]).copy()
    baseline_rows = []

    # Buy & Hold baseline on common sample
    bh_eq = (1 + common["Daily_Return"]).cumprod()
    bh = backtester.calculate_perf_metrics(bh_eq)
    baseline_rows.append(
        {
            "Model": "BuyHold",
            "CAGR": bh.get("CAGR"),
            "Sharpe": bh.get("Sharpe"),
            "MaxDD": bh.get("MaxDD"),
        }
    )

    for col, name in [("Signal_Trend", "SMA50"), ("Signal_Trend_200", "SMA200")]:
        bt = backtester.run_backtest(common, col, cost_bps=DEFAULT_COST_BPS / 10000, rebalance_freq="M")
        sm = backtester.calculate_perf_metrics(bt["Equity_Strategy"])
        baseline_rows.append(
            {
                "Model": name,
                "CAGR": sm.get("CAGR"),
                "Sharpe": sm.get("Sharpe"),
                "MaxDD": sm.get("MaxDD"),
            }
        )

    baseline_df = pd.DataFrame(baseline_rows).set_index("Model")

    return {
        "appendix_cost_sensitivity": cost_df,
        "appendix_rebalance_sensitivity": reb_df,
        "appendix_signal_baseline": baseline_df,
    }


def build_inference_tables(spy: AssetResults) -> dict[str, pd.DataFrame]:
    bt = spy.bt.copy()
    bt = bt[bt["Vol_Regime"].isin(["Low", "Normal", "High"])].copy()

    rows = []
    for reg in ["Low", "Normal", "High"]:
        sub = bt[bt["Vol_Regime"] == reg]
        out_sharpe = bootstrap_diff_ci(sub["Strategy_Net_Return"], sub["Daily_Return"], ann_sharpe, n_boot=2000, seed=17)
        out_cagr = bootstrap_diff_ci(sub["Strategy_Net_Return"], sub["Daily_Return"], cagr_from_returns, n_boot=2000, seed=19)
        rows.append(
            {
                "Regime": reg,
                "Sharpe_Diff": out_sharpe["diff"],
                "Sharpe_CI_Low": out_sharpe["ci_low"],
                "Sharpe_CI_High": out_sharpe["ci_high"],
                "Sharpe_p": out_sharpe["p_value"],
                "CAGR_Diff": out_cagr["diff"],
                "CAGR_CI_Low": out_cagr["ci_low"],
                "CAGR_CI_High": out_cagr["ci_high"],
                "CAGR_p": out_cagr["p_value"],
            }
        )

    regime_diff_df = pd.DataFrame(rows).set_index("Regime")

    high = bt.loc[bt["Vol_Regime"] == "High", "Strategy_Net_Return"]
    normal = bt.loc[bt["Vol_Regime"] == "Normal", "Strategy_Net_Return"]
    hn_sharpe = bootstrap_diff_ci(high, normal, ann_sharpe, n_boot=2000, seed=42)
    hn_cagr = bootstrap_diff_ci(high, normal, cagr_from_returns, n_boot=2000, seed=43)

    hn_df = pd.DataFrame(
        [
            {"Metric": "Sharpe_High_minus_Normal", **hn_sharpe},
            {"Metric": "CAGR_High_minus_Normal", **hn_cagr},
        ]
    ).set_index("Metric")

    return {
        "inference_strategy_vs_benchmark_by_regime": regime_diff_df,
        "inference_high_minus_normal": hn_df,
    }


def write_markdown(
    spy: AssetResults,
    qqq: AssetResults,
    iwm: AssetResults,
    inf_tables: dict[str, pd.DataFrame],
    rb_tables: dict[str, pd.DataFrame],
    md_path: Path,
    snapshot_info: dict,
) -> None:
    sm = spy.strat_metrics
    bm = spy.bench_metrics
    wf = spy.walk_forward.get("summary", {}) if spy.walk_forward else {}
    oos_periods = int(spy.walk_forward.get("n_periods", 0)) if spy.walk_forward else 0
    disp_tables = build_display_tables(inf_tables, rb_tables)

    def safe_get(df: pd.DataFrame, idx, col: str):
        try:
            return df.loc[idx, col]
        except Exception:
            return np.nan

    sample_full_start = snapshot_info["spy_raw_start"]
    sample_end = snapshot_info["spy_raw_end"]
    analysis_start = snapshot_info["spy_analysis_start"]

    low_sh = safe_get(spy.strat_regime, "Low", "Sharpe")
    normal_sh = safe_get(spy.strat_regime, "Normal", "Sharpe")
    high_sh = safe_get(spy.strat_regime, "High", "Sharpe")
    low_bh_sh = safe_get(spy.bench_regime, "Low", "Sharpe")
    normal_bh_sh = safe_get(spy.bench_regime, "Normal", "Sharpe")
    high_bh_sh = safe_get(spy.bench_regime, "High", "Sharpe")

    low_freq = spy.regime_freq.get("Low", np.nan)
    normal_freq = spy.regime_freq.get("Normal", np.nan)
    high_freq = spy.regime_freq.get("High", np.nan)

    p_high_high = safe_get(spy.transition_matrix, "High", "High")
    p_low_normal = safe_get(spy.transition_matrix, "Low", "Normal")
    sh_low_normal = safe_get(spy.transition_stats, "Low→Normal", "Sharpe")

    cost_tbl = rb_tables["appendix_cost_sensitivity"]
    reb_tbl = rb_tables["appendix_rebalance_sensitivity"]
    base_tbl = rb_tables["appendix_signal_baseline"]
    inf_tbl = inf_tables["inference_strategy_vs_benchmark_by_regime"]

    cost0_cagr = safe_get(cost_tbl, 0, "Strategy_CAGR")
    cost50_cagr = safe_get(cost_tbl, 50, "Strategy_CAGR")
    reb_d_sh = safe_get(reb_tbl, "D", "Strategy_Sharpe")
    reb_m_sh = safe_get(reb_tbl, "M", "Strategy_Sharpe")
    sma200_sh = safe_get(base_tbl, "SMA200", "Sharpe")
    sma50_sh = safe_get(base_tbl, "SMA50", "Sharpe")

    low_diff_p = safe_get(inf_tbl, "Low", "Sharpe_p")
    normal_diff_p = safe_get(inf_tbl, "Normal", "Sharpe_p")
    high_diff_p = safe_get(inf_tbl, "High", "Sharpe_p")

    md = f"""# Volatility Regimes and Trend-Following Performance in U.S. Equities: An Empirical Deconstruction

**Author:** Aarjav Ametha  
**Date:** February 2026  
**Repository:** [github.com/aarjava/hedgefund-dashboard](https://github.com/aarjava/hedgefund-dashboard)

## Abstract
This paper evaluates the efficacy of a standard trend-following rule (`Price > 50-day SMA`) on SPY over a 33-year sample (`{sample_full_start}` to `{sample_end}`).
Unconditionally, the strategy underperforms the benchmark on return and Sharpe (`{num(sm.get('Sharpe'))}` vs `{num(bm.get('Sharpe'))}`) but materially improves downside containment (`{pct(sm.get('MaxDD'))}` vs `{pct(bm.get('MaxDD'))}` max drawdown).
A regime-conditional decomposition reveals a structural asymmetry: performance quality is concentrated in Low Volatility (`Sharpe {num(low_sh)}`) and decays during volatility expansion (Normal `{num(normal_sh)}`, High `{num(high_sh)}`).
Walk-forward OOS validation remains directionally consistent (`Sharpe {num(wf.get('Sharpe'))}` across `{oos_periods}` test windows), supporting robustness of the central claim.

## 1. Introduction
Trend-following in equities is often marketed as *crisis alpha*: a strategy that should perform best when volatility spikes and directional dislocations emerge.
This paper tests that claim using a strict regime-conditioned framework rather than unconditional averages.

The main empirical finding is that the classic “smile” narrative does not hold for broad U.S. equities in this sample. Instead, the profile is closer to a **checkmark**:
- strong quality in Low Volatility;
- weak quality in Normal Volatility (whipsaw zone);
- only modest quality in High Volatility, despite clear crash-risk truncation.

This distinction matters for portfolio design. The strategy appears to function more as a **risk-allocation and drawdown-control mechanism** than a universal return enhancer.

## 1.1 Hypotheses and Contributions
Hypotheses tested:
- **H1 (Crisis Alpha):** Trend-following quality is highest in High Volatility states.
- **H2 (Low-Vol Dominance):** Trend-following quality is highest in Low Volatility states.
- **H3 (Transition Bleed):** The largest quality decay occurs during `Low -> Normal` state transitions.

Contributions of this paper:
- Formal OOS regime decomposition on SPY with explicit expanding-window state labels.
- Transition-level microstructure view that isolates where quality is lost.
- Robustness stack (walk-forward, cost/rebalance sensitivity, SMA sweep, cross-asset checks).
- Claim-to-evidence mapping instead of narrative-only interpretation.

## 2. Data and Methodology
### 2.1 Dataset and Trading Rule
- **Primary instrument:** SPY (cross-asset checks on QQQ and IWM).
- **Raw sample window:** `{sample_full_start}` to `{sample_end}`.
- **Effective analysis start:** `{analysis_start}` (after indicator warm-up and OOS regime eligibility).
- **Signal definition:**

  `Position_t = 1 if Price_t > SMA50_t, else 0`

- **Execution assumptions:** monthly rebalance, `10 bps` turnover cost.

### 2.2 Regime Labeling (No Look-Ahead)
Regimes are defined from annualized 21-day realized volatility using expanding-window quantiles:
- **Low:** below 25th percentile of history available at time *t*.
- **Normal:** between 25th and 75th percentiles.
- **High:** above 75th percentile.

Because thresholds are expanding-window estimates, regime labels are out-of-sample by construction.

### 2.3 Statistics and Validation
- Unconditional metrics: CAGR, Sharpe, Max Drawdown, Win Rate.
- Conditional metrics: same statistics computed within each volatility state.
- Inference layer: bootstrap confidence intervals and p-values for regime-level strategy-minus-benchmark differences.
- OOS validation: rolling walk-forward (`24m` train / `6m` test), repeated over `{oos_periods}` periods.

## 3. Unconditional Performance
![Figure 1: SPY Equity Curves (Log Scale)](output/figures/fig_equity_curves.png)

*Figure 1. SPY equity curves (log scale).*

![Figure 2: SPY Drawdown Curves](output/figures/fig_drawdowns.png)

*Figure 2. SPY drawdown curves.*

Figure 1 shows the long-horizon *decoupling* behavior: during major bear phases, the trend strategy flattens as exposure is cut, while buy-and-hold continues to absorb the full drawdown path.
Figure 2 quantifies this decoupling in underwater terms: depth is materially truncated and recovery cycles are shortened relative to the benchmark.

- Strategy CAGR: `{pct(sm.get('CAGR'))}`
- Benchmark CAGR: `{pct(bm.get('CAGR'))}`
- Strategy Sharpe: `{num(sm.get('Sharpe'))}`
- Benchmark Sharpe: `{num(bm.get('Sharpe'))}`
- Strategy MaxDD: `{pct(sm.get('MaxDD'))}`
- Benchmark MaxDD: `{pct(bm.get('MaxDD'))}`
- Strategy Win Rate: `{pct(sm.get('WinRate'))}`
- Benchmark Win Rate: `{pct(bm.get('WinRate'))}`

Interpretation:
- The strategy pays an explicit *lag premium* (lower CAGR) in sustained bull markets.
- In return, it buys meaningful left-tail truncation.
- Economically, this resembles an endogenous de-risking overlay rather than pure alpha extraction.

## 4. Regime Decomposition
![Figure 3: Regime Frequency (SPY, OOS)](output/figures/fig_regime_frequency.png)

*Figure 3. Regime occupancy.*

![Figure 4: Sharpe Ratio by Volatility Regime](output/figures/fig_conditional_sharpe.png)

*Figure 4. Regime-conditional Sharpe.*

Figure 3 confirms occupancy is non-trivial across all states (`Low {pct(low_freq)}`, `Normal {pct(normal_freq)}`, `High {pct(high_freq)}`), so the conditional decomposition is not driven by a tiny corner sample.
Figure 4 shows the central anomaly directly: the quality profile is checkmark-shaped, not smile-shaped.

- Low-vol strategy Sharpe: `{num(spy.strat_regime.loc['Low', 'Sharpe'])}`
- Normal-vol strategy Sharpe: `{num(spy.strat_regime.loc['Normal', 'Sharpe'])}`
- High-vol strategy Sharpe: `{num(spy.strat_regime.loc['High', 'Sharpe'])}`
- Low-vol benchmark Sharpe: `{num(low_bh_sh)}`
- Normal-vol benchmark Sharpe: `{num(normal_bh_sh)}`
- High-vol benchmark Sharpe: `{num(high_bh_sh)}`
- High-minus-Normal Sharpe diff: `{num(spy.bootstrap_hn['diff'])}` (95% CI: `{num(spy.bootstrap_hn['ci_low'])}`, `{num(spy.bootstrap_hn['ci_high'])}`; p={num(spy.bootstrap_hn['p_value'],3)})

Interpretation:
- The quality surface follows a **checkmark** shape, not a smile.
- Low-vol environments support persistent directional drift and lower signal noise.
- Normal and High-vol states degrade quality through mean-reversion, gap risk, and delayed re-entry after sharp rebounds.

## 5. Transition Microstructure
![Figure 5: Regime Transition Matrix](output/figures/fig_transition_matrix.png)

*Figure 5. Regime transition matrix.*

Figure 5 provides the transition diagnostics that explain why performance can decay quickly:
- `P(High_t | High_(t-1)) = {pct(p_high_high)}` (high persistence).
- `P(Normal_t | Low_(t-1)) = {pct(p_low_normal)}` (infrequent but important state break).
- `Low -> Normal` transition Sharpe: `{num(sh_low_normal)}`.

This `Low -> Normal` handoff is the strategy’s main bleed point: trends lose smoothness, volatility expands, and the SMA signal is forced to react late.

## 6. Robustness and Generalization
![Figure 6: SMA Parameter Sweep](output/figures/fig_sma_sweep.png)

*Figure 6. SMA parameter sweep.*

![Figure 7: Cross-Asset Robustness](output/figures/fig_robustness_assets.png)

*Figure 7. Cross-asset robustness.*

- QQQ strategy Sharpe: `{num(qqq.strat_metrics.get('Sharpe'))}` vs benchmark `{num(qqq.bench_metrics.get('Sharpe'))}`
- IWM strategy Sharpe: `{num(iwm.strat_metrics.get('Sharpe'))}` vs benchmark `{num(iwm.bench_metrics.get('Sharpe'))}`
- SMA50 Sharpe (common sample): `{num(sma50_sh)}`
- SMA200 Sharpe (common sample): `{num(sma200_sh)}`

Walk-forward OOS summary:
- OOS CAGR `{pct(wf.get('CAGR'))}`, OOS Sharpe `{num(wf.get('Sharpe'))}`, OOS MaxDD `{pct(wf.get('MaxDD'))}`, periods `{oos_periods}`.

Additional robustness diagnostics:
- Cost sensitivity: CAGR decays from `{pct(cost0_cagr)}` at `0 bps` to `{pct(cost50_cagr)}` at `50 bps`.
- Rebalance sensitivity: daily rebalance Sharpe `{num(reb_d_sh)}` vs monthly `{num(reb_m_sh)}`.

Interpretation:
- The low-volatility dominance is stable across lookback windows.
- Slower trend speeds (e.g., SMA200) can materially reduce whipsaw drag in noisy regimes.
- Cross-asset behavior is coherent with market microstructure: momentum-rich QQQ adapts better than choppier IWM.

## 7. Claim-to-Evidence Alignment
- **H1 (Crisis Alpha): Rejected.** In Figure 4, High-volatility strategy Sharpe (`{num(high_sh)}`) is well below Low-volatility Sharpe (`{num(low_sh)}`) and also below High-vol benchmark Sharpe (`{num(high_bh_sh)}`).
- **H2 (Low-Vol Dominance): Supported.** Figure 4 shows peak quality in Low volatility; this pattern remains visible across the sweep in Figure 6.
- **H3 (Transition Bleed): Supported.** Figure 5 transition analytics show severe `Low -> Normal` degradation (`Sharpe {num(sh_low_normal)}`), consistent with state-break whipsaw.

## 8. Inference Appendix
### Table A1. Strategy vs Benchmark Difference by Regime (Bootstrap)
{markdown_table(disp_tables['a1'], float_digits=4)}

### Table A2. High minus Normal Difference (Strategy Only)
{markdown_table(disp_tables['a2'], float_digits=4)}

## 9. Robustness Appendix
### Table B1. Transaction Cost Sensitivity
{markdown_table(disp_tables['b1'], float_digits=4)}

### Table B2. Rebalance Frequency Sensitivity
{markdown_table(disp_tables['b2'], float_digits=4)}

### Table B3. Baseline Comparison (BuyHold vs SMA50 vs SMA200)
{markdown_table(disp_tables['b3'], float_digits=4)}

## 10. Statistical Reading and Limits
Bootstrap inference confirms directionality but also shows uncertainty around effect sizes:
- Regime Sharpe difference p-values (strategy minus benchmark): Low `{num(low_diff_p,3)}`, Normal `{num(normal_diff_p,3)}`, High `{num(high_diff_p,3)}`.
- High-minus-Normal strategy Sharpe spread has wide confidence bounds.

This means conclusions should be framed as *structural patterns* rather than point-estimate certainty.

## 11. Conclusion
For U.S. equities in this sample, trend-following is not best viewed as crisis alpha.
It is a regime-dependent exposure controller: strongest in low-volatility drift, weakest during volatility transitions, and consistently useful for drawdown truncation.
The most defensible improvement path is volatility-adaptive signal speed and explicit transition-risk handling around `Low -> Normal` breaks.
"""

    md_path.write_text(md)


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def df_to_latex_table(df: pd.DataFrame, caption: str, label: str, digits: int = 4) -> str:
    if df is None or df.empty:
        return f"""\\begin{{table}}[H]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l}}
\\toprule
No data available. \\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    d = df.copy()
    if d.index.name is None:
        d.index.name = "index"
    d = d.reset_index()

    cols = list(d.columns)
    spec = "l" + "r" * (len(cols) - 1)
    head = " & ".join(tex_escape(c) for c in cols) + r" \\"

    rows = []
    for _, row in d.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    vals.append("N/A")
                else:
                    vals.append(f"{v:.{digits}f}")
            else:
                vals.append(tex_escape(str(v)))
        rows.append(" & ".join(vals) + r" \\")

    body = "\n".join(rows)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{{spec}}}
\\toprule
{head}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}"""


def write_arxiv_tex(
    spy: AssetResults,
    qqq: AssetResults,
    iwm: AssetResults,
    inf_tables: dict[str, pd.DataFrame],
    rb_tables: dict[str, pd.DataFrame],
    tex_path: Path,
    snapshot_info: dict,
) -> None:
    sm = spy.strat_metrics
    bm = spy.bench_metrics
    wf = spy.walk_forward.get("summary", {}) if spy.walk_forward else {}
    oos_periods = int(spy.walk_forward.get("n_periods", 0)) if spy.walk_forward else 0
    disp_tables = build_display_tables(inf_tables, rb_tables)

    def safe_get(df: pd.DataFrame, idx, col: str):
        try:
            return df.loc[idx, col]
        except Exception:
            return np.nan

    # Escape injected metric strings so LaTeX does not break on '%' values.
    sm_sharpe = tex_escape(num(sm.get("Sharpe")))
    bm_sharpe = tex_escape(num(bm.get("Sharpe")))
    sm_cagr = tex_escape(pct(sm.get("CAGR")))
    bm_cagr = tex_escape(pct(bm.get("CAGR")))
    sm_maxdd = tex_escape(pct(sm.get("MaxDD")))
    bm_maxdd = tex_escape(pct(bm.get("MaxDD")))
    hn_diff = tex_escape(num(spy.bootstrap_hn["diff"]))
    hn_ci_low = tex_escape(num(spy.bootstrap_hn["ci_low"]))
    hn_ci_high = tex_escape(num(spy.bootstrap_hn["ci_high"]))
    hn_p = tex_escape(num(spy.bootstrap_hn["p_value"], 3))
    oos_cagr = tex_escape(pct(wf.get("CAGR")))
    oos_sharpe = tex_escape(num(wf.get("Sharpe")))
    oos_maxdd = tex_escape(pct(wf.get("MaxDD")))
    sm_win = tex_escape(pct(sm.get("WinRate")))
    bm_win = tex_escape(pct(bm.get("WinRate")))

    low_sh = tex_escape(num(safe_get(spy.strat_regime, "Low", "Sharpe")))
    normal_sh = tex_escape(num(safe_get(spy.strat_regime, "Normal", "Sharpe")))
    high_sh = tex_escape(num(safe_get(spy.strat_regime, "High", "Sharpe")))
    low_bh_sh = tex_escape(num(safe_get(spy.bench_regime, "Low", "Sharpe")))
    normal_bh_sh = tex_escape(num(safe_get(spy.bench_regime, "Normal", "Sharpe")))
    high_bh_sh = tex_escape(num(safe_get(spy.bench_regime, "High", "Sharpe")))

    low_freq = tex_escape(pct(spy.regime_freq.get("Low", np.nan)))
    normal_freq = tex_escape(pct(spy.regime_freq.get("Normal", np.nan)))
    high_freq = tex_escape(pct(spy.regime_freq.get("High", np.nan)))

    p_high_high = tex_escape(pct(safe_get(spy.transition_matrix, "High", "High")))
    p_low_normal = tex_escape(pct(safe_get(spy.transition_matrix, "Low", "Normal")))
    sh_low_normal = tex_escape(num(safe_get(spy.transition_stats, "Low→Normal", "Sharpe")))

    cost_tbl = rb_tables["appendix_cost_sensitivity"]
    reb_tbl = rb_tables["appendix_rebalance_sensitivity"]
    base_tbl = rb_tables["appendix_signal_baseline"]
    inf_tbl = inf_tables["inference_strategy_vs_benchmark_by_regime"]

    cost0_cagr = tex_escape(pct(safe_get(cost_tbl, 0, "Strategy_CAGR")))
    cost50_cagr = tex_escape(pct(safe_get(cost_tbl, 50, "Strategy_CAGR")))
    reb_d_sh = tex_escape(num(safe_get(reb_tbl, "D", "Strategy_Sharpe")))
    reb_m_sh = tex_escape(num(safe_get(reb_tbl, "M", "Strategy_Sharpe")))
    sma200_sh = tex_escape(num(safe_get(base_tbl, "SMA200", "Sharpe")))
    sma50_sh = tex_escape(num(safe_get(base_tbl, "SMA50", "Sharpe")))

    low_diff_p = tex_escape(num(safe_get(inf_tbl, "Low", "Sharpe_p"), 3))
    normal_diff_p = tex_escape(num(safe_get(inf_tbl, "Normal", "Sharpe_p"), 3))
    high_diff_p = tex_escape(num(safe_get(inf_tbl, "High", "Sharpe_p"), 3))
    sample_start = tex_escape(snapshot_info["spy_raw_start"])
    sample_end = tex_escape(snapshot_info["spy_raw_end"])
    analysis_start = tex_escape(snapshot_info["spy_analysis_start"])

    a1 = df_to_latex_table(
        disp_tables["a1"],
        "Strategy minus Benchmark by regime (bootstrap).",
        "tab:inf_regime",
        digits=4,
    )
    a2 = df_to_latex_table(
        disp_tables["a2"],
        "High minus Normal differences (strategy only).",
        "tab:inf_hn",
        digits=4,
    )
    b1 = df_to_latex_table(disp_tables["b1"], "Transaction cost sensitivity.", "tab:cost", digits=4)
    b2 = df_to_latex_table(disp_tables["b2"], "Rebalance frequency sensitivity.", "tab:rebal", digits=4)
    b3 = df_to_latex_table(disp_tables["b3"], "Baseline signal comparison.", "tab:baseline", digits=4)

    content = f"""\\pdfoutput=1
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{float}}
\\usepackage[numbers]{{natbib}}
\\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{{hyperref}}

\\title{{Volatility Regimes and Trend-Following Performance in U.S. Equities: An Empirical Deconstruction}}
\\author{{Aarjav Ametha}}
\\date{{February 2026}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
This paper evaluates a standard trend-following rule ($P_t > SMA_{{50,t}}$) on SPY over a 33-year sample ({sample_start} to {sample_end}). Unconditionally, the strategy underperforms buy-and-hold on Sharpe (\\textbf{{{sm_sharpe}}} vs. \\textbf{{{bm_sharpe}}}) and CAGR (\\textbf{{{sm_cagr}}} vs. \\textbf{{{bm_cagr}}}), but materially improves downside containment (MaxDD \\textbf{{{sm_maxdd}}} vs. \\textbf{{{bm_maxdd}}}). A regime-conditional decomposition reveals structural asymmetry: quality is concentrated in Low Volatility (Sharpe {low_sh}) and decays through volatility expansion (Normal {normal_sh}, High {high_sh}). Walk-forward validation remains directionally consistent (OOS Sharpe {oos_sharpe} across {oos_periods} test windows), supporting robustness of the central result.
\\end{{abstract}}

\\section{{Introduction}}
Trend-following in equities is frequently framed as ``crisis alpha,'' i.e., superior return quality during volatility shocks. We test that claim by conditioning performance on out-of-sample volatility regimes instead of relying on unconditional averages. The framing is consistent with prior evidence on moving-average rules and trend-following across assets \\citep{{brock1992simple,moskowitz2012time,hurst2017century}} as well as broader momentum evidence in equities \\citep{{jegadeesh1993returns}}.

The empirical profile in this sample is not a smile; it is closer to a checkmark. Performance quality is strongest in Low Volatility and degrades in Normal and High Volatility states. The strategy still provides material drawdown truncation, but that benefit is risk-management oriented, not broad crisis-state alpha; practitioner treatments also emphasize this allocation lens \\citep{{antonacci2014dual}}.

\\subsection{{Hypotheses and contributions}}
Hypotheses tested:
\\begin{{itemize}}
\\item \\textbf{{H1 (Crisis alpha):}} trend-following quality is highest in High Volatility states.
\\item \\textbf{{H2 (Low-vol dominance):}} trend-following quality is highest in Low Volatility states.
\\item \\textbf{{H3 (Transition bleed):}} the largest quality decay occurs during $Low \\rightarrow Normal$ transitions.
\\end{{itemize}}

Contributions:
\\begin{{itemize}}
\\item OOS regime decomposition with expanding-window state labels.
\\item Transition-level microstructure diagnostics for where quality is lost.
\\item Robustness stack across walk-forward, cost/rebalance, SMA sweep, and cross-asset checks.
\\item Explicit hypothesis-to-evidence mapping.
\\end{{itemize}}

\\section{{Data and Methodology}}
\\subsection{{Sample and signal}}
Instrument: SPY (robustness assets: QQQ and IWM). Raw sample window: {sample_start} to {sample_end}. Effective analysis starts {analysis_start} after indicator warm-up and out-of-sample regime eligibility.

Trading signal:
\\begin{{equation}}
\\text{{Position}}_t =
\\begin{{cases}}
1 & \\text{{if }} P_t > SMA_{{50,t}} \\\\
0 & \\text{{otherwise}}
\\end{{cases}}
\\end{{equation}}

Execution assumptions: monthly rebalance, 10 bps turnover cost.

\\subsection{{Regime definition and validation}}
Regimes are defined from annualized 21-day realized volatility using expanding-window quantiles:
Low ($<25$th percentile), Normal ($25$th--$75$th), and High ($>75$th). This prevents look-ahead in threshold construction.

Validation stack:
\\begin{{itemize}}
\\item Bootstrap confidence intervals and p-values for strategy-minus-benchmark differences by regime.
\\item High-minus-Normal spread test for strategy-only quality decay.
\\item Rolling walk-forward evaluation (24-month train / 6-month test).
\\end{{itemize}}

\\section{{Results}}
\\subsection{{Unconditional performance: insurance cost vs. tail truncation}}
\\begin{{itemize}}
\\item Strategy CAGR: {sm_cagr}; Benchmark CAGR: {bm_cagr}.
\\item Strategy Sharpe: {sm_sharpe}; Benchmark Sharpe: {bm_sharpe}.
\\item Strategy MaxDD: {sm_maxdd}; Benchmark MaxDD: {bm_maxdd}.
\\item Strategy Win Rate: {sm_win}; Benchmark Win Rate: {bm_win}.
\\end{{itemize}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\linewidth]{{figures/fig_equity_curves.png}}
\\caption{{Figure 1. SPY equity curves (log scale), trend strategy vs. buy-and-hold.}}
\\label{{fig:equity}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\linewidth]{{figures/fig_drawdowns.png}}
\\caption{{Figure 2. SPY drawdown curves showing tail-risk truncation.}}
\\label{{fig:drawdown}}
\\end{{figure}}

Figures~\\ref{{fig:equity}} and \\ref{{fig:drawdown}} show the key trade-off: lower trend participation in long bull runs, but materially reduced tail-depth and faster drawdown recovery dynamics.

\\subsection{{Regime anomaly: where quality actually lives}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\linewidth]{{figures/fig_regime_frequency.png}}
\\caption{{Figure 3. OOS volatility-regime occupancy for SPY.}}
\\label{{fig:regimefreq}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.82\\linewidth]{{figures/fig_conditional_sharpe.png}}
\\caption{{Figure 4. Regime-conditional Sharpe ratios (strategy vs. benchmark).}}
\\label{{fig:condsharpe}}
\\end{{figure}}

Figure~\\ref{{fig:regimefreq}} confirms occupancy is meaningful in all states (Low {low_freq}, Normal {normal_freq}, High {high_freq}), so the conditional profile is not a sparse-sample artifact.
Figure~\\ref{{fig:condsharpe}} directly shows the checkmark profile.
\\begin{{itemize}}
\\item Strategy Sharpe by regime: Low {low_sh}, Normal {normal_sh}, High {high_sh}.
\\item Benchmark Sharpe by regime: Low {low_bh_sh}, Normal {normal_bh_sh}, High {high_bh_sh}.
\\item High-minus-Normal strategy Sharpe spread: {hn_diff} (95\\% CI [{hn_ci_low}, {hn_ci_high}], p={hn_p}).
\\end{{itemize}}

\\subsection{{Transition microstructure and robustness}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.70\\linewidth]{{figures/fig_transition_matrix.png}}
\\caption{{Figure 5. Volatility regime transition matrix.}}
\\label{{fig:transition}}
\\end{{figure}}

Figure~\\ref{{fig:transition}} provides the transition diagnostics:
\\begin{{itemize}}
\\item $P(\\text{{High}}_t \\mid \\text{{High}}_{{t-1}})$ = {p_high_high}.
\\item $P(\\text{{Normal}}_t \\mid \\text{{Low}}_{{t-1}})$ = {p_low_normal}.
\\item Sharpe during $Low \\rightarrow Normal$ transitions = {sh_low_normal}.
\\end{{itemize}}

The $Low \\rightarrow Normal$ handoff is the primary bleed regime: trend smoothness breaks, realized volatility expands, and lagged signals adapt late.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.84\\linewidth]{{figures/fig_sma_sweep.png}}
\\caption{{Figure 6. SMA lookback sweep by regime-level Sharpe ratio.}}
\\label{{fig:smasweep}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\linewidth]{{figures/fig_robustness_assets.png}}
\\caption{{Figure 7. Cross-asset robustness across SPY, QQQ, and IWM.}}
\\label{{fig:crossasset}}
\\end{{figure}}

Additional robustness diagnostics:
\\begin{{itemize}}
\\item OOS walk-forward summary: CAGR {oos_cagr}, Sharpe {oos_sharpe}, MaxDD {oos_maxdd}, periods {oos_periods}.
\\item Cost sensitivity: strategy CAGR declines from {cost0_cagr} (0 bps) to {cost50_cagr} (50 bps).
\\item Rebalance sensitivity: strategy Sharpe is {reb_d_sh} (daily) vs. {reb_m_sh} (monthly).
\\item Baseline comparison: SMA50 Sharpe {sma50_sh} vs. SMA200 Sharpe {sma200_sh} on common sample.
\\end{{itemize}}

Figures~\\ref{{fig:smasweep}} and \\ref{{fig:crossasset}} align with the core interpretation: low-volatility trend quality is persistent across parameterizations and strongest in momentum-rich indices.

\\section{{Hypothesis-to-Evidence Alignment}}
\\begin{{itemize}}
\\item \\textbf{{H1 (Crisis alpha): Rejected.}} Figure~\\ref{{fig:condsharpe}} shows High-vol strategy Sharpe ({high_sh}) below Low-vol strategy Sharpe ({low_sh}) and below High-vol benchmark Sharpe ({high_bh_sh}).
\\item \\textbf{{H2 (Low-vol dominance): Supported.}} Figure~\\ref{{fig:condsharpe}} shows highest strategy quality in Low volatility, and Figure~\\ref{{fig:smasweep}} shows this ordering is robust across lookbacks.
\\item \\textbf{{H3 (Transition bleed): Supported.}} Figure~\\ref{{fig:transition}} isolates severe quality decay in the $Low \\rightarrow Normal$ handoff (Sharpe {sh_low_normal}).
\\end{{itemize}}

\\section{{Appendix: Inference and Robustness Tables}}
{a1}

{a2}

{b1}

{b2}

{b3}

\\section{{Interpretation and Limits}}
Bootstrap inference supports directionality but indicates non-trivial uncertainty in several effect-size gaps. Strategy-minus-benchmark Sharpe p-values by regime are: Low {low_diff_p}, Normal {normal_diff_p}, High {high_diff_p}. Therefore conclusions should be read as structural (state-dependent quality and robust drawdown truncation), not as point-estimate precision claims.

\\section{{Conclusion}}
For U.S. equities in this sample, trend-following is best interpreted as a regime-dependent exposure controller rather than a universal crisis-alpha engine. The strongest improvement path is transition-aware and volatility-adaptive signal speed, especially around the $Low \\rightarrow Normal$ state break where performance decay is most severe.

\\bibliographystyle{{plainnat}}
\\bibliography{{references}}
\\end{{document}}
"""

    tex_path.write_text(content)


def save_tables(
    spy: AssetResults,
    qqq: AssetResults,
    iwm: AssetResults,
    inf_tables: dict[str, pd.DataFrame],
    rb_tables: dict[str, pd.DataFrame],
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    uncond = pd.DataFrame(
        {
            "Strategy": spy.strat_metrics,
            "Benchmark": spy.bench_metrics,
        }
    )

    cond = pd.DataFrame(
        {
            "Avg_Vol_Ann": spy.avg_vol,
            "Strategy_Sharpe": spy.strat_regime["Sharpe"],
            "Benchmark_Sharpe": spy.bench_regime["Sharpe"],
            "Strategy_CAGR": spy.strat_regime["CAGR"],
            "Benchmark_CAGR": spy.bench_regime["CAGR"],
            "Strategy_WinRate": spy.strat_regime["WinRate"],
            "Count": spy.strat_regime["Count"],
        }
    )

    robustness = pd.DataFrame(
        [
            {
                "Asset": "QQQ",
                "Strategy_CAGR": qqq.strat_metrics.get("CAGR"),
                "Strategy_Sharpe": qqq.strat_metrics.get("Sharpe"),
                "Benchmark_CAGR": qqq.bench_metrics.get("CAGR"),
                "Benchmark_Sharpe": qqq.bench_metrics.get("Sharpe"),
            },
            {
                "Asset": "IWM",
                "Strategy_CAGR": iwm.strat_metrics.get("CAGR"),
                "Strategy_Sharpe": iwm.strat_metrics.get("Sharpe"),
                "Benchmark_CAGR": iwm.bench_metrics.get("CAGR"),
                "Benchmark_Sharpe": iwm.bench_metrics.get("Sharpe"),
            },
        ]
    ).set_index("Asset")

    outputs = {
        "main_unconditional.csv": uncond,
        "main_conditional_by_regime.csv": cond,
        "main_regime_frequency.csv": spy.regime_freq.to_frame("Frequency"),
        "main_transition_matrix.csv": spy.transition_matrix,
        "main_transition_stats.csv": spy.transition_stats,
        "main_robustness_assets.csv": robustness,
        "appendix_inference_strategy_vs_benchmark_by_regime.csv": inf_tables["inference_strategy_vs_benchmark_by_regime"],
        "appendix_inference_high_minus_normal.csv": inf_tables["inference_high_minus_normal"],
        "appendix_cost_sensitivity.csv": rb_tables["appendix_cost_sensitivity"],
        "appendix_rebalance_sensitivity.csv": rb_tables["appendix_rebalance_sensitivity"],
        "appendix_signal_baseline.csv": rb_tables["appendix_signal_baseline"],
    }

    for name, df in outputs.items():
        df.to_csv(tables_dir / name, float_format="%.8f")


def sync_figures(canonical_figures: Path, targets: list[Path]) -> None:
    for t in targets:
        t.mkdir(parents=True, exist_ok=True)
        for f in canonical_figures.glob("fig_*.png"):
            shutil.copy2(f, t / f.name)


def write_arxiv_submission_notes(arxiv_dir: Path) -> None:
    readme = """# arXiv Submission Package

This directory contains a source-ready arXiv package.

## Included in the clean tarball
- `main.tex` (primary manuscript)
- `main.bbl` (resolved bibliography)
- `references.bib` (source bibliography)
- `figures/fig_*.png` (all referenced figures)
- `README_ARXIV.md`
- `ARXIV_METADATA.md`

## arXiv compiler settings
- Compiler: `pdfLaTeX`
- Main file: `main.tex`

## Local compile command
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
"""

    metadata = """# Suggested arXiv Metadata

## Title
Volatility Regimes and Trend-Following Performance in U.S. Equities: An Empirical Deconstruction

## Author
Aarjav Ametha

## Abstract
This paper evaluates a standard trend-following rule (`Price > 50-day SMA`) on SPY over a 33-year sample. Unconditionally, the strategy underperforms buy-and-hold on return quality but materially reduces drawdown depth. A regime-conditional decomposition shows a clear state asymmetry: performance quality is concentrated in Low Volatility environments and degrades during volatility expansion, with the sharpest decay around the Low->Normal transition. Walk-forward, parameter sweep, transaction-cost, rebalance-frequency, and cross-asset tests support the robustness of the structural result. For broad U.S. equities, trend-following behaves more like a regime-dependent risk-allocation overlay than a universal crisis-alpha engine.

## Suggested Categories
- Primary: `q-fin.PM`
- Secondary: `q-fin.ST`

## Keywords
- trend following
- volatility regimes
- tactical asset allocation
- drawdown management
- moving averages
"""

    (arxiv_dir / "README_ARXIV.md").write_text(readme)
    (arxiv_dir / "ARXIV_METADATA.md").write_text(metadata)


def validate_arxiv_sources(arxiv_dir: Path) -> None:
    tex_path = arxiv_dir / "main.tex"
    if not tex_path.exists():
        raise FileNotFoundError(f"Missing {tex_path}")

    content = tex_path.read_text()
    missing = []
    for line in content.splitlines():
        if "\\includegraphics" in line and "{" in line and "}" in line:
            rel = line.split("{")[-1].split("}")[0]
            p = arxiv_dir / rel
            if not p.exists():
                missing.append(rel)
    if missing:
        raise FileNotFoundError(f"Missing referenced figure(s): {missing}")

    for required in ["main.tex", "main.bbl", "references.bib"]:
        if not (arxiv_dir / required).exists():
            raise FileNotFoundError(f"Missing required arXiv source: {required}")


def build_clean_arxiv_tarball(arxiv_dir: Path, tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if tar_path.exists():
        tar_path.unlink()

    members = [
        "main.tex",
        "main.bbl",
        "references.bib",
        "README_ARXIV.md",
        "ARXIV_METADATA.md",
    ]
    fig_dir = arxiv_dir / "figures"
    fig_members = sorted([Path("figures") / p.name for p in fig_dir.glob("fig_*.png")])

    with tarfile.open(tar_path, "w:gz") as tf:
        for m in members:
            p = arxiv_dir / m
            if p.exists():
                tf.add(p, arcname=str(Path("arxiv_submission") / m))
        for m in fig_members:
            p = arxiv_dir / m
            tf.add(p, arcname=str(Path("arxiv_submission") / m))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh market data snapshot from yfinance")
    args = parser.parse_args()

    set_style()

    out_root = REPO / "output" / "signature_paper"
    snapshot_raw_dir = out_root / "snapshot" / "raw"
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"

    out_root.mkdir(parents=True, exist_ok=True)

    data = {}
    for ticker in ["SPY", "QQQ", "IWM"]:
        raw = fetch_or_load_snapshot(ticker, snapshot_raw_dir, refresh=args.refresh)
        data[ticker] = compute_asset(raw)

    spy, qqq, iwm = data["SPY"], data["QQQ"], data["IWM"]

    # Inference + robustness appendix tables
    inf_tables = build_inference_tables(spy)
    rb_tables = build_robustness_tables(spy)

    # Save canonical tables
    save_tables(spy, qqq, iwm, inf_tables, rb_tables, tables_dir)

    # Figures (canonical)
    make_figures(spy, qqq, iwm, figures_dir)

    # Snapshot metadata
    snapshot_info = {
        "snapshot_refresh": bool(args.refresh),
        "spy_raw_start": str(spy.raw.index.min().date()),
        "spy_raw_end": str(spy.raw.index.max().date()),
        "spy_analysis_start": str(spy.bt.index.min().date()),
        "spy_analysis_end": str(spy.bt.index.max().date()),
        "signal": "Price > SMA50 (long/cash)",
        "regime": "21d annualized volatility, OOS expanding quantiles",
        "cost_bps": DEFAULT_COST_BPS,
        "rebalance": DEFAULT_REBALANCE_FREQ,
    }
    (out_root / "snapshot" / "snapshot_info.json").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "snapshot" / "snapshot_info.json").write_text(json.dumps(snapshot_info, indent=2))

    # Canonical markdown + synced output markdown
    canonical_md = out_root / "paper.md"
    live_md = REPO / "output" / "Volatility_Regimes_Visual_Deconstruction.md"
    write_markdown(spy, qqq, iwm, inf_tables, rb_tables, canonical_md, snapshot_info)
    shutil.copy2(canonical_md, live_md)

    # PDF from synchronized markdown
    live_pdf = REPO / "output" / "Volatility_Regimes_Visual_Deconstruction.pdf"
    subprocess.run(
        [
            "zsh",
            "-ic",
            f"cd {REPO} && pandoc output/Volatility_Regimes_Visual_Deconstruction.md -o output/Volatility_Regimes_Visual_Deconstruction.pdf --pdf-engine=pdflatex",
        ],
        check=True,
    )

    # LaTeX sync for arXiv package
    arxiv_dir = REPO / "output" / "arxiv_submission"
    arxiv_dir.mkdir(parents=True, exist_ok=True)
    write_arxiv_tex(spy, qqq, iwm, inf_tables, rb_tables, arxiv_dir / "main.tex", snapshot_info)

    # Keep references file in arxiv dir (overwrite for deterministic, research-grade cites)
    ref = arxiv_dir / "references.bib"
    ref.write_text(
        """@article{brock1992simple,
  title={Simple Technical Trading Rules and the Stochastic Properties of Stock Returns},
  author={Brock, William and Lakonishok, Josef and LeBaron, Blake},
  journal={The Journal of Finance},
  volume={47},
  number={5},
  pages={1731--1764},
  year={1992},
  doi={10.1111/j.1540-6261.1992.tb04681.x}
}

@article{jegadeesh1993returns,
  title={Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency},
  author={Jegadeesh, Narasimhan and Titman, Sheridan},
  journal={The Journal of Finance},
  volume={48},
  number={1},
  pages={65--91},
  year={1993},
  doi={10.1111/j.1540-6261.1993.tb04702.x}
}

@article{moskowitz2012time,
  title={Time Series Momentum},
  author={Moskowitz, Tobias J. and Ooi, Yao Hua and Pedersen, Lasse Heje},
  journal={Journal of Financial Economics},
  volume={104},
  number={2},
  pages={228--250},
  year={2012},
  doi={10.1016/j.jfineco.2011.11.003}
}

@article{hurst2017century,
  title={A Century of Evidence on Trend-Following Investing},
  author={Hurst, Brian and Ooi, Yao Hua and Pedersen, Lasse Heje},
  journal={The Journal of Portfolio Management},
  volume={44},
  number={1},
  pages={15--29},
  year={2017},
  doi={10.3905/jpm.2017.44.1.015}
}

@book{antonacci2014dual,
  title={Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk},
  author={Antonacci, Gary},
  publisher={McGraw-Hill},
  year={2014}
}
"""
    )
    write_arxiv_submission_notes(arxiv_dir)

    # Sync figures everywhere used
    sync_figures(
        figures_dir,
        [
            REPO / "output" / "figures",
            REPO / "output" / "arxiv_submission" / "figures",
            REPO / "output" / "gemini_paper_repo" / "figures",
        ],
    )

    # Mirror key tables into gemini package for consistency
    gem_tables = REPO / "output" / "gemini_paper_repo" / "data" / "tables"
    gem_tables.mkdir(parents=True, exist_ok=True)
    for f in tables_dir.glob("*.csv"):
        shutil.copy2(f, gem_tables / f.name)

    # Update gemini manuscript from live markdown summary by keeping canonical tex
    shutil.copy2(arxiv_dir / "main.tex", REPO / "output" / "gemini_paper_repo" / "manuscript" / "main.tex")

    # Compile arxiv tex and refresh bundle
    subprocess.run(
        [
            "zsh",
            "-ic",
            f"cd {arxiv_dir} && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex",
        ],
        check=True,
    )
    validate_arxiv_sources(arxiv_dir)
    build_clean_arxiv_tarball(arxiv_dir, REPO / "output" / "arxiv_submission.tar.gz")

    print(f"Canonical paper: {canonical_md}")
    print(f"Live markdown: {live_md}")
    print(f"Live PDF: {live_pdf}")
    print(f"ArXiv tex/pdf updated: {arxiv_dir / 'main.tex'}")
    print(f"Snapshot metadata: {out_root / 'snapshot' / 'snapshot_info.json'}")


if __name__ == "__main__":
    main()
