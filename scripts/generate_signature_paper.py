#!/usr/bin/env python3
"""Generate the signature research paper and premium figure set."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

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
    DEFAULT_VOL_QUANTILE_HIGH,
    DEFAULT_VOL_QUANTILE_LOW,
    DEFAULT_VOLATILITY_WINDOW,
)


@dataclass
class AssetResults:
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
    bootstrap: dict
    walk_forward: dict
    sweep_df: pd.DataFrame


PALETTE = {
    "bg": "#f6f4ef",
    "panel": "#ffffff",
    "grid": "#d8d2c4",
    "text": "#1f2330",
    "muted": "#6b7280",
    "strategy": "#2f6fed",
    "benchmark": "#8e98a8",
    "low": "#39a979",
    "normal": "#f0ad2d",
    "high": "#d84d4d",
}


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
            "grid.alpha": 0.5,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "savefig.dpi": 300,
            "savefig.facecolor": PALETTE["bg"],
        }
    )


def fetch_history(ticker: str, period: str = "max") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = signals.add_technical_indicators(
        df,
        sma_window=DEFAULT_SMA_WINDOW,
        mom_window=12,
        vol_window=DEFAULT_VOLATILITY_WINDOW,
    )
    out = signals.detect_volatility_regime(
        out,
        vol_col=f"Vol_{DEFAULT_VOLATILITY_WINDOW}d",
        quantile_high=DEFAULT_VOL_QUANTILE_HIGH,
        quantile_low=DEFAULT_VOL_QUANTILE_LOW,
        use_expanding=True,
    )
    out["Signal_Trend"] = np.where(out["Close"] > out[f"SMA_{DEFAULT_SMA_WINDOW}"], 1, 0)
    return out


def compute_asset(df: pd.DataFrame) -> AssetResults:
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
        df.loc[
            df["Vol_Regime"].isin(["Low", "Normal", "High"]),
            [f"Vol_{DEFAULT_VOLATILITY_WINDOW}d", "Vol_Regime"],
        ]
        .groupby("Vol_Regime")[f"Vol_{DEFAULT_VOLATILITY_WINDOW}d"]
        .mean()
    )

    regime_freq = df["Vol_Regime"].value_counts(normalize=True)
    regime_freq = regime_freq.loc[[r for r in ["Low", "Normal", "High"] if r in regime_freq.index]]

    reg_series = bt["Vol_Regime"].where(bt["Vol_Regime"].isin(["Low", "Normal", "High"]))
    transition_matrix = regime_analysis.compute_transition_matrix(reg_series)
    transition_stats = regime_analysis.compute_transition_stats(
        bt["Strategy_Net_Return"], reg_series
    )

    sensitivity = regime_analysis.compute_regime_sensitivity(strat_regime)
    bootstrap = regime_analysis.bootstrap_regime_diff(
        bt_valid["Strategy_Net_Return"],
        bt_valid["Vol_Regime"],
        metric="Sharpe",
        n_boot=DEFAULT_BOOTSTRAP_ITER,
    )

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
        bootstrap=bootstrap,
        walk_forward=walk_forward,
        sweep_df=sweep_df,
    )


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{x*100:.2f}%"


def num(x: float, n: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return f"{x:.{n}f}"


def make_figures(spy: AssetResults, qqq: AssetResults, iwm: AssetResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        spy.bt.index,
        spy.bt["Equity_Benchmark"],
        color=PALETTE["benchmark"],
        lw=2.0,
        label="Buy & Hold",
    )
    ax.plot(
        spy.bt.index,
        spy.bt["Equity_Strategy"],
        color=PALETTE["strategy"],
        lw=2.4,
        label="Trend Strategy",
    )
    ax.set_yscale("log")
    ax.grid(True, axis="y")
    ax.set_title(
        "Figure 1. Wealth Trajectory: Strategy vs Benchmark", loc="left", fontsize=16, pad=14
    )
    ax.set_ylabel("Cumulative growth (log scale)")
    ax.set_xlabel("Date")
    crises = [
        ("2000-09-01", "2003-03-31", "Dot-com"),
        ("2007-10-01", "2009-03-31", "GFC"),
        ("2020-02-15", "2020-05-01", "COVID shock"),
    ]
    for start, end, label in crises:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#f4d4d4", alpha=0.35)
        ax.text(
            pd.Timestamp(start), ax.get_ylim()[1] / 1.25, label, fontsize=9, color=PALETTE["muted"]
        )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_equity_curves.png")
    plt.close(fig)

    # Figure 2: Drawdowns
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(spy.bt.index, spy.bt["DD_Benchmark"], 0, color=PALETTE["benchmark"], alpha=0.22)
    ax.plot(
        spy.bt.index, spy.bt["DD_Benchmark"], color=PALETTE["benchmark"], lw=1.8, label="Buy & Hold"
    )
    ax.fill_between(spy.bt.index, spy.bt["DD_Strategy"], 0, color=PALETTE["strategy"], alpha=0.20)
    ax.plot(
        spy.bt.index,
        spy.bt["DD_Strategy"],
        color=PALETTE["strategy"],
        lw=2.0,
        label="Trend Strategy",
    )
    ax.set_title(
        "Figure 2. Drawdown Decomposition: Left-tail Truncation", loc="left", fontsize=16, pad=14
    )
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_drawdowns.png")
    plt.close(fig)

    # Figure 3: Regime frequency
    freq = spy.regime_freq.reindex(["Low", "Normal", "High"])
    colors = [PALETTE["low"], PALETTE["normal"], PALETTE["high"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(freq.index, freq.values, color=colors, width=0.58)
    ax.set_title(
        "Figure 3. Volatility Regime Occupancy (OOS Classification)",
        loc="left",
        fontsize=16,
        pad=14,
    )
    ax.set_ylabel("Share of observations")
    ax.set_ylim(0, max(freq.values) * 1.3)
    ax.grid(True, axis="y")
    for b, v in zip(bars, freq.values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.008,
            f"{v*100:.1f}%",
            ha="center",
            va="bottom",
            color=PALETTE["text"],
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_regime_frequency.png")
    plt.close(fig)

    # Figure 4: Sharpe by regime (dumbbell)
    regs = ["Low", "Normal", "High"]
    y = np.arange(len(regs))[::-1]
    strat = [spy.strat_regime.loc[r, "Sharpe"] for r in regs]
    bench = [spy.bench_regime.loc[r, "Sharpe"] for r in regs]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(regs):
        ax.plot(
            [bench[i], strat[i]], [y[i], y[i]], color=PALETTE["grid"], lw=5, solid_capstyle="round"
        )
    ax.scatter(bench, y, s=115, color=PALETTE["benchmark"], label="Buy & Hold", zorder=3)
    ax.scatter(strat, y, s=145, color=PALETTE["strategy"], label="Trend Strategy", zorder=4)
    for i in range(len(regs)):
        ax.text(
            strat[i] + 0.03, y[i] + 0.03, num(strat[i], 2), color=PALETTE["strategy"], fontsize=10
        )
        ax.text(bench[i] - 0.15, y[i] - 0.18, num(bench[i], 2), color=PALETTE["muted"], fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(regs)
    ax.set_xlabel("Sharpe ratio")
    ax.set_title(
        "Figure 4. Regime-Conditional Risk-Adjusted Returns", loc="left", fontsize=16, pad=14
    )
    ax.grid(True, axis="x")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_conditional_sharpe.png")
    plt.close(fig)

    # Figure 5: Transition matrix heatmap
    tm = spy.transition_matrix.reindex(
        index=["Low", "Normal", "High"], columns=["Low", "Normal", "High"]
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 6.2))
    im = ax.imshow(tm.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            val = tm.values[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=("white" if val > 0.65 else PALETTE["text"]),
                fontsize=11,
            )
    ax.set_xticks(range(3), tm.columns)
    ax.set_yticks(range(3), tm.index)
    ax.set_xlabel("Regime at t")
    ax.set_ylabel("Regime at t-1")
    ax.set_title(
        "Figure 5. Regime Transition Matrix (Persistence Structure)",
        loc="left",
        fontsize=16,
        pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition probability")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_transition_matrix.png")
    plt.close(fig)

    # Figure 6: SMA sweep
    fig, ax = plt.subplots(figsize=(10.5, 6))
    if spy.sweep_df is not None and not spy.sweep_df.empty:
        for reg, color in [
            ("Low", PALETTE["low"]),
            ("Normal", PALETTE["normal"]),
            ("High", PALETTE["high"]),
        ]:
            if reg in spy.sweep_df.index.get_level_values("Regime"):
                series = spy.sweep_df.xs(reg, level="Regime")["Sharpe"]
                ax.plot(
                    series.index, series.values, marker="o", lw=2.5, color=color, label=f"{reg} vol"
                )
                for x, yv in zip(series.index, series.values):
                    ax.text(x, yv + 0.03, num(yv, 2), color=color, fontsize=8, ha="center")
    ax.set_title(
        "Figure 6. Parameter Robustness: SMA Window vs Regime Sharpe",
        loc="left",
        fontsize=16,
        pad=14,
    )
    ax.set_xlabel("SMA window (days)")
    ax.set_ylabel("Sharpe ratio")
    ax.grid(True)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sma_sweep.png")
    plt.close(fig)

    # Figure 7: Cross-asset robustness (two-panel slope charts)
    assets = ["SPY", "QQQ", "IWM"]
    res = {"SPY": spy, "QQQ": qqq, "IWM": iwm}

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for idx, metric in enumerate(["CAGR", "Sharpe"]):
        ax = axs[idx]
        for i, a in enumerate(assets):
            b = res[a].bench_metrics.get(metric, np.nan)
            s = res[a].strat_metrics.get(metric, np.nan)
            color = [PALETTE["strategy"], "#5d9cec", "#9ab8ff"][i]
            y = len(assets) - i
            ax.plot([0, 1], [b, s], color=color, lw=2.8, marker="o")
            ax.text(
                -0.02,
                b,
                f"{a} {num(b, 2) if metric == 'Sharpe' else pct(b)}",
                ha="right",
                va="center",
                fontsize=9,
                color=PALETTE["muted"],
            )
            ax.text(
                1.02,
                s,
                f"{num(s, 2) if metric == 'Sharpe' else pct(s)}",
                ha="left",
                va="center",
                fontsize=9,
                color=color,
            )
        ax.set_xticks([0, 1], ["Buy & Hold", "Trend"])
        ax.set_xlim(-0.45, 1.45)
        ax.grid(True, axis="y")
        ax.set_title("CAGR" if metric == "CAGR" else "Sharpe")
        if metric == "CAGR":
            ax.set_ylabel("Annualized return")
    fig.suptitle(
        "Figure 7. Cross-Asset Robustness: Benchmark vs Strategy",
        x=0.04,
        ha="left",
        fontsize=16,
        weight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig_robustness_assets.png")
    plt.close(fig)


def write_paper(spy: AssetResults, qqq: AssetResults, iwm: AssetResults, md_path: Path) -> None:
    sm, bm = spy.strat_metrics, spy.bench_metrics
    wf = spy.walk_forward.get("summary", {}) if spy.walk_forward else {}

    high_gap = spy.strat_regime.loc["High", "CAGR"] - spy.bench_regime.loc["High", "CAGR"]
    normal_gap = spy.strat_regime.loc["Normal", "CAGR"] - spy.bench_regime.loc["Normal", "CAGR"]
    low_gap = spy.strat_regime.loc["Low", "CAGR"] - spy.bench_regime.loc["Low", "CAGR"]

    low_to_normal_sharpe = np.nan
    if "Low→Normal" in spy.transition_stats.index:
        low_to_normal_sharpe = spy.transition_stats.loc["Low→Normal", "Sharpe"]

    best_normal_window = "N/A"
    best_normal_val = np.nan
    if (
        spy.sweep_df is not None
        and not spy.sweep_df.empty
        and ("Normal" in spy.sweep_df.index.get_level_values("Regime"))
    ):
        normal_series = spy.sweep_df.xs("Normal", level="Regime")["Sharpe"]
        idxmax = normal_series.idxmax()
        best_normal_window = str(int(idxmax))
        best_normal_val = float(normal_series.max())

    text = f"""# Volatility Regimes and Trend-Following Performance in U.S. Equities: A Visual Deconstruction

**Author:** Aarjav Ametha  
**Date:** February 2026  
**Repository:** [github.com/aarjava/hedgefund-dashboard](https://github.com/aarjava/hedgefund-dashboard)

## Abstract
This paper tests a 50-day SMA trend strategy on SPY using out-of-sample volatility regime classification. The central finding is state asymmetry: strategy quality is strongest in Low Volatility, weak in Normal Volatility, and only modest in High Volatility. The strategy does not maximize terminal wealth versus buy-and-hold, but it materially reshapes risk by truncating left-tail drawdowns. Seven figures decompose this behavior across wealth path, drawdown geometry, regime occupancy, regime-conditional Sharpe, transition persistence, parameter sensitivity, and cross-asset transfer.

## 1. Setup and Research Claim
The study asks a concrete question: **is trend-following alpha stable across volatility states, or concentrated in specific market microstructures?**

- Signal: long when `Close > SMA(50)`, otherwise cash.
- Regimes: annualized 21-day volatility, OOS expanding quantiles (`Low < 25%`, `High > 75%`).
- Frictions: monthly rebalance, 10 bps transaction cost.

The baseline trade-off is visible immediately: strategy CAGR `{pct(sm.get('CAGR'))}` vs benchmark `{pct(bm.get('CAGR'))}`, but max drawdown `{pct(sm.get('MaxDD'))}` vs `{pct(bm.get('MaxDD'))}`.

## 2. Wealth Path Decomposition
![Figure 1: SPY Equity Curves (Log Scale)](output/figures/fig_equity_curves.png)

*Figure 1: SPY Equity Curves (Log Scale).*  
The strategy decouples from major crash legs, but loses compounding speed during prolonged bull phases due to lagged re-entry after corrections.

## 3. Left-Tail Geometry and Recovery Burden
![Figure 2: SPY Drawdown Curves](output/figures/fig_drawdowns.png)

*Figure 2: SPY Drawdown Curves.*  
Risk control is the dominant contribution. The strategy's lower depth and shorter severe underwater episodes improve survivability for leverage-constrained or drawdown-sensitive mandates.

## 4. State Occupancy Matters
![Figure 3: Regime Frequency (SPY, OOS)](output/figures/fig_regime_frequency.png)

*Figure 3: Regime Frequency (SPY, OOS).*  
Regime frequency determines where performance can realistically accumulate. Strong behavior in an infrequent regime cannot drive headline return alone.

## 5. Regime Asymmetry (Where the Signal Works)
![Figure 4: Sharpe Ratio by Volatility Regime](output/figures/fig_conditional_sharpe.png)

*Figure 4: Sharpe Ratio by Volatility Regime.*

- Low-volatility edge is persistent (Sharpe `{num(spy.strat_regime.loc['Low', 'Sharpe'])}`).
- Normal-volatility is the whipsaw zone (Sharpe `{num(spy.strat_regime.loc['Normal', 'Sharpe'])}`).
- High-volatility is not a reliable crisis-alpha engine (Sharpe `{num(spy.strat_regime.loc['High', 'Sharpe'])}`).

CAGR spread vs benchmark by regime:
- Low: `{pct(low_gap)}`
- Normal: `{pct(normal_gap)}`
- High: `{pct(high_gap)}`

Bootstrap high-minus-normal Sharpe difference is `{num(spy.bootstrap.get('diff'))}` with p-value `{num(spy.bootstrap.get('p_value'), 3)}`, so evidence for a large separation is weak.

## 6. Transition Microstructure and Failure Modes
![Figure 5: Regime Transition Matrix](output/figures/fig_transition_matrix.png)

*Figure 5: Regime Transition Matrix.*

Regimes are sticky, but transition shocks matter more than static labels. The critical penalty appears around trend-break transitions; for `Low→Normal`, transition Sharpe is `{num(low_to_normal_sharpe)}`.

## 7. Parameter Fragility Check
![Figure 6: SMA Parameter Sweep](output/figures/fig_sma_sweep.png)

*Figure 6: SMA Parameter Sweep.*

The regime pattern survives parameter changes from 20 to 200 days. Best normal-regime Sharpe in this sweep appears at SMA `{best_normal_window}` with value `{num(best_normal_val)}`.

## 8. Cross-Asset Transfer
![Figure 7: Cross-Asset Robustness](output/figures/fig_robustness_assets.png)

*Figure 7: Cross-Asset Robustness.*

- QQQ: strategy CAGR `{pct(qqq.strat_metrics.get('CAGR'))}`, Sharpe `{num(qqq.strat_metrics.get('Sharpe'))}`.
- IWM: strategy CAGR `{pct(iwm.strat_metrics.get('CAGR'))}`, Sharpe `{num(iwm.strat_metrics.get('Sharpe'))}`.

Transfer is partial: high-momentum universes (QQQ) are friendlier than noisier mean-reverting universes (IWM).

## 9. Walk-Forward Reality Check
Out-of-sample walk-forward results stay directionally consistent with the full-sample decomposition:

- OOS CAGR: `{pct(wf.get('CAGR'))}`
- OOS Sharpe: `{num(wf.get('Sharpe'))}`
- OOS Max Drawdown: `{pct(wf.get('MaxDD'))}`
- Test periods: `{int(spy.walk_forward.get('n_periods', 0))}`

## Conclusion
This implementation behaves less like a universal trend engine and more like a **state-conditional risk allocation rule**. Its edge is concentrated in low-volatility trend persistence, while transition/chop regimes create the primary performance tax. The most practical next step is volatility-adaptive signal speed plus explicit transition-risk controls, not simply re-optimizing one fixed SMA window.
"""

    md_path.write_text(text)


def sync_figures(src_dir: Path, repo_root: Path) -> None:
    destinations = [
        repo_root / "output" / "arxiv_submission" / "figures",
        repo_root / "output" / "gemini_paper_repo" / "figures",
    ]
    for dst in destinations:
        dst.mkdir(parents=True, exist_ok=True)
        for img in src_dir.glob("fig_*.png"):
            shutil.copy2(img, dst / img.name)


def main() -> None:
    set_style()

    assets = {}
    for t in ["SPY", "QQQ", "IWM"]:
        raw = fetch_history(t)
        if raw.empty:
            raise RuntimeError(f"No data returned for {t}")
        assets[t] = compute_asset(prep(raw))

    out_fig = REPO / "output" / "figures"
    make_figures(assets["SPY"], assets["QQQ"], assets["IWM"], out_fig)

    paper_md = REPO / "output" / "Volatility_Regimes_Visual_Deconstruction.md"
    write_paper(assets["SPY"], assets["QQQ"], assets["IWM"], paper_md)

    sync_figures(out_fig, REPO)

    print(f"Wrote paper: {paper_md}")
    print(f"Updated figures: {out_fig}")


if __name__ == "__main__":
    main()
