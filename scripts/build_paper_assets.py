#!/usr/bin/env python3
"""
Build a Gemini-ready research package with reproducible tables, time series,
and figures for the hedgefund-dashboard research paper.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO = Path("/Users/aarjavametha/Desktop/Projects/hedgefund-dashboard")
SRC = REPO / "src"

import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC))

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

OUT = REPO / "output" / "gemini_paper_repo"
TABLE_DIR = OUT / "data" / "tables"
TS_DIR = OUT / "data" / "time_series"
FIG_DIR = OUT / "figures"
MANUSCRIPT_DIR = OUT / "manuscript"
META_DIR = OUT / "metadata"

TICKERS = ["SPY", "QQQ", "IWM"]


def fetch_yf(ticker: str, period: str = "max") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = signals.add_technical_indicators(
        df,
        sma_window=DEFAULT_SMA_WINDOW,
        mom_window=12,
        vol_window=DEFAULT_VOLATILITY_WINDOW,
    )
    df = signals.detect_volatility_regime(
        df,
        vol_col=f"Vol_{DEFAULT_VOLATILITY_WINDOW}d",
        quantile_high=DEFAULT_VOL_QUANTILE_HIGH,
        quantile_low=DEFAULT_VOL_QUANTILE_LOW,
        use_expanding=True,
    )
    df["Signal_Trend"] = np.where(df["Close"] > df[f"SMA_{DEFAULT_SMA_WINDOW}"], 1, 0)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["Close", "Daily_Return", f"SMA_{DEFAULT_SMA_WINDOW}"])
    if df.empty:
        return {}

    bt = backtester.run_backtest(
        df,
        signal_col="Signal_Trend",
        cost_bps=DEFAULT_COST_BPS / 10000,
        rebalance_freq=DEFAULT_REBALANCE_FREQ,
    )

    strat_metrics = backtester.calculate_perf_metrics(
        bt["Equity_Strategy"], include_bootstrap_ci=True, n_bootstrap=DEFAULT_BOOTSTRAP_ITER
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

    sens = regime_analysis.compute_regime_sensitivity(strat_regime)
    boot = regime_analysis.bootstrap_regime_diff(
        bt_valid["Strategy_Net_Return"], bt_valid["Vol_Regime"], metric="Sharpe", n_boot=DEFAULT_BOOTSTRAP_ITER
    )

    wf = backtester.walk_forward_backtest(
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

    return {
        "df": df,
        "bt": bt,
        "strat_metrics": strat_metrics,
        "bench_metrics": bench_metrics,
        "strat_regime": strat_regime,
        "bench_regime": bench_regime,
        "avg_vol": avg_vol,
        "regime_freq": regime_freq,
        "transition_matrix": transition_matrix,
        "transition_stats": transition_stats,
        "sensitivity": sens,
        "bootstrap": boot,
        "walk_forward": wf,
        "sweep": sweep_df,
    }


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def export():
    for d in [TABLE_DIR, TS_DIR, FIG_DIR, MANUSCRIPT_DIR, META_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    results = {}
    for ticker in TICKERS:
        raw = fetch_yf(ticker)
        if raw.empty:
            continue
        prep = prepare_df(raw)
        res = compute_metrics(prep)
        if res:
            results[ticker] = res

    if "SPY" not in results:
        raise RuntimeError("SPY data unavailable")

    spy = results["SPY"]
    sm = spy["strat_metrics"]
    bm = spy["bench_metrics"]
    start_date = str(spy["df"].index.min().date())
    end_date = str(spy["df"].index.max().date())

    # Tables
    uncond = pd.DataFrame(
        {
            "Trend_Strategy": {
                "CAGR": sm.get("CAGR"),
                "Sharpe": sm.get("Sharpe"),
                "MaxDD": sm.get("MaxDD"),
                "Vol": sm.get("Vol"),
                "Sortino": sm.get("Sortino"),
                "Calmar": sm.get("Calmar"),
                "WinRate": sm.get("WinRate"),
                "Sharpe_CI_Lower": sm.get("Sharpe_CI_Lower"),
                "Sharpe_CI_Upper": sm.get("Sharpe_CI_Upper"),
            },
            "Buy_and_Hold": {
                "CAGR": bm.get("CAGR"),
                "Sharpe": bm.get("Sharpe"),
                "MaxDD": bm.get("MaxDD"),
                "Vol": bm.get("Vol"),
                "Sortino": bm.get("Sortino"),
                "Calmar": bm.get("Calmar"),
                "WinRate": bm.get("WinRate"),
                "Sharpe_CI_Lower": np.nan,
                "Sharpe_CI_Upper": np.nan,
            },
        }
    )

    sr = spy["strat_regime"]
    br = spy["bench_regime"]
    av = spy["avg_vol"]

    conditional_rows = []
    for reg in ["Low", "Normal", "High"]:
        if reg not in sr.index:
            continue
        conditional_rows.append(
            {
                "Regime": reg,
                "Avg_Vol_Ann": av.get(reg, np.nan),
                "Strategy_Sharpe": sr.loc[reg, "Sharpe"],
                "Benchmark_Sharpe": br.loc[reg, "Sharpe"] if reg in br.index else np.nan,
                "Strategy_CAGR": sr.loc[reg, "CAGR"],
                "Benchmark_CAGR": br.loc[reg, "CAGR"] if reg in br.index else np.nan,
                "Strategy_WinRate": sr.loc[reg, "WinRate"],
                "Count": sr.loc[reg, "Count"],
            }
        )
    conditional = pd.DataFrame(conditional_rows).set_index("Regime")

    regime_frequency = spy["regime_freq"].to_frame("Frequency")
    transition_matrix = spy["transition_matrix"].reindex(index=["Low", "Normal", "High"], columns=["Low", "Normal", "High"]).fillna(0)
    transition_performance = spy["transition_stats"].sort_index()

    wf_summary = pd.DataFrame()
    wf = spy["walk_forward"]
    if wf and wf.get("summary"):
        s = wf["summary"]
        wf_summary = pd.DataFrame(
            {
                "OOS_CAGR": [s.get("CAGR")],
                "OOS_Sharpe": [s.get("Sharpe")],
                "OOS_MaxDD": [s.get("MaxDD")],
                "OOS_Vol": [s.get("Vol")],
                "OOS_WinRate": [s.get("WinRate")],
                "Periods": [wf.get("n_periods")],
            }
        )

    sweep_df = spy["sweep"]
    sma_sweep = pd.DataFrame()
    if sweep_df is not None and not sweep_df.empty:
        sma_sweep = sweep_df["Sharpe"].reset_index().pivot(index="SMA", columns="Regime", values="Sharpe")

    robust_rows = []
    for ticker in ["QQQ", "IWM"]:
        if ticker not in results:
            continue
        tsm = results[ticker]["strat_metrics"]
        tbm = results[ticker]["bench_metrics"]
        robust_rows.append(
            {
                "Asset": ticker,
                "Trend_CAGR": tsm.get("CAGR"),
                "Trend_Sharpe": tsm.get("Sharpe"),
                "Trend_MaxDD": tsm.get("MaxDD"),
                "BuyHold_CAGR": tbm.get("CAGR"),
                "BuyHold_Sharpe": tbm.get("Sharpe"),
                "BuyHold_MaxDD": tbm.get("MaxDD"),
            }
        )
    robustness = pd.DataFrame(robust_rows).set_index("Asset")

    regime_sensitivity = pd.DataFrame(
        {
            "Metric": ["Sharpe_Diff_High_minus_Normal", "CAGR_Diff_High_minus_Normal", "Bootstrap_Diff", "Bootstrap_PValue"],
            "Value": [
                spy["sensitivity"].get("Sharpe_Diff"),
                spy["sensitivity"].get("CAGR_Diff"),
                spy["bootstrap"].get("diff"),
                spy["bootstrap"].get("p_value"),
            ],
        }
    )

    table_files = {
        "unconditional_performance.csv": uncond,
        "conditional_performance_by_regime.csv": conditional,
        "regime_frequency.csv": regime_frequency,
        "regime_transition_matrix.csv": transition_matrix,
        "regime_transition_performance.csv": transition_performance,
        "walk_forward_oos_summary.csv": wf_summary,
        "sma_sweep_sharpe_by_regime.csv": sma_sweep,
        "robustness_assets_unconditional.csv": robustness,
        "regime_sensitivity_bootstrap.csv": regime_sensitivity,
    }

    for filename, df in table_files.items():
        df.to_csv(TABLE_DIR / filename, float_format="%.6f")

    # Time series exports
    ts_cols = [
        "Close",
        "Daily_Return",
        f"Vol_{DEFAULT_VOLATILITY_WINDOW}d",
        "Vol_Regime",
        f"SMA_{DEFAULT_SMA_WINDOW}",
        "Signal_Trend",
    ]
    bt_cols = [
        "Position",
        "Strategy_Return",
        "Cost",
        "Strategy_Net_Return",
        "Equity_Benchmark",
        "Equity_Strategy",
        "DD_Benchmark",
        "DD_Strategy",
    ]

    for ticker, res in results.items():
        merged = pd.concat([res["df"][ts_cols], res["bt"][bt_cols]], axis=1)
        merged.index.name = "Date"
        merged.to_csv(TS_DIR / f"{ticker.lower()}_backtest_daily.csv", float_format="%.8f")

    # Figure copies
    figure_map = [
        ("fig_equity_curves.png", "SPY equity curves (log scale)."),
        ("fig_drawdowns.png", "SPY drawdown curves."),
        ("fig_regime_frequency.png", "Regime frequency for SPY using OOS quantiles."),
        ("fig_conditional_sharpe.png", "Strategy vs benchmark Sharpe by volatility regime."),
        ("fig_transition_matrix.png", "Volatility regime transition probability matrix."),
        ("fig_sma_sweep.png", "SMA window robustness for regime Sharpe."),
        ("fig_robustness_assets.png", "Cross-asset robustness for CAGR and Sharpe (QQQ and IWM)."),
    ]

    src_fig_root = REPO / "output" / "figures"
    figure_index_rows = []
    for filename, caption in figure_map:
        copied = copy_if_exists(src_fig_root / filename, FIG_DIR / filename)
        if copied:
            figure_index_rows.append({"filename": filename, "caption": caption})
    pd.DataFrame(figure_index_rows).to_csv(META_DIR / "figure_index.csv", index=False)

    # Manuscript source copies
    copy_if_exists(REPO / "output" / "arxiv_submission" / "main.tex", MANUSCRIPT_DIR / "main.tex")
    copy_if_exists(REPO / "output" / "arxiv_submission" / "references.bib", MANUSCRIPT_DIR / "references.bib")

    summary = {
        "report_date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "primary_asset": "SPY",
        "robustness_assets": ["QQQ", "IWM"],
        "sample_start": start_date,
        "sample_end": end_date,
        "signal": f"Long when Close > SMA_{DEFAULT_SMA_WINDOW}; else cash",
        "vol_regime": {
            "vol_window_days": DEFAULT_VOLATILITY_WINDOW,
            "quantile_low": DEFAULT_VOL_QUANTILE_LOW,
            "quantile_high": DEFAULT_VOL_QUANTILE_HIGH,
            "classification": "expanding-window out-of-sample quantiles",
        },
        "backtest": {
            "rebalance": DEFAULT_REBALANCE_FREQ,
            "transaction_cost_bps": DEFAULT_COST_BPS,
            "bootstrap_iterations": DEFAULT_BOOTSTRAP_ITER,
        },
        "key_results": {
            "trend_cagr": sm.get("CAGR"),
            "buyhold_cagr": bm.get("CAGR"),
            "trend_sharpe": sm.get("Sharpe"),
            "buyhold_sharpe": bm.get("Sharpe"),
            "trend_maxdd": sm.get("MaxDD"),
            "buyhold_maxdd": bm.get("MaxDD"),
            "high_minus_normal_sharpe_diff": spy["bootstrap"].get("diff"),
            "high_minus_normal_sharpe_pvalue": spy["bootstrap"].get("p_value"),
        },
    }
    (META_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    export()
    print(f"Built Gemini package at: {OUT}")
