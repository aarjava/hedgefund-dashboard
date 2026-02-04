import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import custom modules
try:
    from modules import backtester, data_model, signals
    from modules.config import (
        DEFAULT_COST_BPS,
        DEFAULT_MOMENTUM_WINDOW,
        DEFAULT_SMA_WINDOW,
        DEFAULT_VOL_QUANTILE_HIGH,
        MIN_DATA_POINTS,
        PRESET_UNIVERSE,
    )
except ImportError:
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.modules import backtester, data_model, signals
    from src.modules.config import (
        DEFAULT_COST_BPS,
        DEFAULT_MOMENTUM_WINDOW,
        DEFAULT_SMA_WINDOW,
        DEFAULT_VOL_QUANTILE_HIGH,
        MIN_DATA_POINTS,
        PRESET_UNIVERSE,
    )


def get_cache_key(*args) -> str:
    """Generate a hash key for caching based on input parameters."""
    key_str = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


# Initialize session state for caching expensive computations
if "computed_signals" not in st.session_state:
    st.session_state.computed_signals = {}
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = {}


# --- Configuration ---
st.set_page_config(
    page_title="Quantitative Research Dashboard",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Styling ---
st.markdown(
    """
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .bullish { color: #00ff00; font-weight: bold; }
    .bearish { color: #ff4b4b; font-weight: bold; }
    .neutral { color: #888888; font-weight: bold; }
    .regime-high { color: #ff4b4b; font-weight: bold; }
    .regime-low { color: #00ff00; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("🎛️ Research Config")

    st.subheader("1. Asset Selection")
    t_mode = st.radio("Selection Mode", ["Preset Universe", "Custom Ticker"], horizontal=True)

    if t_mode == "Preset Universe":
        ticker = st.selectbox("Symbol", PRESET_UNIVERSE, index=0)
    else:
        ticker = st.text_input("Enter Symbol (Yahoo Finance)", value="NVDA").upper()

    st.subheader("2. Time Horizon")
    date_mode = st.selectbox("Date Range", ["Last 5 Years", "Last 10 Years", "Max", "Custom"])

    if date_mode == "Custom":
        d_col1, d_col2 = st.columns(2)
        start_date = d_col1.date_input("Start", value=datetime.today() - timedelta(days=365 * 2))
        end_date = d_col2.date_input("End", value=datetime.today())
        period_arg = "max"
    else:
        period_map = {"Last 5 Years": "5y", "Last 10 Years": "10y", "Max": "max"}
        period_arg = period_map[date_mode]

    st.subheader("3. Signal Parameters")
    sma_window = st.slider(
        "Trend SMA Window",
        10,
        200,
        DEFAULT_SMA_WINDOW,
        10,
        help="Lookback days for Simple Moving Average trend signal.",
    )
    mom_window = st.slider(
        "Momentum Lookback (Months)",
        1,
        24,
        DEFAULT_MOMENTUM_WINDOW,
        1,
        help="Lookback months for Momentum signal.",
    )

    st.markdown("---")
    st.subheader("4. Research Rigor")
    use_oos = st.toggle(
        "Out-of-Sample Mode",
        value=False,
        help="Uses expanding-window quantiles for regime classification to avoid look-ahead bias. Enable for rigorous backtesting.",
    )
    if use_oos:
        st.success("✓ Look-ahead bias removed")
    else:
        st.info("Using full-sample quantiles (exploratory mode)")

    vol_q_high = st.slider(
        "High Volatility Quantile",
        0.5,
        0.95,
        DEFAULT_VOL_QUANTILE_HIGH,
        0.05,
        help="Threshold to define 'High Volatility'. E.g., 0.75 means top 25% of volatility readings.",
    )

    st.subheader("5. Backtest Settings")
    bt_cost = (
        st.number_input(
            "Transaction Cost (bps)",
            value=DEFAULT_COST_BPS,
            step=1,
            help="Transaction cost in basis points. E.g., 10 bps = 0.10%.",
        )
        / 10000
    )
    allow_short = st.checkbox("Allow Short Selling?", value=False)


# --- Data Ingestion ---
with st.spinner(f"Fetching data for {ticker}..."):
    raw_df = data_model.fetch_stock_data(ticker, period=period_arg)

if raw_df.empty:
    st.error(f"Could not load data for {ticker}. Please check the symbol.")
    st.stop()

# Filter by date for custom range
if date_mode == "Custom":
    mask = (raw_df.index.date >= start_date) & (raw_df.index.date <= end_date)
    df = raw_df.loc[mask].copy()
else:
    df = raw_df.copy()

if len(df) < MIN_DATA_POINTS:
    st.warning(
        f"Not enough data points for selected range/period (need at least {MIN_DATA_POINTS})."
    )
    st.stop()

# --- Signal Calculation (with session state caching) ---
signal_cache_key = get_cache_key(ticker, period_arg, sma_window, mom_window, date_mode)

if signal_cache_key not in st.session_state.computed_signals:
    with st.spinner("Computing technical indicators..."):
        computed_df = signals.add_technical_indicators(
            df, sma_window=sma_window, mom_window=mom_window
        )
        st.session_state.computed_signals[signal_cache_key] = computed_df

df = st.session_state.computed_signals[signal_cache_key].copy()

# --- Regime Detection ---
# Using 21-day annualized vol with option for out-of-sample analysis
df = signals.detect_volatility_regime(
    df,
    vol_col="Vol_21d",
    quantile_high=vol_q_high,
    quantile_low=0.25,
    use_expanding=use_oos,  # Toggle between in-sample and out-of-sample
)

# --- Dashboard Header ---
st.markdown("## 🔍 Research Question")
st.markdown(
    "> **How sensitive is trend-following performance to volatility regimes in US equities?**"
)

latest = df.iloc[-1]
prev = df.iloc[-2]
chg_pct = latest["Daily_Return"]

h1, h2, h3, h4 = st.columns(4)
h1.metric("Asset", f"{ticker} (${latest['Close']:.2f})", f"{chg_pct:.2%}")
h2.metric("Current Regime", latest["Vol_Regime"])
h3.metric(f"Volatility ({vol_q_high:.0%}-tile)", f"{latest['Vol_21d']:.2%}")
h4.metric("Trend Status", "BULLISH" if latest["Close"] > latest[f"SMA_{sma_window}"] else "BEARISH")

# --- Tabs ---
tab_ov, tab_regime, tab_bt, tab_rep = st.tabs(
    ["📈 Overview", "🌪️ Regime Analysis", "🧪 Backtest Engine", "📄 Report"]
)

# --- TAB 1: OVERVIEW ---
with tab_ov:
    # Interactive Price Chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Close Price", line={"color": "white", "width": 1})
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"SMA_{sma_window}"],
            name=f"{sma_window}-Day SMA",
            line={"color": "#ff9f43", "width": 1},
        )
    )

    # Highlight High Volatility Regimes
    # Filter high vol periods
    high_vol_mask = df["Vol_Regime"] == "High"
    # We can plot markers or shade areas. Shading is valid but tricky in Plotly without shapes list.
    # Let's plot points
    high_vol_pts = df[high_vol_mask]
    fig.add_trace(
        go.Scatter(
            x=high_vol_pts.index,
            y=high_vol_pts["Close"],
            mode="markers",
            name="High Volatility",
            marker={"color": "red", "size": 2},
        )
    )

    fig.update_layout(
        title=f"{ticker} Price History & Regime Context",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red dots indicate days classified as 'High Volatility' regime.")

# --- TAB 2: REGIME ANALYSIS ---
with tab_regime:
    st.subheader("Volatility Regime Classification")

    c1, c2 = st.columns(2)
    with c1:
        # Scatter: Vol vs Returns needed? Maybe just distribution
        fig_hist = px.histogram(
            df,
            x="Vol_21d",
            color="Vol_Regime",
            nbins=50,
            title="Volatility Distribution",
            template="plotly_dark",
            color_discrete_map={"High": "#ff4b4b", "Low": "#00ff00", "Normal": "#888888"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        # Pie chart of time spent in regimes
        regime_counts = df["Vol_Regime"].value_counts()
        fig_pie = px.pie(
            values=regime_counts,
            names=regime_counts.index,
            title="Time Spent in Regimes",
            template="plotly_dark",
            color=regime_counts.index,
            color_discrete_map={"High": "#ff4b4b", "Low": "#00ff00", "Normal": "#888888"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Regime Characteristics")
    stats = df.groupby("Vol_Regime")[["Daily_Return", "Vol_21d"]].mean()
    # Annualize return
    stats["Ann_Return"] = stats["Daily_Return"] * 252
    st.dataframe(stats.style.format("{:.2%}"))

# --- TAB 3: BACKTEST ---
with tab_bt:
    st.subheader("Strategy Simulation")

    # Out-of-sample mode indicator
    if use_oos:
        st.success(
            "🔬 **Out-of-Sample Mode Active** - Regime classification uses only past data at each point"
        )

    # Define Strategy
    # Trend Following
    df["Signal_Trend"] = np.where(
        df["Close"] > df[f"SMA_{sma_window}"], 1, -1 if allow_short else 0
    )

    # Run Backtest (with session state caching)
    bt_cache_key = get_cache_key(signal_cache_key, bt_cost, allow_short, use_oos, vol_q_high)

    if bt_cache_key not in st.session_state.backtest_results:
        with st.spinner("Running backtest simulation..."):
            res_df = backtester.run_backtest(
                df, "Signal_Trend", cost_bps=bt_cost, rebalance_freq="M"
            )
            st.session_state.backtest_results[bt_cache_key] = res_df

    res_df = st.session_state.backtest_results[bt_cache_key]

    if not res_df.empty:
        # Add Regime to Backtest Results (forward fill valid for analysis)
        res_df["Vol_Regime"] = df["Vol_Regime"]

        # 1. Global Metrics with Bootstrap CI
        strat_metrics = backtester.calculate_perf_metrics(
            res_df["Equity_Strategy"], include_bootstrap_ci=True, n_bootstrap=500
        )
        bench_metrics = backtester.calculate_perf_metrics(res_df["Equity_Benchmark"])

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Global CAGR", f"{strat_metrics['CAGR']:.2%}")

        # Show Sharpe with CI if available
        sharpe_display = f"{strat_metrics['Sharpe']:.2f}"
        if strat_metrics.get("Sharpe_CI_Lower") is not None:
            sharpe_display += (
                f" [{strat_metrics['Sharpe_CI_Lower']:.2f}, {strat_metrics['Sharpe_CI_Upper']:.2f}]"
            )
        col_m2.metric("Sharpe (95% CI)", sharpe_display)

        col_m3.metric("Max Drawdown", f"{strat_metrics['MaxDD']:.2%}")
        col_m4.metric("Max DD Duration", f"{strat_metrics.get('MaxDD_Duration', 0)} days")

        # Additional metrics row
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        col_a1.metric("Sortino", f"{strat_metrics.get('Sortino', 0):.2f}")
        col_a2.metric("Calmar", f"{strat_metrics.get('Calmar', 0):.2f}")
        col_a3.metric("Win Rate", f"{strat_metrics.get('WinRate', 0):.1%}")
        col_a4.metric("Avg DD Duration", f"{strat_metrics.get('AvgDD_Duration', 0):.0f} days")

        # 2. Equity Curve
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=res_df.index,
                y=res_df["Equity_Strategy"],
                name="Trend Strategy",
                line={"color": "#00ff00"},
            )
        )
        fig_eq.add_trace(
            go.Scatter(
                x=res_df.index,
                y=res_df["Equity_Benchmark"],
                name="Buy & Hold",
                line={"color": "gray", "dash": "dot"},
            )
        )
        fig_eq.update_layout(title="Equity Curve", template="plotly_dark", height=400)
        st.plotly_chart(fig_eq, use_container_width=True)

        # 3. Drawdown Chart
        with st.expander("📉 Drawdown Analysis", expanded=False):
            fig_dd = go.Figure()
            fig_dd.add_trace(
                go.Scatter(
                    x=res_df.index,
                    y=res_df["DD_Strategy"] * 100,
                    name="Strategy Drawdown",
                    fill="tozeroy",
                    line={"color": "#ff4b4b"},
                )
            )
            fig_dd.add_trace(
                go.Scatter(
                    x=res_df.index,
                    y=res_df["DD_Benchmark"] * 100,
                    name="Benchmark Drawdown",
                    line={"color": "gray", "dash": "dot"},
                )
            )
            fig_dd.update_layout(
                title="Underwater Equity (Drawdown %)",
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=300,
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        # 4. Conditional Analysis
        st.markdown("### 🔬 Conditional Performance by Regime")
        st.info("Does the strategy outperform during High Volatility?")

        cond_stats = backtester.calculate_conditional_stats(
            res_df, "Strategy_Net_Return", "Vol_Regime"
        )

        # Add Benchmark Conditional Stats for comparison
        bench_cond = backtester.calculate_conditional_stats(res_df, "Daily_Return", "Vol_Regime")

        # Merge
        comparison = pd.concat(
            [cond_stats.add_suffix("_Strat"), bench_cond.add_suffix("_Bench")], axis=1
        )

        # Reorder columns - handle missing columns gracefully
        available_cols = []
        for col in [
            "Ann_Return_Strat",
            "Ann_Return_Bench",
            "Sharpe_Strat",
            "Sharpe_Bench",
            "WinRate_Strat",
        ]:
            if col in comparison.columns:
                available_cols.append(col)
        comparison = comparison[available_cols]

        st.dataframe(
            comparison.style.background_gradient(
                cmap="RdYlGn", subset=["Ann_Return_Strat", "Sharpe_Strat"]
            ).format("{:.2f}")
        )

        st.markdown(
            "**Key Insight:** Compare 'Sharpe_Strat' vs 'Sharpe_Bench' in the **High** volatility row."
        )

        # 5. Walk-Forward Validation (Advanced)
        with st.expander("🚀 Walk-Forward Validation (Advanced)", expanded=False):
            st.markdown("""
            Walk-forward validation splits data into rolling train/test windows to evaluate
            out-of-sample performance. This is more rigorous than a single full-sample backtest.
            """)

            wf_col1, wf_col2 = st.columns(2)
            wf_train = wf_col1.number_input(
                "Training Window (months)", value=24, min_value=6, max_value=60
            )
            wf_test = wf_col2.number_input(
                "Test Window (months)", value=6, min_value=1, max_value=12
            )

            if st.button("Run Walk-Forward Analysis"):
                with st.spinner("Running walk-forward validation..."):
                    wf_results = backtester.walk_forward_backtest(
                        df,
                        "Signal_Trend",
                        train_months=wf_train,
                        test_months=wf_test,
                        cost_bps=bt_cost,
                        rebalance_freq="M",
                    )

                if wf_results:
                    st.success(f"✅ Completed {wf_results['n_periods']} walk-forward periods")

                    wf_summary = wf_results["summary"]
                    wf_c1, wf_c2, wf_c3 = st.columns(3)
                    wf_c1.metric("OOS CAGR", f"{wf_summary.get('CAGR', 0):.2%}")
                    wf_c2.metric("OOS Sharpe", f"{wf_summary.get('Sharpe', 0):.2f}")
                    wf_c3.metric("OOS Max DD", f"{wf_summary.get('MaxDD', 0):.2%}")

                    # Show per-period results
                    st.markdown("#### Per-Period Results")
                    period_data = []
                    for p in wf_results["periods"]:
                        period_data.append(
                            {
                                "Test Period": f"{p['test_start']} to {p['test_end']}",
                                "CAGR": p["metrics"].get("CAGR", 0),
                                "Sharpe": p["metrics"].get("Sharpe", 0),
                                "MaxDD": p["metrics"].get("MaxDD", 0),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(period_data).style.format(
                            {"CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}"}
                        )
                    )
                else:
                    st.warning(
                        "Insufficient data for walk-forward validation with current settings."
                    )

# --- TAB 4: REPORT ---
with tab_rep:
    st.subheader("Research Note Generation")

    st.markdown("### Findings Summary")
    st.write(f"**Asset**: {ticker}")
    st.write(f"**Trend Model**: {sma_window}-Day SMA")

    if not res_df.empty:
        # Create text summary
        high_vol_perf = cond_stats.loc["High", "Sharpe"] if "High" in cond_stats.index else 0
        normal_vol_perf = cond_stats.loc["Normal", "Sharpe"] if "Normal" in cond_stats.index else 0

        st.success(f"Strategy Sharpe in High Vol: **{high_vol_perf:.2f}**")
        st.info(f"Strategy Sharpe in Normal Vol: **{normal_vol_perf:.2f}**")

        st.download_button(
            label="Download Full Research Data (CSV)",
            data=res_df.to_csv().encode("utf-8"),
            file_name=f"{ticker}_research_data.csv",
            mime="text/csv",
        )
