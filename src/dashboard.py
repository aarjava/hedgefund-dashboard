import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import custom modules
try:
    from modules import (
        alerts,
        backtester,
        data_model,
        factors,
        liquidity,
        portfolio,
        regime_analysis,
        reporting,
        risk,
        scenario,
        signals,
        sweep,
    )
    from modules.config import (
        DEFAULT_ADV_PCT,
        DEFAULT_BENCHMARK,
        DEFAULT_BOOTSTRAP_ITER,
        DEFAULT_COST_BPS,
        DEFAULT_MOMENTUM_WINDOW,
        DEFAULT_PORTFOLIO_VALUE,
        DEFAULT_SMA_SWEEP,
        DEFAULT_SMA_WINDOW,
        DEFAULT_VOL_QUANTILE_HIGH,
        FACTOR_PROXIES,
        MACRO_PROXIES,
        MIN_DATA_POINTS,
        PRESET_UNIVERSE,
    )
except ImportError:
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.modules import (
        alerts,
        backtester,
        data_model,
        factors,
        liquidity,
        portfolio,
        regime_analysis,
        reporting,
        risk,
        scenario,
        signals,
        sweep,
    )
    from src.modules.config import (
        DEFAULT_ADV_PCT,
        DEFAULT_BENCHMARK,
        DEFAULT_BOOTSTRAP_ITER,
        DEFAULT_COST_BPS,
        DEFAULT_MOMENTUM_WINDOW,
        DEFAULT_PORTFOLIO_VALUE,
        DEFAULT_SMA_SWEEP,
        DEFAULT_SMA_WINDOW,
        DEFAULT_VOL_QUANTILE_HIGH,
        FACTOR_PROXIES,
        MACRO_PROXIES,
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
    mode = st.radio("Mode", ["Single-Asset", "Portfolio"], horizontal=True)

    st.subheader("1. Asset Selection")
    portfolio_upload = None
    portfolio_csv = None
    portfolio_value = DEFAULT_PORTFOLIO_VALUE
    benchmark_ticker = DEFAULT_BENCHMARK
    weights_input = ""
    adv_pct = float(DEFAULT_ADV_PCT)
    factor_window = 63
    vol_window = 21

    if mode == "Single-Asset":
        t_mode = st.radio("Selection Mode", ["Preset Universe", "Custom Ticker"], horizontal=True)
        if t_mode == "Preset Universe":
            ticker = st.selectbox("Symbol", PRESET_UNIVERSE, index=0)
        else:
            ticker = st.text_input("Enter Symbol (Yahoo Finance)", value="NVDA").upper()
    else:
        portfolio_upload = st.file_uploader("Upload Portfolio CSV", type=["csv"])
        st.caption("CSV columns: ticker, weight (optional: asset_class, strategy)")
        manual_tickers = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL")
        preset_select = st.multiselect("Preset Universe", PRESET_UNIVERSE, default=[])
        weights_input = st.text_input("Weights (comma-separated, optional)")
        portfolio_value = st.number_input(
            "Portfolio Value (USD)", value=float(DEFAULT_PORTFOLIO_VALUE), step=100000.0
        )
        benchmark_ticker = st.text_input("Benchmark Ticker", value=DEFAULT_BENCHMARK).upper()

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
    if mode == "Single-Asset":
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
    else:
        factor_window = st.slider(
            "Factor Beta Window (days)",
            20,
            252,
            63,
            7,
            help="Lookback days for computing rolling beta to factors.",
        )
        vol_window = st.slider(
            "Regime Vol Window (days)",
            10,
            60,
            21,
            5,
            help="Lookback days for volatility calculation (regime detection).",
        )
        adv_pct = st.slider(
            "ADV Participation %",
            0.01,
            0.30,
            float(DEFAULT_ADV_PCT),
            0.01,
            help="Max % of Average Daily Volume to trade. Example: 0.10 = 10% of daily volume.",
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
        help="Threshold for 'High Volatility'. 0.80 means top 20% of volatility readings.",
    )

    if mode == "Single-Asset":
        st.subheader("5. Backtest Settings")
        bt_cost = (
            st.number_input(
                "Transaction Cost (bps)",
                value=DEFAULT_COST_BPS,
                step=1,
                help="Cost per trade. Example: 10 bps = 0.10%",
            )
            / 10000
        )
        allow_short = st.checkbox("Allow Short Selling?", value=False)
    else:
        st.subheader("5. Alert Thresholds")
        dd_alert = st.slider(
            "Max Drawdown Alert",
            -0.6,
            -0.05,
            -0.2,
            0.05,
            help="Trigger alert when drawdown exceeds this level (e.g., -0.20 = 20% loss).",
        )
        vol_alert = st.slider(
            "Volatility Alert (ann.)",
            0.1,
            1.0,
            0.35,
            0.05,
            help="Trigger alert when annualized volatility exceeds this level.",
        )
        beta_alert = st.slider(
            "Beta Alert",
            0.5,
            2.0,
            1.3,
            0.1,
            help="Trigger alert when portfolio beta to benchmark exceeds this level.",
        )
        dttl_alert = st.slider(
            "Days-to-Liquidate Alert",
            1.0,
            20.0,
            5.0,
            1.0,
            help="Trigger alert when estimated days to liquidate entire portfolio exceeds this number.",
        )


# --- Portfolio Mode ---
if mode == "Portfolio":
    compute_start = datetime.now()
    tickers = []
    weights = []

    if portfolio_upload is not None:
        try:
            portfolio_csv = pd.read_csv(portfolio_upload)
        except Exception as e:
            st.error(f"Could not read portfolio CSV: {e}")
            st.stop()

        required_cols = {"ticker", "weight"}
        if not required_cols.issubset(set(portfolio_csv.columns)):
            st.error("Portfolio CSV must include columns: ticker, weight")
            st.stop()

        tickers = portfolio_csv["ticker"].astype(str).str.upper().tolist()
        weights = portfolio_csv["weight"].astype(float).tolist()
    else:
        manual_list = [t.strip().upper() for t in manual_tickers.split(",") if t.strip()]
        tickers = list(dict.fromkeys(manual_list + preset_select))
        if not tickers:
            st.warning("Please provide at least one portfolio ticker.")
            st.stop()

        if weights_input.strip():
            try:
                weights = [float(w) for w in weights_input.split(",")]
            except ValueError:
                st.warning("Weights must be numeric. Falling back to equal weights.")
                weights = [1.0] * len(tickers)

            if len(weights) != len(tickers):
                st.warning("Weights length does not match tickers. Falling back to equal weights.")
                weights = [1.0] * len(tickers)
        else:
            weights = [1.0] * len(tickers)

    weights = portfolio.normalize_weights(tickers, weights)

    fetch_tickers = list(dict.fromkeys(tickers + [benchmark_ticker]))
    with st.spinner("Fetching portfolio data..."):
        data_map = data_model.fetch_multi_asset_data(tuple(fetch_tickers), period=period_arg)

    portfolio_data = {t: data_map.get(t) for t in tickers if t in data_map}
    price_df = data_model.align_close_prices(portfolio_data)
    volume_df = data_model.align_volume(portfolio_data)

    if price_df.empty:
        st.error("Could not load portfolio data. Check tickers.")
        st.stop()

    if date_mode == "Custom":
        mask = (price_df.index.date >= start_date) & (price_df.index.date <= end_date)
        price_df = price_df.loc[mask].copy()
        volume_df = volume_df.loc[mask].copy()

    if len(price_df) < MIN_DATA_POINTS:
        st.warning(
            f"Not enough data points for selected range/period (need at least {MIN_DATA_POINTS})."
        )
        st.stop()

    # Align weights to available tickers
    weights = weights.reindex(price_df.columns).fillna(0.0)
    if weights.sum() <= 0:
        weights = portfolio.normalize_weights(price_df.columns, [1.0] * len(price_df.columns))
    else:
        weights = weights / weights.sum()

    port_df = portfolio.build_portfolio(price_df, weights)
    port_returns = port_df["Portfolio_Return"]
    port_equity = port_df["Portfolio_Equity"]

    # Benchmark
    benchmark_df = data_map.get(benchmark_ticker, pd.DataFrame())
    if benchmark_df is not None and not benchmark_df.empty:
        if date_mode == "Custom":
            bmask = (benchmark_df.index.date >= start_date) & (benchmark_df.index.date <= end_date)
            benchmark_df = benchmark_df.loc[bmask].copy()
    benchmark_returns = (
        benchmark_df["Close"].pct_change().dropna()
        if not benchmark_df.empty
        else pd.Series(dtype=float)
    )

    # Risk metrics
    ann_vol = port_returns.std() * np.sqrt(252)
    max_dd = risk.compute_max_drawdown(port_equity)
    beta = risk.compute_beta(port_returns, benchmark_returns)
    var_cvar = risk.compute_var_cvar(port_returns, level=0.95)

    # Liquidity metrics
    liquidity_df = liquidity.compute_liquidity_metrics(
        price_df, volume_df, weights, portfolio_value, adv_pct=adv_pct
    )
    max_dttl = liquidity_df["DaysToLiquidate"].max() if not liquidity_df.empty else np.nan

    # Factor attribution
    factor_data = data_model.fetch_multi_asset_data(
        tuple(FACTOR_PROXIES.values()), period=period_arg
    )
    factor_prices = data_model.align_close_prices(factor_data)
    if date_mode == "Custom" and not factor_prices.empty:
        fmask = (factor_prices.index.date >= start_date) & (factor_prices.index.date <= end_date)
        factor_prices = factor_prices.loc[fmask].copy()
    factor_returns = factors.compute_factor_returns(factor_prices)
    factor_betas = factors.compute_factor_betas(port_returns, factor_returns, window=factor_window)
    factor_contrib = factors.compute_factor_contributions(factor_betas, factor_returns)

    # Macro betas for scenario shocks
    macro_data = data_model.fetch_multi_asset_data(tuple(MACRO_PROXIES.values()), period=period_arg)
    macro_prices = data_model.align_close_prices(macro_data)
    if date_mode == "Custom" and not macro_prices.empty:
        mmask = (macro_prices.index.date >= start_date) & (macro_prices.index.date <= end_date)
        macro_prices = macro_prices.loc[mmask].copy()
    macro_returns = factors.compute_factor_returns(macro_prices)
    macro_betas = factors.compute_factor_betas(port_returns, macro_returns, window=factor_window)

    # Alpha series
    alpha_series = factors.compute_alpha_series(
        port_returns, benchmark_returns, window=factor_window
    )

    # Regime classification on benchmark
    regime_label = "N/A"
    benchmark_regimes = None
    if not benchmark_df.empty:
        bench_ind = signals.add_technical_indicators(
            benchmark_df, sma_window=200, mom_window=DEFAULT_MOMENTUM_WINDOW, vol_window=vol_window
        )
        bench_ind = signals.detect_volatility_regime(
            bench_ind,
            vol_col=f"Vol_{vol_window}d",
            quantile_high=vol_q_high,
            quantile_low=0.25,
            use_expanding=use_oos,
        )
        latest_bench = bench_ind.iloc[-1]
        trend_regime = "Bull" if latest_bench["Close"] > latest_bench["SMA_200"] else "Bear"
        regime_label = f"{latest_bench['Vol_Regime']} / {trend_regime}"
        benchmark_regimes = bench_ind["Vol_Regime"]

    # Regime sensitivity cross-section
    regime_sensitivity_df = pd.DataFrame()
    portfolio_sensitivity = {"Sharpe_Diff": np.nan, "CAGR_Diff": np.nan}
    benchmark_sensitivity = {"Sharpe_Diff": np.nan, "CAGR_Diff": np.nan}
    if benchmark_regimes is not None:
        rows = []
        for asset in price_df.columns:
            asset_returns = price_df[asset].pct_change()
            temp = pd.concat([asset_returns, benchmark_regimes], axis=1).dropna()
            if temp.empty:
                continue
            temp.columns = ["Return", "Vol_Regime"]
            stats = backtester.calculate_regime_stats(temp, "Return", "Vol_Regime")
            sens = regime_analysis.compute_regime_sensitivity(stats)
            rows.append(
                {
                    "Asset": asset,
                    "Sharpe_Diff": sens.get("Sharpe_Diff", np.nan),
                    "CAGR_Diff": sens.get("CAGR_Diff", np.nan),
                }
            )
        if rows:
            regime_sensitivity_df = (
                pd.DataFrame(rows).set_index("Asset").sort_values("Sharpe_Diff", ascending=False)
            )

        port_temp = pd.concat([port_returns, benchmark_regimes], axis=1).dropna()
        if not port_temp.empty:
            port_temp.columns = ["Return", "Vol_Regime"]
            port_stats = backtester.calculate_regime_stats(port_temp, "Return", "Vol_Regime")
            portfolio_sensitivity = regime_analysis.compute_regime_sensitivity(port_stats)

        if not benchmark_returns.empty:
            bench_temp = pd.concat([benchmark_returns, benchmark_regimes], axis=1).dropna()
            if not bench_temp.empty:
                bench_temp.columns = ["Return", "Vol_Regime"]
                bench_stats = backtester.calculate_regime_stats(bench_temp, "Return", "Vol_Regime")
                benchmark_sensitivity = regime_analysis.compute_regime_sensitivity(bench_stats)

    # Alerts
    thresholds = {
        "MaxDrawdown": {"type": "<", "value": dd_alert, "severity": "High"},
        "Volatility": {"type": ">", "value": vol_alert, "severity": "Medium"},
        "Beta": {"type": ">", "value": beta_alert, "severity": "Medium"},
        "DaysToLiquidate": {"type": ">", "value": dttl_alert, "severity": "High"},
    }
    metrics = {
        "MaxDrawdown": max_dd,
        "Volatility": ann_vol,
        "Beta": beta,
        "DaysToLiquidate": max_dttl,
    }
    alert_df = alerts.evaluate_alerts(metrics, thresholds)
    anomalies = alerts.detect_zscore_anomalies(port_returns)

    compute_end = datetime.now()
    compute_latency = (compute_end - compute_start).total_seconds()
    last_data_ts = price_df.index[-1]

    # --- Portfolio Header ---
    st.markdown("## 📊 Portfolio Dashboard")
    st.caption(f"Data as of {last_data_ts.date()} • Compute latency: {compute_latency:.2f}s")

    m1, m2, m3, m4 = st.columns(4)
    total_return = port_equity.iloc[-1] - 1
    m1.metric("Total Return", f"{total_return:.2%}")
    m2.metric("Ann Vol", f"{ann_vol:.2%}")
    m3.metric("Max Drawdown", f"{max_dd:.2%}")
    m4.metric("Regime", regime_label)

    # --- Portfolio Tabs ---
    tab_ov, tab_risk, tab_attr, tab_scen, tab_sig, tab_alert, tab_rep = st.tabs(
        [
            "📈 Overview",
            "🛡️ Risk & Liquidity",
            "🧬 Attribution",
            "🧪 Scenario",
            "📡 Signals Health",
            "🚨 Alerts",
            "📄 Report",
        ]
    )

    # Overview
    with tab_ov:
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.subheader("Portfolio Weights")
            st.dataframe(weights.to_frame("Weight").style.format("{:.2%}"))
        with col_q2:
            st.subheader("Quick Query")
            query = st.text_input("Ask", placeholder="top movers, factor drift, largest drawdown")
            if query:
                q = query.lower()
                if "top" in q and "mover" in q:
                    last_returns = price_df.pct_change().iloc[-1].sort_values(ascending=False)
                    st.write(
                        last_returns.head(5).to_frame("Last Day Return").style.format("{:.2%}")
                    )
                elif "factor" in q and "drift" in q and not factor_betas.empty:
                    drift = factor_betas.iloc[-1] - factor_betas.iloc[-min(21, len(factor_betas))]
                    st.write(drift.to_frame("Beta Drift (20d)").style.format("{:.2f}"))
                elif "drawdown" in q:
                    dd_series = risk.compute_drawdown_series(port_equity)
                    worst = dd_series.idxmin()
                    st.write(f"Worst drawdown {dd_series.min():.2%} on {worst.date()}")
                else:
                    st.info("Try: top movers, factor drift, largest drawdown")

        st.subheader("Equity Curve vs Benchmark")
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=port_equity.index, y=port_equity, name="Portfolio", line=dict(color="#00ff00")
            )
        )
        if not benchmark_returns.empty:
            bench_equity = (1 + benchmark_returns).cumprod()
            fig_eq.add_trace(
                go.Scatter(
                    x=bench_equity.index,
                    y=bench_equity,
                    name=benchmark_ticker,
                    line=dict(color="#888"),
                )
            )
        fig_eq.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Regime Lab (Cross-Section)")
        if regime_sensitivity_df.empty:
            st.info("Not enough data to compute regime sensitivity cross-section.")
        else:
            top5 = regime_sensitivity_df.head(5)
            bot5 = regime_sensitivity_df.tail(5).sort_values("Sharpe_Diff")
            c_top, c_bot = st.columns(2)
            with c_top:
                st.caption("Most Regime-Resilient (Sharpe High - Normal)")
                st.dataframe(top5.style.format("{:.2f}"))
            with c_bot:
                st.caption("Most Regime-Fragile (Sharpe High - Normal)")
                st.dataframe(bot5.style.format("{:.2f}"))

            st.caption(
                f"Portfolio sensitivity (Sharpe diff): {portfolio_sensitivity.get('Sharpe_Diff', np.nan):.2f} | "
                f"Benchmark sensitivity (Sharpe diff): {benchmark_sensitivity.get('Sharpe_Diff', np.nan):.2f}"
            )

    # Risk & Liquidity
    with tab_risk:
        st.subheader("Risk Posture")
        score = risk.risk_posture_score(ann_vol, max_dd, beta, max_dttl)
        st.metric("Risk Posture Score", f"{score:.0f}/100")
        st.caption(
            "Higher is better. Penalizes high vol, deep drawdowns, high beta, and illiquidity."
        )

        st.subheader("Drawdown")
        dd_series = risk.compute_drawdown_series(port_equity)
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=dd_series.index,
                y=dd_series * 100,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="#ff4b4b"),
            )
        )
        fig_dd.update_layout(template="plotly_dark", height=300, yaxis_title="Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)

        st.subheader("Liquidity")
        if liquidity_df.empty:
            st.info("Liquidity data not available.")
        else:
            st.dataframe(
                liquidity_df.style.format(
                    {
                        "Weight": "{:.2%}",
                        "PositionValue": "${:,.0f}",
                        "ADV$": "${:,.0f}",
                        "DaysToLiquidate": "{:.1f}",
                    }
                )
            )

        st.subheader("Correlation Matrix")
        corr = risk.compute_correlation_matrix(price_df.pct_change().dropna())
        if not corr.empty:
            st.dataframe(corr.style.background_gradient(cmap="RdYlGn").format("{:.2f}"))

    # Attribution
    with tab_attr:
        st.subheader("Factor Betas (Rolling)")
        if factor_betas.empty:
            st.info("Not enough data to compute factor betas.")
        else:
            latest_betas = factor_betas.iloc[-1].sort_values(ascending=False)
            fig_beta = px.bar(latest_betas, title="Latest Factor Betas", template="plotly_dark")
            st.plotly_chart(fig_beta, use_container_width=True)

        st.subheader("Factor Contributions")
        if not factor_contrib.empty:
            last_contrib = factor_contrib.iloc[-1].sort_values(ascending=False)
            st.dataframe(last_contrib.to_frame("Contribution").style.format("{:.2%}"))
        else:
            st.info("Factor contribution data not available.")

        st.subheader("Rolling Alpha")
        if not alpha_series.empty:
            fig_alpha = go.Figure()
            fig_alpha.add_trace(
                go.Scatter(
                    x=alpha_series.index, y=alpha_series, name="Alpha", line=dict(color="#00ff00")
                )
            )
            fig_alpha.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_alpha, use_container_width=True)

    # Scenario
    with tab_scen:
        st.subheader("What-If Scenario Shocks")
        if macro_betas.empty:
            st.info("Not enough data to compute macro betas.")
        else:
            latest_macro = macro_betas.iloc[-1]
            shocks = {}
            cols = st.columns(3)
            macro_items = list(MACRO_PROXIES.keys())
            for i, name in enumerate(macro_items):
                col = cols[i % 3]
                shocks[name] = col.slider(f"{name} Shock (%)", -10.0, 10.0, 0.0, 0.5) / 100

            # Map shocks to proxy names in betas
            betas_series = latest_macro.copy()
            betas_series.index = list(MACRO_PROXIES.keys())
            impact = scenario.run_scenario_shocks(betas_series, shocks)
            if not impact.empty:
                st.dataframe(impact.to_frame("Impact").style.format("{:.2%}"))
                st.metric("Total Scenario Impact", f"{impact['Total']:.2%}")

    # Signals Health
    with tab_sig:
        st.subheader("Signal Decay (Benchmark)")
        if benchmark_df.empty:
            st.info("Benchmark data not available.")
        else:
            bench_tmp = benchmark_df.copy()
            bench_tmp["Daily_Return"] = bench_tmp["Close"].pct_change()
            bench_tmp["Signal"] = np.sign(
                bench_tmp["Close"] - bench_tmp["Close"].rolling(50).mean()
            )
            for h in [21, 63, 126]:
                bench_tmp[f"Fwd_{h}"] = bench_tmp["Close"].pct_change(h).shift(-h)
            decay = {
                "1M": bench_tmp.groupby("Signal")["Fwd_21"].mean(),
                "3M": bench_tmp.groupby("Signal")["Fwd_63"].mean(),
                "6M": bench_tmp.groupby("Signal")["Fwd_126"].mean(),
            }
            decay_df = pd.DataFrame(decay)
            st.dataframe(decay_df.style.format("{:.2%}"))

            st.subheader("Rolling IC (Signal vs 1M Forward Return)")
            ic = bench_tmp["Signal"].rolling(63).corr(bench_tmp["Fwd_21"])
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Scatter(x=ic.index, y=ic, name="IC", line=dict(color="#ff9f43")))
            fig_ic.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_ic, use_container_width=True)

    # Alerts
    with tab_alert:
        st.subheader("Active Alerts")
        if alert_df.empty:
            st.success("No active alerts.")
        else:
            st.dataframe(alert_df)

        st.subheader("Return Anomalies")
        if anomalies.empty:
            st.info("No anomalies detected.")
        else:
            st.dataframe(anomalies.tail(10).style.format({"Return": "{:.2%}", "ZScore": "{:.2f}"}))

    # Report
    with tab_rep:
        st.subheader("Client-Ready Report")
        summary = {
            "Portfolio": f"{len(price_df.columns)} assets",
            "Benchmark": benchmark_ticker,
            "Total Return": f"{total_return:.2%}",
            "Ann Vol": f"{ann_vol:.2%}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Beta": f"{beta:.2f}" if not np.isnan(beta) else "N/A",
            "Regime Sensitivity (Sharpe Diff)": f"{portfolio_sensitivity.get('Sharpe_Diff', np.nan):.2f}",
            "Benchmark Sensitivity (Sharpe Diff)": f"{benchmark_sensitivity.get('Sharpe_Diff', np.nan):.2f}",
            "VaR (95%)": f"{var_cvar['VaR']:.2%}" if not np.isnan(var_cvar["VaR"]) else "N/A",
            "CVaR (95%)": f"{var_cvar['CVaR']:.2%}" if not np.isnan(var_cvar["CVaR"]) else "N/A",
        }
        tables = {
            "Weights": weights.to_frame("Weight"),
            "Liquidity": liquidity_df,
            "Latest Factor Betas": (
                factor_betas.tail(1) if not factor_betas.empty else pd.DataFrame()
            ),
            "Regime Sensitivity": (
                regime_sensitivity_df if not regime_sensitivity_df.empty else pd.DataFrame()
            ),
        }
        payload = reporting.build_report_payload(summary, tables)
        st.markdown(payload["markdown"])

        st.download_button(
            label="Download Report (Markdown)",
            data=payload["markdown"].encode("utf-8"),
            file_name="portfolio_report.md",
            mime="text/markdown",
        )
        st.download_button(
            label="Download Report (HTML)",
            data=payload["html"].encode("utf-8"),
            file_name="portfolio_report.html",
            mime="text/html",
        )
        st.download_button(
            label="Download Portfolio Returns (CSV)",
            data=port_df.to_csv().encode("utf-8"),
            file_name="portfolio_returns.csv",
            mime="text/csv",
        )

    st.stop()


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

# --- Backtest (cached for reuse) ---
df["Signal_Trend"] = np.where(df["Close"] > df[f"SMA_{sma_window}"], 1, -1 if allow_short else 0)
bt_cache_key = get_cache_key(signal_cache_key, bt_cost, allow_short, use_oos, vol_q_high)

if bt_cache_key not in st.session_state.backtest_results:
    with st.spinner("Running backtest simulation..."):
        res_df = backtester.run_backtest(df, "Signal_Trend", cost_bps=bt_cost, rebalance_freq="M")
        st.session_state.backtest_results[bt_cache_key] = res_df

res_df = st.session_state.backtest_results[bt_cache_key]
if not res_df.empty:
    res_df = res_df.copy()
    res_df["Vol_Regime"] = df["Vol_Regime"]

    cond_stats = backtester.calculate_conditional_stats(res_df, "Strategy_Net_Return", "Vol_Regime")
    bench_cond = backtester.calculate_conditional_stats(res_df, "Daily_Return", "Vol_Regime")
    stats_by_regime = backtester.calculate_regime_stats(res_df, "Strategy_Net_Return", "Vol_Regime")
    regime_sensitivity = regime_analysis.compute_regime_sensitivity(stats_by_regime)
    bootstrap_sharpe = regime_analysis.bootstrap_regime_diff(
        res_df["Strategy_Net_Return"],
        res_df["Vol_Regime"],
        metric="Sharpe",
        n_boot=DEFAULT_BOOTSTRAP_ITER,
    )

    transition_matrix = regime_analysis.compute_transition_matrix(df["Vol_Regime"])
    transition_stats = regime_analysis.compute_transition_stats(
        res_df["Strategy_Net_Return"], res_df["Vol_Regime"]
    )
    sweep_df = sweep.run_sma_regime_sweep(df, DEFAULT_SMA_SWEEP, mom_window, vol_q_high, use_oos)
else:
    cond_stats = pd.DataFrame()
    bench_cond = pd.DataFrame()
    stats_by_regime = pd.DataFrame()
    regime_sensitivity = {"Sharpe_Diff": np.nan, "CAGR_Diff": np.nan}
    bootstrap_sharpe = {"diff": np.nan, "p_value": np.nan}
    transition_matrix = pd.DataFrame()
    transition_stats = pd.DataFrame()
    sweep_df = pd.DataFrame()

# --- Tabs ---
tab_ov, tab_regime, tab_lab, tab_bt, tab_rep = st.tabs(
    ["📈 Overview", "🌪️ Regime Analysis", "🧪 Regime Lab", "🧪 Backtest Engine", "📄 Report"]
)

# --- TAB 1: OVERVIEW ---
with tab_ov:
    # Interactive Price Chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Close Price", line=dict(color="white", width=1))
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"SMA_{sma_window}"],
            name=f"{sma_window}-Day SMA",
            line=dict(color="#ff9f43", width=1),
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
            marker=dict(color="red", size=2),
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

# --- TAB 3: REGIME LAB ---
with tab_lab:
    st.subheader("Regime Transition Matrix")
    if transition_matrix.empty:
        st.info("Not enough data to compute transition matrix.")
    else:
        fig_tm = go.Figure(
            data=go.Heatmap(
                z=transition_matrix.values,
                x=transition_matrix.columns,
                y=transition_matrix.index,
                colorscale="Blues",
                zmin=0,
                zmax=1,
            )
        )
        fig_tm.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Current Regime",
            yaxis_title="Previous Regime",
        )
        st.plotly_chart(fig_tm, use_container_width=True)

    st.subheader("Transition Impact")
    if transition_stats.empty:
        st.info("Not enough data to compute transition performance.")
    else:
        st.dataframe(
            transition_stats.style.format(
                {
                    "Mean": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "WinRate": "{:.1%}",
                    "CAGR": "{:.2%}",
                    "Count": "{:.0f}",
                }
            )
        )

    st.subheader("Regime Sensitivity")
    c_s1, c_s2, c_s3 = st.columns(3)
    c_s1.metric("Sharpe High - Normal", f"{regime_sensitivity.get('Sharpe_Diff', np.nan):.2f}")
    c_s2.metric("CAGR High - Normal", f"{regime_sensitivity.get('CAGR_Diff', np.nan):.2%}")
    pval = bootstrap_sharpe.get("p_value", np.nan)
    if np.isnan(pval):
        c_s3.metric("Bootstrap p-value", "N/A")
    else:
        c_s3.metric("Bootstrap p-value", f"{pval:.3f}")

    st.subheader("Parameter Robustness Sweep (Sharpe)")
    if sweep_df.empty:
        st.info("Not enough data to compute SMA sweep.")
    else:
        sweep_sharpe = sweep_df.reset_index().pivot(index="SMA", columns="Regime", values="Sharpe")
        fig_sweep = px.imshow(
            sweep_sharpe,
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="Sharpe by SMA Window and Regime",
        )
        fig_sweep.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_sweep, use_container_width=True)

# --- TAB 3: BACKTEST ---
with tab_bt:
    st.subheader("Strategy Simulation")

    # Out-of-sample mode indicator
    if use_oos:
        st.success(
            "🔬 **Out-of-Sample Mode Active** - Regime classification uses only past data at each point"
        )

    if not res_df.empty:

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
                line=dict(color="#00ff00"),
            )
        )
        fig_eq.add_trace(
            go.Scatter(
                x=res_df.index,
                y=res_df["Equity_Benchmark"],
                name="Buy & Hold",
                line=dict(color="gray", dash="dot"),
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
                    line=dict(color="#ff4b4b"),
                )
            )
            fig_dd.add_trace(
                go.Scatter(
                    x=res_df.index,
                    y=res_df["DD_Benchmark"] * 100,
                    name="Benchmark Drawdown",
                    line=dict(color="gray", dash="dot"),
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

        transition_risk = "N/A"
        if not transition_stats.empty and "Sharpe" in transition_stats.columns:
            transition_risk = transition_stats.sort_values("Sharpe").index[0]

        sweep_stability = "N/A"
        if not sweep_df.empty and "Sharpe" in sweep_df.columns:
            sweep_std = sweep_df.groupby("Regime")["Sharpe"].std().dropna()
            if not sweep_std.empty:
                sweep_stability = ", ".join([f"{k}: {v:.2f}" for k, v in sweep_std.items()])

        st.success(f"Strategy Sharpe in High Vol: **{high_vol_perf:.2f}**")
        st.info(f"Strategy Sharpe in Normal Vol: **{normal_vol_perf:.2f}**")
        st.write(
            f"**Regime Sensitivity (Sharpe High - Normal)**: {regime_sensitivity.get('Sharpe_Diff', np.nan):.2f}"
        )
        st.write(f"**Top Transition Risk**: {transition_risk}")
        st.write(f"**Sweep Stability (Sharpe Std)**: {sweep_stability}")

        st.download_button(
            label="Download Full Research Data (CSV)",
            data=res_df.to_csv().encode("utf-8"),
            file_name=f"{ticker}_research_data.csv",
            mime="text/csv",
        )
