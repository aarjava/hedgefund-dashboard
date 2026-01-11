import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import custom modules
try:
    from modules import data_model, signals, backtester
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.modules import data_model, signals, backtester

# --- Configuration ---
st.set_page_config(
    page_title="Quantitative Research Dashboard",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
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
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("🎛️ Research Config")
    
    st.subheader("1. Asset Selection")
    t_mode = st.radio("Selection Mode", ["Preset Universe", "Custom Ticker"], horizontal=True)
    
    if t_mode == "Preset Universe":
        universe = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLK", "XLE", "BTC-USD", "ETH-USD"]
        ticker = st.selectbox("Symbol", universe, index=0)
    else:
        ticker = st.text_input("Enter Symbol (Yahoo Finance)", value="NVDA", help="Enter a valid Yahoo Finance ticker symbol (e.g., AAPL, MSFT, BTC-USD).").upper()
    
    st.subheader("2. Time Horizon")
    date_mode = st.selectbox("Date Range", ["Last 5 Years", "Last 10 Years", "Max", "Custom"])
    
    if date_mode == "Custom":
        d_col1, d_col2 = st.columns(2)
        start_date = d_col1.date_input("Start", value=datetime.today() - timedelta(days=365*2))
        end_date = d_col2.date_input("End", value=datetime.today())
        period_arg = "max"
    else:
        period_map = {"Last 5 Years": "5y", "Last 10 Years": "10y", "Max": "max"}
        period_arg = period_map[date_mode]

    st.subheader("3. Signal Parameters")
    sma_window = st.slider("Trend SMA Window", 10, 200, 50, 10, help="Lookback days for Simple Moving Average trend signal.")
    mom_window = st.slider("Momentum Lookback (Months)", 1, 24, 12, 1, help="Lookback months for Momentum signal.")
    
    st.markdown("---")
    st.subheader("4. Research Rigor")
    st.info("Regime Classification: Active")
    vol_q_high = st.slider("High Volatility Quantile", 0.5, 0.95, 0.75, 0.05, help="Threshold to define 'High Volatility' regime (e.g., 0.75 means top 25% of volatility readings).")
    
    st.subheader("5. Backtest Settings")
    bt_cost = st.number_input("Transaction Cost (bps)", value=10, step=1, help="Transaction cost per trade in basis points (1 bp = 0.01%).") / 10000
    allow_short = st.checkbox("Allow Short Selling?", value=False, help="If enabled, the strategy will sell short when the trend is negative.")


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

if len(df) < 50:
    st.warning("Not enough data points for selected range/period.")
    st.stop()

# --- Signal Calculation ---
df = signals.add_technical_indicators(df, sma_window=sma_window, mom_window=mom_window)

# --- Regime Detection ---
# Using 21-day annualized vol
df = signals.detect_volatility_regime(df, vol_col='Vol_21d', quantile_high=vol_q_high, quantile_low=0.25)

# --- Dashboard Header ---
st.markdown("## 🔍 Research Question")
st.markdown("> **How sensitive is trend-following performance to volatility regimes in US equities?**")

latest = df.iloc[-1]
prev = df.iloc[-2]
chg_pct = latest['Daily_Return']

h1, h2, h3, h4 = st.columns(4)
h1.metric("Asset", f"{ticker} (${latest['Close']:.2f})", f"{chg_pct:.2%}")
h2.metric("Current Regime", latest['Vol_Regime'])
h3.metric(f"Volatility ({vol_q_high:.0%}-tile)", f"{latest['Vol_21d']:.2%}")
h4.metric("Trend Status", "BULLISH" if latest['Close'] > latest[f'SMA_{sma_window}'] else "BEARISH")

# --- Tabs ---
tab_ov, tab_regime, tab_bt, tab_rep = st.tabs(["📈 Overview", "🌪️ Regime Analysis", "🧪 Backtest Engine", "📄 Report"])

# --- TAB 1: OVERVIEW ---
with tab_ov:
    # Interactive Price Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='white', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{sma_window}'], name=f'{sma_window}-Day SMA', line=dict(color='#ff9f43', width=1)))
    
    # Highlight High Volatility Regimes
    # Filter high vol periods
    high_vol_mask = df['Vol_Regime'] == 'High'
    # We can plot markers or shade areas. Shading is valid but tricky in Plotly without shapes list.
    # Let's plot points
    high_vol_pts = df[high_vol_mask]
    fig.add_trace(go.Scatter(x=high_vol_pts.index, y=high_vol_pts['Close'], mode='markers', name='High Volatility', marker=dict(color='red', size=2)))
    
    fig.update_layout(
        title=f"{ticker} Price History & Regime Context",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red dots indicate days classified as 'High Volatility' regime.")

# --- TAB 2: REGIME ANALYSIS ---
with tab_regime:
    st.subheader("Volatility Regime Classification")
    
    c1, c2 = st.columns(2)
    with c1:
        # Scatter: Vol vs Returns needed? Maybe just distribution
        fig_hist = px.histogram(df, x="Vol_21d", color="Vol_Regime", nbins=50, title="Volatility Distribution", template="plotly_dark",
                                color_discrete_map={"High": "#ff4b4b", "Low": "#00ff00", "Normal": "#888888"})
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        # Pie chart of time spent in regimes
        regime_counts = df['Vol_Regime'].value_counts()
        fig_pie = px.pie(values=regime_counts, names=regime_counts.index, title="Time Spent in Regimes", template="plotly_dark",
                         color=regime_counts.index, color_discrete_map={"High": "#ff4b4b", "Low": "#00ff00", "Normal": "#888888"})
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("### Regime Characteristics")
    stats = df.groupby('Vol_Regime')[['Daily_Return', 'Vol_21d']].mean()
    # Annualize return
    stats['Ann_Return'] = stats['Daily_Return'] * 252
    st.dataframe(stats.style.format("{:.2%}"))

# --- TAB 3: BACKTEST ---
with tab_bt:
    st.subheader("Strategy Simulation")
    
    # Define Strategy
    # Trend Following
    df['Signal_Trend'] = np.where(df['Close'] > df[f'SMA_{sma_window}'], 1, -1 if allow_short else 0)
    
    # Run Backtest
    res_df = backtester.run_backtest(df, 'Signal_Trend', cost_bps=bt_cost, rebalance_freq='M')
    
    if not res_df.empty:
        # Add Regime to Backtest Results (forward fill valid for analysis)
        res_df['Vol_Regime'] = df['Vol_Regime']
        
        # 1. Global Metrics
        strat_metrics = backtester.calculate_perf_metrics(res_df['Equity_Strategy'])
        bench_metrics = backtester.calculate_perf_metrics(res_df['Equity_Benchmark'])
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Global CAGR", f"{strat_metrics['CAGR']:.2%}")
        col_m2.metric("Global Sharpe", f"{strat_metrics['Sharpe']:.2f}")
        col_m3.metric("Max Drawdown", f"{strat_metrics['MaxDD']:.2%}")
        
        # 2. Equity Curve
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=res_df.index, y=res_df['Equity_Strategy'], name='Trend Strategy', line=dict(color='#00ff00')))
        fig_eq.add_trace(go.Scatter(x=res_df.index, y=res_df['Equity_Benchmark'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
        fig_eq.update_layout(title="Equity Curve", template="plotly_dark", height=400)
        st.plotly_chart(fig_eq, use_container_width=True)
        
        # 3. Conditional Analysis
        st.markdown("### 🔬 Conditional Performance by Regime")
        st.info("Does the strategy outperform during High Volatility?")
        
        cond_stats = backtester.calculate_conditional_stats(res_df, 'Strategy_Net_Return', 'Vol_Regime')
        
        # Add Benchmark Conditional Stats for comparison
        bench_cond = backtester.calculate_conditional_stats(res_df, 'Daily_Return', 'Vol_Regime')
        
        # Merge
        comparison = pd.concat([cond_stats.add_suffix('_Strat'), bench_cond.add_suffix('_Bench')], axis=1)
        
        # Reorder columns
        comparison = comparison[['Ann_Return_Strat', 'Ann_Return_Bench', 'Sharpe_Strat', 'Sharpe_Bench', 'WinRate_Strat']]
        
        st.dataframe(comparison.style.background_gradient(cmap='RdYlGn', subset=['Ann_Return_Strat', 'Sharpe_Strat']).format("{:.2f}"))
        
        st.markdown("**Key Insight:** Compare 'Sharpe_Strat' vs 'Sharpe_Bench' in the **High** volatility row.")

# --- TAB 4: REPORT ---
with tab_rep:
    st.subheader("Research Note Generation")
    
    st.markdown("### Findings Summary")
    st.write(f"**Asset**: {ticker}")
    st.write(f"**Trend Model**: {sma_window}-Day SMA")
    
    if not res_df.empty:
        # Create text summary
        high_vol_perf = cond_stats.loc['High', 'Sharpe'] if 'High' in cond_stats.index else 0
        normal_vol_perf = cond_stats.loc['Normal', 'Sharpe'] if 'Normal' in cond_stats.index else 0
        
        st.success(f"Strategy Sharpe in High Vol: **{high_vol_perf:.2f}**")
        st.info(f"Strategy Sharpe in Normal Vol: **{normal_vol_perf:.2f}**")
        
        st.download_button(
            label="Download Full Research Data (CSV)",
            data=res_df.to_csv().encode('utf-8'),
            file_name=f"{ticker}_research_data.csv",
            mime="text/csv"
        )
