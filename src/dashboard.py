import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import custom modules
# Check if running from root or src, adjust path if needed or assume simple import if in python path
try:
    from modules import data_model, signals, backtester
except ImportError:
    # Fallback if running from src directory directly
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.modules import data_model, signals, backtester

# --- Configuration ---
st.set_page_config(
    page_title="HedgeFund Analyst Dashboard",
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
</style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("🎛️ Strategy Config")
    
    st.subheader("1. Asset Selection")
    t_mode = st.radio("Selection Mode", ["Preset Universe", "Custom Ticker"], horizontal=True)
    
    if t_mode == "Preset Universe":
        universe = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLK", "XLE", "BTC-USD", "ETH-USD"]
        ticker = st.selectbox("Symbol", universe, index=0)
    else:
        ticker = st.text_input("Enter Symol (Yahoo Finance)", value="NVDA").upper()
    
    st.subheader("2. Time Horizon")
    date_mode = st.selectbox("Date Range", ["Last 5 Years", "Last 10 Years", "Max", "Custom"])
    
    if date_mode == "Custom":
        d_col1, d_col2 = st.columns(2)
        start_date = d_col1.date_input("Start", value=datetime.today() - timedelta(days=365*2))
        end_date = d_col2.date_input("End", value=datetime.today())
        period_arg = "max" # We fetch max then slice for custom
    else:
        period_map = {"Last 5 Years": "5y", "Last 10 Years": "10y", "Max": "max"}
        period_arg = period_map[date_mode]

    st.subheader("3. Signal Parameters")
    sma_window = st.slider("Trend SMA Window", 10, 200, 50, 10, help="Lookback days for Simple Moving Average trend signal.")
    mom_window = st.slider("Momentum Lookback (Months)", 1, 24, 12, 1, help="Lookback months for Momentum signal.")
    
    st.subheader("4. Backtest Settings")
    bt_cost = st.number_input("Transaction Cost (bps)", value=10, step=1, help="Basis points per trade (e.g., 10 bps = 0.10%)") / 10000
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

if len(df) < 50:
    st.warning("Not enough data points for selected range/period.")
    st.stop()

# --- Signal Calculation ---
df = signals.add_technical_indicators(df, sma_window=sma_window, mom_window=mom_window)

# --- Dashboard Header ---
st.title(f"{ticker} Quantitative Analysis")
latest = df.iloc[-1]
prev = df.iloc[-2]
chg = (latest['Close'] - prev['Close'])
chg_pct = latest['Daily_Return']

h1, h2, h3, h4 = st.columns(4)
h1.metric("Price", f"${latest['Close']:.2f}", f"{chg_pct:.2%}")
h2.metric("Trend (SMA)", f"${latest[f'SMA_{sma_window}']:.2f}", delta_color="off")
h3.metric("Momentum", f"{latest[f'Momentum_{mom_window}M_1M']:.2%}")
h4.metric("Volatility (21d)", f"{latest['Vol_21d']:.2%}")

# --- Tabs ---
tab_ov, tab_sig, tab_bt, tab_rep = st.tabs(["📈 Overview", "🚦 Signal Logic", "🧪 Backtest Engine", "📄 Report"])

# --- TAB 1: OVERVIEW ---
with tab_ov:
    # Interactive Price Chart
    fig = go.Figure()
    
    # Candlestick or Line? Let's do Line for cleanliness over long periods, user can zoom.
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='white', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{sma_window}'], name=f'{sma_window}-Day SMA', line=dict(color='#ff9f43', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='200-Day SMA', line=dict(color='#2e86de', width=1)))
    
    fig.update_layout(
        title=f"{ticker} Price Action & Key Levels",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume Chart (sub-chart)
    # Keeping it simple for now, just description
    st.info("The chart above overlays the selected Trend SMA (Orange) and the Long-Team 200 SMA (Blue).")

# --- TAB 2: SIGNALS ---
with tab_sig:
    col1, col2, col3 = st.columns(3)
    
    # 1. Trend Signal
    trend_state = "BULLISH" if latest['Close'] > latest[f'SMA_{sma_window}'] else "BEARISH"
    trend_color = "green" if trend_state == "BULLISH" else "red"
    
    with col1:
        st.markdown("### Trend Following")
        st.markdown(f"Status: <span style='color:{trend_color}; font-size:24px'>{trend_state}</span>", unsafe_allow_html=True)
        st.write(f"Current Price > {sma_window}-Day SMA")
        st.progress(max(0.0, min(1.0, (latest['Close'] / latest[f'SMA_{sma_window}']) - 0.5))) # Just a visual thing
        st.caption("Strategy goes LONG, when Price is above SMA. Moves to Cash below.")

    # 2. Momentum Signal
    mom_val = latest[f'Momentum_{mom_window}M_1M']
    mom_state = "POSITIVE" if mom_val > 0 else "NEGATIVE"
    mom_color = "green" if mom_val > 0 else "red"
    
    with col2:
        st.markdown(f"### Momentum ({mom_window}-1)")
        st.markdown(f"Status: <span style='color:{mom_color}; font-size:24px'>{mom_state}</span> ({mom_val:.1%})", unsafe_allow_html=True)
        st.write("Past performance (excl. last month)")
        st.caption("Academic momentum factor (Jegadeesh & Titman). Positive momentum implies further upside statistically.")

    # 3. RSI Signal (Mean Reversion)
    rsi_val = latest['RSI_14']
    rsi_state = "NEUTRAL"
    if rsi_val > 70: rsi_state = "OVERBOUGHT"
    elif rsi_val < 30: rsi_state = "OVERSOLD"
    
    with col3:
        st.markdown("### Mean Reversion (RSI)")
        st.markdown(f"Status: <strong style='font-size:24px'>{rsi_state}</strong> ({rsi_val:.1f})", unsafe_allow_html=True)
        st.slider("RSI Level", 0, 100, int(rsi_val), disabled=True)
        st.caption("Extremes (>70 or <30) may indicate potential reversals.")

    st.markdown("---")
    st.subheader("Signal Correlation")
    # Show correlation between price changes and signal changes?
    # Maybe just a simple correlation matrix of indicators
    corr_cols = ['Daily_Return', f'Momentum_{mom_window}M_1M', 'RSI_14', 'Vol_21d']
    st.write(df[corr_cols].corr())

# --- TAB 3: BACKTEST ---
with tab_bt:
    st.subheader("Strategy Simulation")
    
    bt_method = st.radio("Select Strategy to Test", ["Trend Following (SMA)", "Momentum Only", "Trend + Momentum"], horizontal=True)
    
    # 1. Define Signal Column
    if bt_method == "Trend Following (SMA)":
        # 1 if Price > SMA, else 0 (or -1 if short)
        df['Signal_Test'] = np.where(df['Close'] > df[f'SMA_{sma_window}'], 1, -1 if allow_short else 0)
    elif bt_method == "Momentum Only":
        df['Signal_Test'] = np.where(df[f'Momentum_{mom_window}M_1M'] > 0, 1, -1 if allow_short else 0)
    else:
        # Combined: Must be Bullish Trend AND Positive Momentum
        c1 = df['Close'] > df[f'SMA_{sma_window}']
        c2 = df[f'Momentum_{mom_window}M_1M'] > 0
        df['Signal_Test'] = np.where(c1 & c2, 1, 0) # Conservative: Cash if any fails
        
    # Run Backtest
    # Default monthly for stability, but let's offer Daily strictly for Trend?
    # User asked for monthly re-balancing in original spec, but user *inputs* allow flexibility.
    # Let's stick to Monthly for robustness as Daily SMA crossing creates huge drag.
    
    res_df = backtester.run_backtest(df, 'Signal_Test', cost_bps=bt_cost, rebalance_freq='M')
    
    if not res_df.empty:
        # Metrics
        strat_metrics = backtester.calculate_perf_metrics(res_df['Equity_Strategy'])
        bench_metrics = backtester.calculate_perf_metrics(res_df['Equity_Benchmark'])
        
        # Display Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Strategy CAGR", f"{strat_metrics['CAGR']:.2%}", delta=f"{strat_metrics['CAGR']-bench_metrics['CAGR']:.2%} vs Bench")
        col_m2.metric("Sharpe Ratio", f"{strat_metrics['Sharpe']:.2f}")
        col_m3.metric("Max Drawdown", f"{strat_metrics['MaxDD']:.2%}")
        
        # Detailed Table
        with st.expander("Full Performance Stats"):
            stats_df = pd.DataFrame([strat_metrics, bench_metrics], index=["Strategy", "Benchmark"])
            st.dataframe(stats_df.style.format("{:.2f}"))
            
        # Charts
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=res_df.index, y=res_df['Equity_Strategy'], name='Strategy', line=dict(color='#00ff00')))
        fig_eq.add_trace(go.Scatter(x=res_df.index, y=res_df['Equity_Benchmark'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
        fig_eq.update_layout(title="Equity Curve", template="plotly_dark", height=500, yaxis_title="Growth of $1")
        st.plotly_chart(fig_eq, use_container_width=True)
        
        # Drawdown
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=res_df.index, y=res_df['DD_Strategy'], name='Strategy DD', fill='tozeroy', line=dict(color='red')))
        fig_dd.update_layout(title="Drawdown Profile", template="plotly_dark", height=300, yaxis_title="% Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)
        
    else:
        st.warning("Backtest calculation failed.")

# --- TAB 4: REPORT ---
with tab_rep:
    st.subheader("Export Analysis")
    st.write("Download the processed data and backtest results for further analysis.")
    
    if not res_df.empty:
        csv_data = res_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name=f"{ticker}_analysis_report.csv",
            mime="text/csv"
        )
        
        st.markdown("### Summary Note")
        st.write(f"""
        **Analysis for {ticker}:**
        - **Period**: {df.index[0].date()} to {df.index[-1].date()}
        - **Model**: {bt_method}
        - **Performance**: The strategy achieved a CAGR of **{strat_metrics['CAGR']:.2%}** with a Sharpe Ratio of **{strat_metrics['Sharpe']:.2f}**.
        
        *Disclaimer: Past performance is not indicative of future results.*
        """)
