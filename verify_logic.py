import pandas as pd
import numpy as np
import yfinance as yf


# Copying the logic from dashboard.py for verification
def calculate_metrics(df):
    """Calculates SMAs, Volatility, Returns, and Momentum."""
    if df.empty:
        return df
    df = df.copy()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Momentum_12_1"] = df["Close"].shift(21) / df["Close"].shift(252) - 1
    return df


def run_backtest(df):
    if df.empty or "SMA_50" not in df.columns:
        return None

    bt_df = df.copy()
    # 1. Trend Signal
    bt_df["Signal_Trend"] = np.where(bt_df["Close"] > bt_df["SMA_50"], 1, 0)

    # 2. Monthly Rebalance Logic
    bt_df["Month"] = bt_df.index.to_period("M")
    monthly_signals = bt_df.groupby("Month")["Signal_Trend"].last()
    monthly_positions = monthly_signals.shift(1)
    bt_df["Position"] = bt_df["Month"].map(monthly_positions)
    bt_df["Position"] = bt_df["Position"].fillna(0)

    # 3. Returns and Costs
    bt_df["Strategy_Return"] = bt_df["Position"] * bt_df["Daily_Return"]
    bt_df["Trade_Size"] = bt_df["Position"].diff().abs().fillna(0)
    cost_bps = 0.0010
    bt_df["Cost"] = bt_df["Trade_Size"] * cost_bps
    bt_df["Strategy_Net_Return"] = bt_df["Strategy_Return"] - bt_df["Cost"]

    bt_df["Equity_Strategy"] = (1 + bt_df["Strategy_Net_Return"]).cumprod()

    return bt_df


# Test
print("Fetching data...")
try:
    df = yf.Ticker("SPY").history(period="2y")
    if df.empty:
        print("Error: No data fetched for SPY")
    else:
        print(f"Data fetched: {len(df)} rows")
        df = calculate_metrics(df)
        print("Metrics calculated.")

        bt_results = run_backtest(df)
        if bt_results is not None:
            print("Backtest run successfully.")
            print(
                bt_results[
                    ["Close", "SMA_50", "Signal_Trend", "Position", "Strategy_Net_Return"]
                ].tail()
            )

            final_return = bt_results["Equity_Strategy"].iloc[-1] - 1
            print(f"Final Strategy Return: {final_return:.2%}")
        else:
            print("Backtest returned None")

except Exception as e:
    print(f"FAILED: {e}")
