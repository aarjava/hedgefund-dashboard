import pandas as pd
import numpy as np

def calculate_perf_metrics(equity_curve: pd.Series, freq: int = 252) -> dict:
    """
    Calculates comprehensive performance metrics.
    """
    if equity_curve.empty:
        return {}

    # Returns
    daily_rets = equity_curve.pct_change().dropna()
    
    # Total Time
    try:
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        if years < 0.1: years = 0.1 # Avoid div by zero for short periods
    except:
        years = 1

    total_return = equity_curve.iloc[-1] - 1
    cagr = (equity_curve.iloc[-1])**(1/years) - 1

    # Volatility
    ann_vol = daily_rets.std() * (freq**0.5)

    # Sharpe (Assume 0% risk free for simplicity or make configurable)
    sharpe = cagr / ann_vol if ann_vol != 0 else 0

    # Sortino (Downside deviation)
    downside_rets = daily_rets[daily_rets < 0]
    downside_dev = downside_rets.std() * (freq**0.5)
    sortino = cagr / downside_dev if downside_dev != 0 else 0

    # Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Win Rate (Daily)
    win_rate = (daily_rets > 0).mean()

    return {
        "CAGR": cagr,
        "Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "WinRate": win_rate
    }

def run_backtest(df: pd.DataFrame, 
                 signal_col: str, 
                 cost_bps: float = 0.0010, 
                 rebalance_freq: str = 'M') -> pd.DataFrame:
    """
    Runs a backtest based on a signal column.
    
    Args:
        df: Dataframe with date index, 'Close', 'Daily_Return' and the signal column.
        signal_col: Name of column with 1 (Long), 0 (Cash), -1 (Short).
        cost_bps: Cost per trade (e.g. 0.0010 for 10bps).
        rebalance_freq: 'D' for daily, 'M' for monthly, 'W' for weekly.
    """
    if df.empty or signal_col not in df.columns:
        return pd.DataFrame()

    bt_df = df.copy()
    
    # 1. Signal Processing
    # We assume 'signal_col' contains the target position for the NEXT period.
    # But usually, signals are generated EOD. So Position tomorrow = Signal today.
    # For Monthly rebalancing:
    # We take the signal at the last day of the month to determine position for next month.
    
    if rebalance_freq == 'D':
        # Daily Rebalance: Position today is determined by Signal yesterday
        bt_df['Position'] = bt_df[signal_col].shift(1).fillna(0)
    
    elif rebalance_freq == 'M':
        # Monthly Rebalance
        # Create a period mapping
        bt_df['Period'] = bt_df.index.to_period('M')
        
        # Get signal at the very last available day of each month
        # Logic: Valid signal is the one present at month-end.
        # We'll forward fill signals to ensure if month-end is missing we take last known.
        # Actually safer to resample.
        
        monthly_signals = bt_df.groupby('Period')[signal_col].last()
        
        # Position for Month X is derived from Signal of Month X-1
        monthly_positions = monthly_signals.shift(1)
        
        # Map back to daily
        bt_df['Position'] = bt_df['Period'].map(monthly_positions)
        bt_df['Position'] = bt_df['Position'].fillna(0)
        
    # 2. Strategy Returns
    bt_df['Strategy_Return'] = bt_df['Position'] * bt_df['Daily_Return']
    
    # 3. Transaction Costs
    # Change in position * Cost
    bt_df['Position_Change'] = bt_df['Position'].diff().abs().fillna(0)
    bt_df['Cost'] = bt_df['Position_Change'] * cost_bps
    
    bt_df['Strategy_Net_Return'] = bt_df['Strategy_Return'] - bt_df['Cost']
    
    # 4. Equity Curves
    bt_df['Equity_Benchmark'] = (1 + bt_df['Daily_Return']).cumprod()
    bt_df['Equity_Strategy'] = (1 + bt_df['Strategy_Net_Return']).cumprod()
    
    # 5. Drawdown Curves
    bt_df['DD_Benchmark'] = (bt_df['Equity_Benchmark'] / bt_df['Equity_Benchmark'].cummax()) - 1
    bt_df['DD_Strategy'] = (bt_df['Equity_Strategy'] / bt_df['Equity_Strategy'].cummax()) - 1
    
    return bt_df
