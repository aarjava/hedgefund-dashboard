import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame, sma_window: int = 50, mom_window: int = 12, vol_window: int = 21) -> pd.DataFrame:
    """
    Adds technical indicators to the dataframe.
    
    Args:
        df (pd.DataFrame): OHLCV data.
        sma_window (int): Trend lookback (days).
        mom_window (int): Momentum lookback (months).
        vol_window (int): Volatility lookback (days).
        
    Returns:
        pd.DataFrame: DF with added columns.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # 1. Trend: Moving Averages
    df[f'SMA_{sma_window}'] = df['Close'].rolling(window=sma_window).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean() # Standard long-term benchmark
    
    # 2. Momentum (12-1 Month equivalent)
    # We approximate months as 21 trading days.
    # Momentum 12-1 = Return from 12 months ago to 1 month ago.
    # t-252 to t-21
    # Check if we have enough data
    lag_start = 21
    lag_end = 252 # approx 12 months
    
    # Customize if mom_window is different from standard 12
    # If mom_window = X, we look back X months.
    lag_end_custom = mom_window * 21
    
    df[f'Momentum_{mom_window}M_1M'] = df['Close'].shift(lag_start) / df['Close'].shift(lag_end_custom) - 1
    
    # 3. Volatility (Annualized)
    df['Daily_Return'] = df['Close'].pct_change()
    df[f'Vol_{vol_window}d'] = df['Daily_Return'].rolling(window=vol_window).std() * (252**0.5)
    
    # 4. Relative Strength Index (RSI) - 14 day standard
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 5. Distance from SMA (Trend Strength)
    df['Trend_Strength_Pct'] = (df['Close'] - df[f'SMA_{sma_window}']) / df[f'SMA_{sma_window}']
    
    return df
