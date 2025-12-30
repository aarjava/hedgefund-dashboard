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

def detect_volatility_regime(df: pd.DataFrame, vol_col: str = 'Vol_21d', quantile_high: float = 0.75, quantile_low: float = 0.25) -> pd.DataFrame:
    """
    Classifies periods into Volatility Regimes (Low, Normal, High).
    
    Args:
        df (pd.DataFrame): Dataframe containing the volatility column.
        vol_col (str): Name of the volatility column.
        quantile_high (float): Percentile threshold for High Volatility (e.g., 0.75).
        quantile_low (float): Percentile threshold for Low Volatility (e.g., 0.25).
        
    Returns:
        pd.DataFrame: DF with 'Vol_Regime' column.
                     'High' if vol > quantile_high
                     'Low' if vol < quantile_low
                     'Normal' otherwise.
    """
    if df.empty or vol_col not in df.columns:
        return df
    
    df = df.copy()
    
    # Calculate thresholds based on the entire history (Hindsight bias: YES, but standard for regime analysis)
    # Ideally, one would use a rolling window for true out-of-sample, but for this research question,
    # we want to categorize the historical distribution.
    
    thresh_high = df[vol_col].quantile(quantile_high)
    thresh_low = df[vol_col].quantile(quantile_low)
    
    conditions = [
        (df[vol_col] > thresh_high),
        (df[vol_col] < thresh_low)
    ]
    choices = ['High', 'Low']
    
    df['Vol_Regime'] = np.select(conditions, choices, default='Normal')
    
    return df
