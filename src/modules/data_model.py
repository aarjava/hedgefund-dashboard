import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600*24) # Cache for 24 hours
def fetch_stock_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    """
    Fetches historical data from yfinance with caching.
    
    Args:
        ticker (str): The asset symbol (e.g., 'SPY', 'BTC-USD').
        period (str): varying period string '1y', '5y', 'max', etc.
        
    Returns:
        pd.DataFrame: Dataframe with Date index and columns like Open, High, Low, Close, Volume.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Basic cleaning
        if df.empty:
            return df
            
        # Drop timezone if present for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
