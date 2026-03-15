"""
Data ingestion module for fetching historical market data.

Supports fetching from Yahoo Finance with caching for performance.
"""

import logging
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from .config import CACHE_TTL_SECONDS
except ImportError:
    CACHE_TTL_SECONDS = 3600 * 24  # 24 hours

# Configure module logger
logger = logging.getLogger(__name__)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_stock_data(ticker: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance with caching.

    Args:
        ticker: The asset symbol (e.g., 'SPY', 'BTC-USD').
        period: Time period string - '1y', '5y', '10y', 'max', etc.
        interval: Data interval - '1d', '1wk', '1mo'.

    Returns:
        DataFrame with Date index and columns: Open, High, Low, Close, Volume.
        Returns empty DataFrame on error.
    """
    logger.info(f"Fetching data for {ticker}, period={period}, interval={interval}")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return df

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Drop timezone if present for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_multi_asset_data(tickers: tuple, period: str = "10y", interval: str = "1d") -> dict:
    """
    Fetch historical data for multiple tickers.

    Args:
        tickers: Tuple of ticker symbols.
        period: Time period string.
        interval: Data interval.

    Returns:
        Dict mapping ticker -> DataFrame.
    """
    data = {}
    for t in tickers:
        df = fetch_stock_data(t, period=period, interval=interval)
        data[t] = df
    return data


def align_close_prices(data: dict) -> pd.DataFrame:
    """
    Align close prices for multiple assets on a common date index.

    Args:
        data: Dict mapping ticker -> DataFrame with 'Close'.

    Returns:
        DataFrame of close prices with tickers as columns.
    """
    frames = []
    for t, df in data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].rename(t)
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1, join="inner").sort_index()
    return prices.dropna(how="any")


def align_volume(data: dict) -> pd.DataFrame:
    """
    Align volume for multiple assets on a common date index.
    """
    frames = []
    for t, df in data.items():
        if df is None or df.empty or "Volume" not in df.columns:
            continue
        s = df["Volume"].rename(t)
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    vol = pd.concat(frames, axis=1, join="inner").sort_index()
    return vol.dropna(how="any")


def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol exists and has data.

    Args:
        ticker: The ticker symbol to validate.

    Returns:
        True if ticker is valid and has data, False otherwise.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Check if we got valid info back
        return info.get("regularMarketPrice") is not None
    except Exception as e:
        logger.debug(f"Ticker validation failed for {ticker}: {e}")
        return False


def get_ticker_info(ticker: str) -> Optional[dict]:
    """
    Get basic info about a ticker.

    Args:
        ticker: The ticker symbol.

    Returns:
        Dictionary with ticker info or None on error.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
        }
    except Exception as e:
        logger.debug(f"Could not get info for {ticker}: {e}")
        return None
