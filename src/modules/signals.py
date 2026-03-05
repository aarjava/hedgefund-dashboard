import pandas as pd
import numpy as np
from typing import Literal, Optional

try:
    from .config import (
        TRADING_DAYS_PER_YEAR,
        TRADING_DAYS_PER_MONTH,
        DEFAULT_RSI_WINDOW,
        MIN_PERIODS_FOR_EXPANDING,
    )
except ImportError:
    # Fallback for direct execution
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    DEFAULT_RSI_WINDOW = 14
    MIN_PERIODS_FOR_EXPANDING = 60


def add_technical_indicators(
    df: pd.DataFrame, sma_window: int = 50, mom_window: int = 12, vol_window: int = 21
) -> pd.DataFrame:
    """
    Adds technical indicators to the dataframe.

    Args:
        df: OHLCV data with 'Close' column.
        sma_window: Trend lookback in days.
        mom_window: Momentum lookback in months.
        vol_window: Volatility lookback in days.

    Returns:
        DataFrame with added indicator columns.
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. Trend: Moving Averages
    df[f"SMA_{sma_window}"] = df["Close"].rolling(window=sma_window).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()  # Standard long-term benchmark

    # 2. Momentum (12-1 Month equivalent)
    # We approximate months as 21 trading days.
    # Momentum 12-1 = Return from 12 months ago to 1 month ago.
    lag_start = TRADING_DAYS_PER_MONTH  # Skip most recent month
    lag_end_custom = mom_window * TRADING_DAYS_PER_MONTH

    df[f"Momentum_{mom_window}M_1M"] = (
        df["Close"].shift(lag_start) / df["Close"].shift(lag_end_custom) - 1
    )

    # 3. Volatility (Annualized)
    df["Daily_Return"] = df["Close"].pct_change()
    df[f"Vol_{vol_window}d"] = df["Daily_Return"].rolling(window=vol_window).std() * (
        TRADING_DAYS_PER_YEAR**0.5
    )

    # 4. Relative Strength Index (RSI) - Vectorized calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=DEFAULT_RSI_WINDOW).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=DEFAULT_RSI_WINDOW).mean()
    rs = gain / loss.replace(0, np.nan)  # Avoid division by zero
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 5. Distance from SMA (Trend Strength)
    df["Trend_Strength_Pct"] = (df["Close"] - df[f"SMA_{sma_window}"]) / df[f"SMA_{sma_window}"]

    return df


def detect_volatility_regime(
    df: pd.DataFrame,
    vol_col: str = "Vol_21d",
    quantile_high: float = 0.75,
    quantile_low: float = 0.25,
    use_expanding: bool = False,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Classifies periods into Volatility Regimes (Low, Normal, High).

    Args:
        df: DataFrame containing the volatility column.
        vol_col: Name of the volatility column.
        quantile_high: Percentile threshold for High Volatility (e.g., 0.75).
        quantile_low: Percentile threshold for Low Volatility (e.g., 0.25).
        use_expanding: If True, uses expanding window quantiles to avoid look-ahead bias.
                      This is preferred for out-of-sample backtesting validity.
                      If False, uses full-sample quantiles (faster, for exploratory analysis).
        min_periods: Minimum periods required for expanding window calculation.
                    Only used if use_expanding=True. Defaults to MIN_PERIODS_FOR_EXPANDING.

    Returns:
        DataFrame with 'Vol_Regime' column:
            'High' if vol > quantile_high threshold
            'Low' if vol < quantile_low threshold
            'Normal' otherwise

    Note:
        When use_expanding=False (default), regime classification uses full-sample
        quantiles which introduces look-ahead bias. This is acceptable for exploratory
        analysis but not for rigorous out-of-sample backtesting.
    """
    if df.empty or vol_col not in df.columns:
        return df

    df = df.copy()

    if min_periods is None:
        min_periods = MIN_PERIODS_FOR_EXPANDING

    if use_expanding:
        # OUT-OF-SAMPLE: Expanding window quantiles (no look-ahead bias)
        # At each point in time, we only use data available up to that point
        thresh_high = df[vol_col].expanding(min_periods=min_periods).quantile(quantile_high)
        thresh_low = df[vol_col].expanding(min_periods=min_periods).quantile(quantile_low)

        # Vectorized regime classification with expanding thresholds
        df["Vol_Regime"] = "Normal"
        df.loc[df[vol_col] > thresh_high, "Vol_Regime"] = "High"
        df.loc[df[vol_col] < thresh_low, "Vol_Regime"] = "Low"

        # Mark early periods as 'Unknown' where we don't have enough data
        df.loc[thresh_high.isna(), "Vol_Regime"] = "Unknown"
    else:
        # IN-SAMPLE: Full-sample quantiles (look-ahead bias, but standard for regime analysis)
        # Use this for exploratory analysis and visualization
        thresh_high = df[vol_col].quantile(quantile_high)
        thresh_low = df[vol_col].quantile(quantile_low)

        conditions = [(df[vol_col] > thresh_high), (df[vol_col] < thresh_low)]
        choices = ["High", "Low"]

        df["Vol_Regime"] = np.select(conditions, choices, default="Normal")

    return df


def detect_volatility_regime_oos(
    df: pd.DataFrame,
    vol_col: str = "Vol_21d",
    quantile_high: float = 0.75,
    quantile_low: float = 0.25,
    min_periods: int = MIN_PERIODS_FOR_EXPANDING,
) -> pd.DataFrame:
    """
    Convenience wrapper for out-of-sample regime detection.

    This function should be used for backtesting to ensure no look-ahead bias.

    Args:
        df: DataFrame containing the volatility column.
        vol_col: Name of the volatility column.
        quantile_high: Percentile threshold for High Volatility.
        quantile_low: Percentile threshold for Low Volatility.
        min_periods: Minimum periods required before classification starts.

    Returns:
        DataFrame with 'Vol_Regime' column using expanding-window quantiles.
    """
    return detect_volatility_regime(
        df=df,
        vol_col=vol_col,
        quantile_high=quantile_high,
        quantile_low=quantile_low,
        use_expanding=True,
        min_periods=min_periods,
    )
