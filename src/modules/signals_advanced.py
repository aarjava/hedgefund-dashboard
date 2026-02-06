"""
Advanced signal generation module.

Provides additional trading signals beyond basic trend and momentum,
including mean reversion, volatility breakout, and composite signals.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config import (
        DEFAULT_RSI_WINDOW,
        TRADING_DAYS_PER_MONTH,
        TRADING_DAYS_PER_YEAR,
    )
except ImportError:
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    DEFAULT_RSI_WINDOW = 14

logger = logging.getLogger(__name__)


def calculate_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0, price_col: str = "Close"
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for mean reversion signals.

    Args:
        df: DataFrame with price data.
        window: Lookback window for moving average.
        num_std: Number of standard deviations for bands.
        price_col: Column name for price data.

    Returns:
        DataFrame with added columns: BB_Middle, BB_Upper, BB_Lower, BB_Width, BB_Position.
    """
    if df.empty or price_col not in df.columns:
        logger.warning(f"Invalid input for Bollinger Bands: missing '{price_col}'")
        return df

    df = df.copy()

    # Calculate bands
    df["BB_Middle"] = df[price_col].rolling(window=window).mean()
    rolling_std = df[price_col].rolling(window=window).std()
    df["BB_Upper"] = df["BB_Middle"] + (rolling_std * num_std)
    df["BB_Lower"] = df["BB_Middle"] - (rolling_std * num_std)

    # Band width (volatility measure)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # Price position within bands (-1 at lower, 0 at middle, +1 at upper)
    df["BB_Position"] = (df[price_col] - df["BB_Middle"]) / (df["BB_Upper"] - df["BB_Middle"])

    logger.debug(f"Bollinger Bands calculated: window={window}, std={num_std}")
    return df


def generate_mean_reversion_signal(
    df: pd.DataFrame,
    rsi_col: str = "RSI_14",
    oversold: int = 30,
    overbought: int = 70,
    use_bollinger: bool = True,
    bb_position_col: str = "BB_Position",
) -> pd.Series:
    """
    Generate mean reversion signal based on RSI and optionally Bollinger Bands.

    Signal Logic:
    - Buy (1): RSI oversold AND (optionally) price near lower BB
    - Sell (-1): RSI overbought AND (optionally) price near upper BB
    - Hold (0): Otherwise

    Args:
        df: DataFrame with RSI and optionally BB columns.
        rsi_col: Column name for RSI values.
        oversold: RSI threshold for oversold condition.
        overbought: RSI threshold for overbought condition.
        use_bollinger: Whether to incorporate Bollinger Band position.
        bb_position_col: Column name for BB position.

    Returns:
        Series with signal values: 1 (buy), -1 (sell), 0 (hold).
    """
    if df.empty or rsi_col not in df.columns:
        logger.error(f"Missing required column: {rsi_col}")
        return pd.Series(0, index=df.index)

    signal = pd.Series(0, index=df.index)

    # RSI conditions
    rsi_oversold = df[rsi_col] < oversold
    rsi_overbought = df[rsi_col] > overbought

    if use_bollinger and bb_position_col in df.columns:
        # Combine with Bollinger Band position
        bb_low = df[bb_position_col] < -0.8  # Near lower band
        bb_high = df[bb_position_col] > 0.8  # Near upper band

        signal[rsi_oversold & bb_low] = 1  # Strong buy
        signal[rsi_overbought & bb_high] = -1  # Strong sell
    else:
        # RSI only
        signal[rsi_oversold] = 1
        signal[rsi_overbought] = -1

    logger.info(f"Mean reversion signal: {(signal == 1).sum()} buy, {(signal == -1).sum()} sell")
    return signal


def generate_volatility_breakout_signal(
    df: pd.DataFrame,
    vol_col: str = "Vol_21d",
    vol_threshold_percentile: float = 0.80,
    trend_col: Optional[str] = None,
) -> pd.Series:
    """
    Generate volatility breakout signal.

    Signal Logic:
    - When volatility spikes above threshold, follow the trend direction
    - This aims to capture momentum during volatility expansion

    Args:
        df: DataFrame with volatility data.
        vol_col: Column name for volatility.
        vol_threshold_percentile: Percentile threshold for "high" vol.
        trend_col: Optional column indicating trend (1=up, -1=down).

    Returns:
        Series with signal values.
    """
    if df.empty or vol_col not in df.columns:
        logger.error(f"Missing required column: {vol_col}")
        return pd.Series(0, index=df.index)

    df = df.copy()

    # Expanding threshold (no look-ahead bias)
    vol_threshold = df[vol_col].expanding(min_periods=60).quantile(vol_threshold_percentile)
    high_vol = df[vol_col] > vol_threshold

    # Determine trend direction from recent returns if not provided
    if trend_col is None or trend_col not in df.columns:
        # Use 5-day return direction
        df["_temp_trend"] = np.sign(df["Close"].pct_change(5))
        trend = df["_temp_trend"]
    else:
        trend = df[trend_col]

    # Signal: follow trend when vol is high
    signal = pd.Series(0, index=df.index)
    signal[high_vol] = trend[high_vol]

    # Clean up
    if "_temp_trend" in df.columns:
        df.drop("_temp_trend", axis=1, inplace=True)

    logger.info(f"Volatility breakout signal: {(signal != 0).sum()} active days")
    return signal


def generate_dual_momentum_signal(
    df: pd.DataFrame,
    abs_mom_col: str = "Momentum_12M_1M",
    rel_benchmark_return: Optional[pd.Series] = None,
    abs_threshold: float = 0.0,
) -> pd.Series:
    """
    Generate dual momentum signal (absolute + relative momentum).

    Signal Logic:
    - Long (1): Positive absolute momentum AND better than benchmark
    - Cash (0): Otherwise

    This is based on Gary Antonacci's dual momentum research.

    Args:
        df: DataFrame with momentum column.
        abs_mom_col: Column for absolute momentum.
        rel_benchmark_return: Optional benchmark return series for relative momentum.
        abs_threshold: Threshold for considering momentum "positive".

    Returns:
        Series with signal values: 1 (long) or 0 (cash).
    """
    if df.empty or abs_mom_col not in df.columns:
        logger.error(f"Missing required column: {abs_mom_col}")
        return pd.Series(0, index=df.index)

    # Absolute momentum: is the asset trending up?
    abs_mom_positive = df[abs_mom_col] > abs_threshold

    if rel_benchmark_return is not None:
        # Relative momentum: is the asset beating the benchmark?
        rel_mom_positive = df[abs_mom_col] > rel_benchmark_return
        signal = pd.Series(0, index=df.index)
        signal[abs_mom_positive & rel_mom_positive] = 1
    else:
        # Just absolute momentum
        signal = pd.Series(0, index=df.index)
        signal[abs_mom_positive] = 1

    logger.info(
        f"Dual momentum signal: {(signal == 1).sum()} long days, {(signal == 0).sum()} cash days"
    )
    return signal


def generate_composite_signal(
    df: pd.DataFrame, signals: dict, weights: Optional[dict] = None, threshold: float = 0.5
) -> pd.Series:
    """
    Combine multiple signals into a composite signal.

    Args:
        df: DataFrame (used for index).
        signals: Dictionary of {name: signal_series}.
        weights: Optional dictionary of {name: weight}. Defaults to equal weights.
        threshold: Threshold for composite signal to trigger position.

    Returns:
        Series with signal values: 1 (long), -1 (short), 0 (neutral).
    """
    if not signals:
        logger.error("No signals provided")
        return pd.Series(0, index=df.index)

    # Default to equal weights
    if weights is None:
        weights = {name: 1.0 / len(signals) for name in signals}

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate weighted average
    composite = pd.Series(0.0, index=df.index)
    for name, signal in signals.items():
        weight = weights.get(name, 0)
        composite += signal * weight

    # Convert to discrete signal
    final_signal = pd.Series(0, index=df.index)
    final_signal[composite >= threshold] = 1
    final_signal[composite <= -threshold] = -1

    logger.info(
        f"Composite signal: {(final_signal == 1).sum()} long, "
        f"{(final_signal == -1).sum()} short, {(final_signal == 0).sum()} neutral"
    )
    return final_signal


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for position sizing and stops.

    Args:
        df: DataFrame with High, Low, Close columns.
        window: ATR lookback period.

    Returns:
        Series with ATR values.
    """
    if df.empty or not all(col in df.columns for col in ["High", "Low", "Close"]):
        logger.warning("Missing required columns for ATR calculation")
        return pd.Series(dtype=float, index=df.index if not df.empty else None)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Average True Range
    atr = true_range.rolling(window=window).mean()

    logger.debug(f"ATR calculated: window={window}, current={atr.iloc[-1]:.4f}")
    return atr


def calculate_position_size(
    account_value: float,
    risk_per_trade: float,
    atr: float,
    atr_multiplier: float = 2.0,
    price: float = 1.0,
) -> Tuple[int, float]:
    """
    Calculate position size based on ATR volatility.

    Args:
        account_value: Total account value.
        risk_per_trade: Fraction of account to risk (e.g., 0.01 for 1%).
        atr: Current ATR value.
        atr_multiplier: Multiplier for stop distance (e.g., 2 ATR).
        price: Current asset price.

    Returns:
        Tuple of (shares, stop_distance).
    """
    if atr <= 0 or price <= 0:
        logger.warning("Invalid ATR or price for position sizing")
        return (0, 0.0)

    risk_amount = account_value * risk_per_trade
    stop_distance = atr * atr_multiplier

    # Position size = Risk Amount / Risk per Share
    position_value = risk_amount / (stop_distance / price)
    shares = int(position_value / price)

    logger.debug(
        f"Position size: {shares} shares, "
        f"stop_distance={stop_distance:.2f}, risk=${risk_amount:.2f}"
    )
    return (shares, stop_distance)
