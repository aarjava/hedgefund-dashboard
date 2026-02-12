"""
Backtesting engine for momentum and trend-following strategies.

Features:
- Vectorized backtest simulation
- Transaction cost modeling
- Performance metrics calculation
- Conditional statistics by regime
- Bootstrap confidence intervals for Sharpe ratio
- Walk-forward validation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config import TRADING_DAYS_PER_YEAR
except ImportError:
    TRADING_DAYS_PER_YEAR = 252

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics with confidence intervals."""

    cagr: float
    volatility: float
    sharpe: float
    sharpe_ci_lower: Optional[float] = None
    sharpe_ci_upper: Optional[float] = None
    sortino: float = 0.0
    max_dd: float = 0.0
    max_dd_duration: int = 0  # Days
    avg_dd_duration: float = 0.0  # Days
    calmar: float = 0.0
    win_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "CAGR": self.cagr,
            "Vol": self.volatility,
            "Sharpe": self.sharpe,
            "Sharpe_CI_Lower": self.sharpe_ci_lower,
            "Sharpe_CI_Upper": self.sharpe_ci_upper,
            "Sortino": self.sortino,
            "MaxDD": self.max_dd,
            "MaxDD_Duration": self.max_dd_duration,
            "AvgDD_Duration": self.avg_dd_duration,
            "Calmar": self.calmar,
            "WinRate": self.win_rate,
        }


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for Sharpe ratio.

    Args:
        returns: Series of daily returns.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound) for the CI.
    """
    if len(returns) < 30:
        logger.warning("Insufficient data for reliable bootstrap CI (n < 30)")
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)
    sharpes = []

    returns_arr = returns.dropna().values
    n = len(returns_arr)

    for _ in range(n_bootstrap):
        sample = rng.choice(returns_arr, size=n, replace=True)
        sample_std = sample.std()
        if sample_std > 0:
            sample_sharpe = sample.mean() / sample_std * np.sqrt(TRADING_DAYS_PER_YEAR)
            sharpes.append(sample_sharpe)

    if not sharpes:
        return (np.nan, np.nan)

    alpha = 1 - confidence_level
    lower = np.percentile(sharpes, alpha / 2 * 100)
    upper = np.percentile(sharpes, (1 - alpha / 2) * 100)

    logger.debug(f"Bootstrap Sharpe CI ({confidence_level:.0%}): [{lower:.3f}, {upper:.3f}]")
    return (lower, upper)


def calculate_drawdown_duration(equity_curve: pd.Series) -> Tuple[int, float]:
    """
    Calculate maximum and average drawdown duration.

    Args:
        equity_curve: Series of equity values (cumulative returns).

    Returns:
        Tuple of (max_duration_days, avg_duration_days).
    """
    if equity_curve.empty:
        return (0, 0.0)

    rolling_max = equity_curve.cummax()
    underwater = equity_curve < rolling_max

    # Find contiguous underwater periods
    underwater_periods = []
    current_duration = 0

    for is_underwater in underwater:
        if is_underwater:
            current_duration += 1
        else:
            if current_duration > 0:
                underwater_periods.append(current_duration)
            current_duration = 0

    # Don't forget the last period if still underwater
    if current_duration > 0:
        underwater_periods.append(current_duration)

    if not underwater_periods:
        return (0, 0.0)

    max_duration = max(underwater_periods)
    avg_duration = sum(underwater_periods) / len(underwater_periods)

    logger.debug(f"Drawdown durations: max={max_duration} days, avg={avg_duration:.1f} days")
    return (max_duration, avg_duration)


def calculate_perf_metrics(
    equity_curve: pd.Series,
    freq: int = TRADING_DAYS_PER_YEAR,
    include_bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: Series of equity values (starting from 1.0).
        freq: Trading days per year for annualization.
        include_bootstrap_ci: Whether to compute bootstrap CI for Sharpe.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        Dictionary of performance metrics.
    """
    if equity_curve.empty:
        logger.warning("Empty equity curve provided")
        return {}

    # Returns
    daily_rets = equity_curve.pct_change().dropna()

    # Total Time
    try:
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        if years < 0.1:
            years = 0.1  # Avoid div by zero for short periods
            logger.warning("Very short period detected, using minimum 0.1 years")
    except (IndexError, TypeError) as e:
        logger.warning(f"Could not calculate period: {e}")
        years = 1

    cagr = (equity_curve.iloc[-1]) ** (1 / years) - 1

    # Volatility
    ann_vol = daily_rets.std() * (freq**0.5)

    # Sharpe (Assume 0% risk free for simplicity)
    sharpe = cagr / ann_vol if ann_vol != 0 else 0

    # Bootstrap CI for Sharpe
    sharpe_ci_lower, sharpe_ci_upper = None, None
    if include_bootstrap_ci:
        sharpe_ci_lower, sharpe_ci_upper = bootstrap_sharpe_ci(daily_rets, n_bootstrap=n_bootstrap)

    # Sortino (Downside deviation)
    downside_rets = daily_rets[daily_rets < 0]
    downside_dev = downside_rets.std() * (freq**0.5) if len(downside_rets) > 0 else 0
    sortino = cagr / downside_dev if downside_dev != 0 else 0

    # Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Drawdown Duration
    max_dd_duration, avg_dd_duration = calculate_drawdown_duration(equity_curve)

    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win Rate (Daily)
    win_rate = (daily_rets > 0).mean()

    logger.info(f"Performance: CAGR={cagr:.2%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}")

    return {
        "CAGR": cagr,
        "Vol": ann_vol,
        "Sharpe": sharpe,
        "Sharpe_CI_Lower": sharpe_ci_lower,
        "Sharpe_CI_Upper": sharpe_ci_upper,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "MaxDD_Duration": max_dd_duration,
        "AvgDD_Duration": avg_dd_duration,
        "Calmar": calmar,
        "WinRate": win_rate,
    }


def run_backtest(
    df: pd.DataFrame,
    signal_col: str,
    cost_bps: float = 0.0010,
    rebalance_freq: Literal["D", "W", "M"] = "M",
) -> pd.DataFrame:
    """
    Run a vectorized backtest based on a signal column.

    Args:
        df: DataFrame with date index, 'Close', 'Daily_Return' and the signal column.
        signal_col: Name of column with 1 (Long), 0 (Cash), -1 (Short).
        cost_bps: Cost per trade (e.g., 0.0010 for 10bps).
        rebalance_freq: 'D' for daily, 'W' for weekly, 'M' for monthly.

    Returns:
        DataFrame with backtest results including equity curves and drawdowns.
    """
    if df.empty or signal_col not in df.columns:
        logger.error(f"Invalid input: empty df or missing signal column '{signal_col}'")
        return pd.DataFrame()

    logger.info(
        f"Running backtest: signal={signal_col}, cost={cost_bps*10000:.0f}bps, freq={rebalance_freq}"
    )

    bt_df = df.copy()

    # 1. Signal Processing
    if rebalance_freq == "D":
        # Daily Rebalance: Position today is determined by Signal yesterday
        bt_df["Position"] = bt_df[signal_col].shift(1).fillna(0)

    elif rebalance_freq == "W":
        # Weekly Rebalance
        bt_df["Period"] = bt_df.index.to_period("W")
        weekly_signals = bt_df.groupby("Period")[signal_col].last()
        weekly_positions = weekly_signals.shift(1)
        bt_df["Position"] = bt_df["Period"].map(weekly_positions)
        bt_df["Position"] = bt_df["Position"].fillna(0)

    elif rebalance_freq == "M":
        # Monthly Rebalance
        bt_df["Period"] = bt_df.index.to_period("M")
        monthly_signals = bt_df.groupby("Period")[signal_col].last()
        monthly_positions = monthly_signals.shift(1)
        bt_df["Position"] = bt_df["Period"].map(monthly_positions)
        bt_df["Position"] = bt_df["Position"].fillna(0)
    else:
        logger.error(f"Invalid rebalance frequency: {rebalance_freq}")
        return pd.DataFrame()

    # 2. Strategy Returns
    bt_df["Strategy_Return"] = bt_df["Position"] * bt_df["Daily_Return"]

    # 3. Transaction Costs
    bt_df["Position_Change"] = bt_df["Position"].diff().abs().fillna(0)
    bt_df["Cost"] = bt_df["Position_Change"] * cost_bps
    bt_df["Strategy_Net_Return"] = bt_df["Strategy_Return"] - bt_df["Cost"]

    # 4. Equity Curves
    bt_df["Equity_Benchmark"] = (1 + bt_df["Daily_Return"]).cumprod()
    bt_df["Equity_Strategy"] = (1 + bt_df["Strategy_Net_Return"]).cumprod()

    # 5. Drawdown Curves
    bt_df["DD_Benchmark"] = (bt_df["Equity_Benchmark"] / bt_df["Equity_Benchmark"].cummax()) - 1
    bt_df["DD_Strategy"] = (bt_df["Equity_Strategy"] / bt_df["Equity_Strategy"].cummax()) - 1

    logger.info(
        f"Backtest complete: {len(bt_df)} days, "
        f"Final equity: {bt_df['Equity_Strategy'].iloc[-1]:.2f}"
    )

    return bt_df


def calculate_conditional_stats(
    df: pd.DataFrame, strategy_col: str, regime_col: str
) -> pd.DataFrame:
    """
    Calculate performance stats conditioned on a regime column.

    Args:
        df: DataFrame with strategy returns and regime column.
        strategy_col: Column name of strategy returns.
        regime_col: Column name of regime classification.

    Returns:
        DataFrame with metrics per regime.
    """
    if df.empty or regime_col not in df.columns:
        logger.warning(f"Invalid input for conditional stats: missing '{regime_col}'")
        return pd.DataFrame()

    regimes = df[regime_col].unique()
    results = []

    for reg in regimes:
        subset = df[df[regime_col] == reg][strategy_col]

        if subset.empty:
            continue

        avg_ret = subset.mean() * TRADING_DAYS_PER_YEAR
        vol = subset.std() * (TRADING_DAYS_PER_YEAR**0.5)
        sharpe = avg_ret / vol if vol != 0 else 0
        win_rate = (subset > 0).mean()

        results.append(
            {
                "Regime": reg,
                "Ann_Return": avg_ret,
                "Volatility": vol,
                "Sharpe": sharpe,
                "WinRate": win_rate,
                "Count": len(subset),
            }
        )

    logger.debug(f"Conditional stats calculated for {len(results)} regimes")
    return pd.DataFrame(results).set_index("Regime")


def walk_forward_backtest(
    df: pd.DataFrame,
    signal_col: str,
    train_months: int = 24,
    test_months: int = 6,
    cost_bps: float = 0.0010,
    rebalance_freq: Literal["D", "W", "M"] = "M",
) -> Dict[str, Any]:
    """
    Perform walk-forward validation with rolling training windows.

    This method splits the data into overlapping train/test periods,
    evaluates the strategy on each out-of-sample test period, and
    aggregates the results.

    Args:
        df: DataFrame with date index, 'Close', 'Daily_Return', and signal column.
        signal_col: Name of column with 1 (Long), 0 (Cash), -1 (Short).
        train_months: Number of months for training window.
        test_months: Number of months for test window.
        cost_bps: Transaction cost in basis points.
        rebalance_freq: Rebalancing frequency.

    Returns:
        Dictionary containing:
        - 'summary': Aggregated performance metrics
        - 'periods': List of per-period results
        - 'oos_returns': Concatenated out-of-sample returns
    """
    if df.empty or signal_col not in df.columns:
        logger.error("Invalid input for walk-forward backtest")
        return {}

    logger.info(f"Walk-forward validation: train={train_months}m, test={test_months}m")

    # Convert to monthly periods for slicing
    df = df.copy()
    df["YearMonth"] = df.index.to_period("M")
    unique_months = df["YearMonth"].unique()

    total_months = len(unique_months)
    min_required = train_months + test_months

    if total_months < min_required:
        logger.warning(f"Insufficient data: {total_months} months < {min_required} required")
        return {}

    periods_results: List[Dict[str, Any]] = []
    all_oos_returns: List[pd.Series] = []

    # Walk forward through the data
    start_idx = 0
    while start_idx + min_required <= total_months:
        # Define train and test periods
        train_end_idx = start_idx + train_months
        test_end_idx = train_end_idx + test_months

        train_months_range = unique_months[start_idx:train_end_idx]
        test_months_range = unique_months[train_end_idx:test_end_idx]

        # Filter data
        test_mask = df["YearMonth"].isin(test_months_range)

        test_df = df[test_mask].copy()

        if len(test_df) == 0:
            break

        # Run backtest on test period only (signal already generated)
        bt_results = run_backtest(
            test_df, signal_col, cost_bps=cost_bps, rebalance_freq=rebalance_freq
        )

        if bt_results.empty:
            start_idx += test_months
            continue

        # Calculate metrics for this period
        period_metrics = calculate_perf_metrics(bt_results["Equity_Strategy"])

        periods_results.append(
            {
                "train_start": str(train_months_range[0]),
                "train_end": str(train_months_range[-1]),
                "test_start": str(test_months_range[0]),
                "test_end": str(test_months_range[-1]),
                "metrics": period_metrics,
            }
        )

        all_oos_returns.append(bt_results["Strategy_Net_Return"])

        # Slide forward by test_months
        start_idx += test_months

    if not periods_results:
        logger.warning("No valid walk-forward periods")
        return {}

    # Aggregate out-of-sample returns
    oos_returns = pd.concat(all_oos_returns)
    oos_equity = (1 + oos_returns).cumprod()

    # Calculate aggregate metrics
    aggregate_metrics = calculate_perf_metrics(oos_equity, include_bootstrap_ci=True)

    logger.info(
        f"Walk-forward complete: {len(periods_results)} periods, "
        f"OOS Sharpe={aggregate_metrics.get('Sharpe', 0):.2f}"
    )

    return {
        "summary": aggregate_metrics,
        "periods": periods_results,
        "oos_returns": oos_returns,
        "n_periods": len(periods_results),
    }
