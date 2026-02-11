"""
Centralized configuration for the HedgeFund Dashboard.
Avoids magic numbers scattered throughout the codebase.
"""

# === Trading Constants ===
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21

# === Default Signal Parameters ===
DEFAULT_SMA_WINDOW = 50
DEFAULT_MOMENTUM_WINDOW = 12  # months
DEFAULT_VOLATILITY_WINDOW = 21  # days
DEFAULT_RSI_WINDOW = 14

# === Default Regime Thresholds ===
DEFAULT_VOL_QUANTILE_HIGH = 0.75
DEFAULT_VOL_QUANTILE_LOW = 0.25

# === Backtest Defaults ===
DEFAULT_COST_BPS = 10  # In basis points (10 bps = 0.10%)
DEFAULT_REBALANCE_FREQ = "M"  # Monthly

# === Caching ===
CACHE_TTL_SECONDS = 3600 * 24  # 24 hours

# === Asset Universe ===
PRESET_UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLK", "XLE", "BTC-USD", "ETH-USD"]

# === Minimum Data Requirements ===
MIN_DATA_POINTS = 50
MIN_PERIODS_FOR_EXPANDING = 60  # For out-of-sample regime detection
