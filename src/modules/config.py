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

# === Regime Lab Defaults ===
DEFAULT_BOOTSTRAP_ITER = 500
DEFAULT_SMA_SWEEP = [20, 50, 100, 150, 200]

# === Backtest Defaults ===
DEFAULT_COST_BPS = 10  # In basis points (10 bps = 0.10%)
DEFAULT_REBALANCE_FREQ = "M"  # Monthly

# === Portfolio Defaults ===
DEFAULT_BENCHMARK = "SPY"
DEFAULT_PORTFOLIO_VALUE = 1_000_000
DEFAULT_ADV_PCT = 0.10  # 10% of ADV for liquidation estimate

# === Factor & Macro Proxies (Yahoo Finance tickers) ===
FACTOR_PROXIES = {
    "Market": "SPY",
    "Momentum": "MTUM",
    "Value": "IWD",
    "Growth": "IWF",
    "Quality": "QUAL",
    "LowVol": "USMV",
    "Size": "IWM",
}

MACRO_PROXIES = {
    "Rates": "TLT",
    "USD": "UUP",
    "Vol": "^VIX",
    "Gold": "GLD",
    "Oil": "USO",
}

# === Caching ===
CACHE_TTL_SECONDS = 3600 * 24  # 24 hours

# === Asset Universe ===
PRESET_UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLK", "XLE", "BTC-USD", "ETH-USD"]

# === Minimum Data Requirements ===
MIN_DATA_POINTS = 50
MIN_PERIODS_FOR_EXPANDING = 60  # For out-of-sample regime detection
