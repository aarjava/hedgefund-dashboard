# Research Note: Volatility Regimes and Trend Performance in US Equities

**Date**: December 2025
**Author**: Quantitative Research Team

## 1. Research Question
"How sensitive is trend-following performance to volatility regimes in US equities?"

Specifically, we investigate whether a simple Moving Average Trend strategy generates "Crisis Alpha" (positive returns during market stress) or suffers from whipsaws during high-volatility sideways markets.

## 2. Methodology

### 2.1 Universe
- **Primary Asset**: SPY (S&P 500 ETF).
- **Secondary Assets for Robustness**: QQQ (Nasdaq 100), IWM (Russell 2000).

### 2.2 Signal Definition
- **Trend Signal**: Long when Price > 50-day SMA, else Cash (or Short).
- **Regime Definition**:
    - **High Volatility**: Annualized 21-day volatility > 75th percentile of history.
    - **Low Volatility**: Annualized 21-day volatility < 25th percentile of history.
    - **Normal**: Between 25th and 75th percentile.

### 2.3 Backtest Methodology
- **Rebalancing**: Monthly.
- **Transaction Costs**: 10 basis points per trade.
- **Period**: Max available (typically 20-30 years for ETFs).

## 3. Empirical Results

*Note: Run the Dashboard to populate exact figures.*

### 3.1 Unconditional Performance
| Metric | Trend Strategy | Buy & Hold |
| :--- | :--- | :--- |
| **CAGR** | [Value]% | [Value]% |
| **Sharpe** | [Value] | [Value] |
| **Max DD** | [Value]% | [Value]% |

### 3.2 Conditional Performance (by Regime)
The hypothesis is that Trend Following performs best during "Normal" vol (sustained trends) and potentially protects during "High" vol (down trends), but suffers during transitions.

| Regime | Average Vol | Strategy Sharpe | Benchmark Sharpe | Win Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Low Vol** | < 10% | [Value] | [Value] | [Value] |
| **Normal** | 10-20% | [Value] | [Value] | [Value] |
| **High Vol** | > 20% | [Value] | [Value] | [Value] |

**Observation**: During High Volatility regimes (which correlate with market crashes), the Trend Strategy [outperformed/underperformed] the benchmark by [X]%.

## 4. Limitations
1.  **Hindsight Bias in Quartiles**: The regime thresholds (25th/75th percentiles) are calculated over the full sample. In a live setting, one would use a rolling window.
2.  **Lag**: The 50-day SMA is a lagging indicator. In "V-shaped" recoveries (high vol), the strategy is slow to re-enter, potentially missing the rebound.

## 5. Conclusion
A 50-day trend-following system demonstrates [sensitivity/robustness] to volatility regimes. It acts as effective insurance during prolonged high-volatility corrections but can underperform in choppy, mean-reverting high-volatility environments.
