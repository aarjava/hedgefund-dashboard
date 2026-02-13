# Research Note: Volatility Regimes and Trend Performance in US Equities

**Date**: February 2026
**Author**: Aarjav Ametha

## 1. Research Question
"How sensitive is trend-following performance to volatility regimes in US equities?"

Specifically, we investigate whether a simple Moving Average Trend strategy generates "Crisis Alpha" (positive returns during market stress) or suffers from whipsaws during high-volatility sideways markets.

## 2. Methodology

### 2.1 Universe
- **Primary Asset**: SPY (S&P 500 ETF).
- **Secondary Assets for Robustness**: QQQ (Nasdaq 100), IWM (Russell 2000).

### 2.2 Signal Definition
- **Trend Signal**: Long when Price > 50-day SMA, else Cash.
- **Regime Definition**: Annualized 21-day volatility.
    - **High Volatility**: > 75th percentile (expanding-window OOS quantiles)
    - **Low Volatility**: < 25th percentile (expanding-window OOS quantiles)
    - **Normal**: Between the two thresholds.

### 2.3 Backtest Methodology
- **Rebalancing**: Monthly
- **Transaction Costs**: 10 bps per trade
- **Period**: 1993-01-29 to 2026-02-09 (max available)

## 3. Empirical Results (SPY)

### 3.1 Unconditional Performance
| Metric | Trend Strategy | Buy & Hold |
| :--- | :--- | :--- |
| **CAGR** | 3.90% | 8.75% |
| **Sharpe** | 0.33 | 0.47 |
| **Max DD** | -36.46% | -56.47% |

### 3.2 Conditional Performance (by Regime, OOS)
| Regime | Avg Vol (ann.) | Strategy Sharpe | Benchmark Sharpe | Strategy Win Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | 7.96% | 1.55 | 1.74 | 50.65% |
| **Normal** | 13.60% | 0.12 | 0.44 | 35.56% |
| **High** | 26.00% | 0.21 | 0.45 | 21.34% |

**Observation**: During High Volatility regimes, the Trend Strategy underperformed the benchmark by 9.55 percentage points in annualized return. The strongest relative results occur in Low Volatility regimes.

### 3.3 Statistical Significance
- **High vs Normal Sharpe diff (bootstrapped)**: 0.09 (p=0.808)
- **Regime Sensitivity**: Sharpe diff (High-Normal) = 0.09, CAGR diff = 1.35%

### 3.4 Regime Transition Effects (Strategy)
| Transition | Sharpe | CAGR | Win Rate | Count |
| :--- | :--- | :--- | :--- | :--- |
| Normal→Normal | 0.48 | 4.78% | 36.00% | 3547 |
| High→High | 0.42 | 5.08% | 21.39% | 2375 |
| Low→Low | 1.46 | 12.00% | 50.89% | 1853 |
| Low→Normal | -5.09 | -66.73% | 33.33% | 135 |
| Normal→Low | 3.15 | 21.48% | 47.41% | 135 |
| High→Normal | 1.84 | 13.17% | 22.34% | 94 |

### 3.5 Walk-Forward OOS Summary
- **OOS CAGR**: 3.61%, **OOS Sharpe**: 0.33, **Max DD**: -33.22%, **Periods**: 61

### 3.6 SMA Robustness Sweep (Regime Sharpe)
| SMA | Normal Sharpe | High Sharpe | Low Sharpe |
| :--- | :--- | :--- | :--- |
| 20 | 0.35 | 0.00 | 1.28 |
| 50 | 0.28 | 0.06 | 1.57 |
| 100 | 0.22 | 0.03 | 1.88 |
| 150 | 0.25 | 0.17 | 1.83 |
| 200 | 0.57 | 0.09 | 1.80 |

### 3.7 Robustness Across Assets (Unconditional)
| Asset | Trend CAGR | Trend Sharpe | Buy&Hold CAGR | Buy&Hold Sharpe |
| :--- | :--- | :--- | :--- | :--- |
| QQQ | 7.40% | 0.43 | 9.52% | 0.35 |
| IWM | 3.35% | 0.21 | 6.78% | 0.28 |

## 4. Limitations & Mitigations

| Limitation | Status | Mitigation |
| :--- | :--- | :--- |
| **Hindsight Bias in Quartiles** | ✅ Addressed | Regime thresholds use expanding-window quantiles (OOS). |
| **SMA Lag** | ⚠️ Inherent | SMA is lagging; V-shaped recoveries can be missed. |
| **Transaction Costs** | ✅ Configurable | Backtest includes 10 bps per trade. |
| **Survivorship Bias** | ⚠️ Possible | ETF data may exclude delisted constituents. |

## 5. Conclusion
A 50-day trend-following system shows measurable sensitivity to volatility regimes. In this sample, performance is strongest in Low Volatility regimes and weaker in Normal and High regimes relative to buy-and-hold, while still reducing max drawdown. Results are broadly consistent across SPY, QQQ, and IWM with varying magnitudes.
