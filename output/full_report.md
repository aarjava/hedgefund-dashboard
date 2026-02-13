# HedgeFund Dashboard Report

## Summary
- **Report Date**: 2026-02-09
- **Data Range (SPY)**: 1993-01-29 to 2026-02-09
- **Strategy**: Trend: Price > 50d SMA (Long/Cash)
- **Regime Model**: 21d vol, OOS expanding quantiles (Low<25%, High>75%)
- **Rebalance**: Monthly
- **Transaction Cost**: 10 bps per trade
- **Unconditional Trend Sharpe**: 0.33
- **Unconditional Buy&Hold Sharpe**: 0.47
- **High-Normal Sharpe Diff (boot)**: 0.09 (p=0.808)
- **Regime Sensitivity (Sharpe/CAGR)**: 0.09 / 1.35%

## Unconditional Performance (SPY)
| index | Trend Strategy | Buy & Hold |
| --- | --- | --- |
| CAGR | 0.0390 | 0.0875 |
| Sharpe | 0.3274 | 0.4684 |
| MaxDD | -0.3646 | -0.5647 |
| Vol | 0.1190 | 0.1868 |
| Sortino | 0.3399 | 0.5995 |
| Calmar | 0.1069 | 0.1550 |
| WinRate | 0.3481 | 0.5368 |
| Sharpe_CI_Lower | 0.0337 | nan |
| Sharpe_CI_Upper | 0.7751 | nan |

## Conditional Performance by Regime (SPY)
| Regime | Avg_Vol_Ann | Strategy_Sharpe | Benchmark_Sharpe | Strategy_CAGR | Benchmark_CAGR | Strategy_WinRate | Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Low | 0.0796 | 1.5479 | 1.7391 | 0.1262 | 0.1500 | 0.5065 | 1988 |
| Normal | 0.1360 | 0.1234 | 0.4435 | 0.0076 | 0.0538 | 0.3556 | 3777 |
| High | 0.2600 | 0.2147 | 0.4474 | 0.0210 | 0.0907 | 0.2134 | 2469 |

## Regime Frequency (SPY)
| Vol_Regime | Frequency |
| --- | --- |
| Low | 0.2405 |
| Normal | 0.4570 |
| High | 0.2987 |

## Regime Transition Matrix (SPY)
| Vol_Regime | Low | Normal | High |
| --- | --- | --- | --- |
| Low | 0.9321 | 0.0679 | 0.0000 |
| Normal | 0.0358 | 0.9394 | 0.0249 |
| High | 0.0000 | 0.0381 | 0.9619 |

## Regime Transition Performance (Strategy)
| Transition | Mean | Sharpe | WinRate | CAGR | Count |
| --- | --- | --- | --- | --- | --- |
| High→High | 0.0599 | 0.4170 | 0.2139 | 0.0508 | 2375 |
| High→Normal | 0.1260 | 1.8386 | 0.2234 | 0.1317 | 94 |
| Low→Low | 0.1165 | 1.4592 | 0.5089 | 0.1200 | 1853 |
| Low→Normal | -1.0759 | -5.0937 | 0.3333 | -0.6673 | 135 |
| Normal→High | -0.6746 | -2.8132 | 0.2021 | -0.5055 | 94 |
| Normal→Low | 0.1966 | 3.1458 | 0.4741 | 0.2148 | 135 |
| Normal→Normal | 0.0527 | 0.4787 | 0.3600 | 0.0478 | 3547 |

## Walk-Forward OOS Summary
| index | OOS_CAGR | OOS_Sharpe | OOS_MaxDD | OOS_Vol | OOS_WinRate | Periods |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.0361 | 0.3251 | -0.3322 | 0.1110 | 0.2985 | 61.0000 |

## SMA Sweep (Sharpe by Regime)
| SMA | High | Low | Normal | Unknown |
| --- | --- | --- | --- | --- |
| 20.0000 | 0.0018 | 1.2846 | 0.3531 | -3.4039 |
| 50.0000 | 0.0557 | 1.5689 | 0.2808 | -2.3014 |
| 100.0000 | 0.0314 | 1.8797 | 0.2244 | nan |
| 150.0000 | 0.1658 | 1.8350 | 0.2543 | nan |
| 200.0000 | 0.0876 | 1.8048 | 0.5672 | nan |

## Robustness Across Assets (Unconditional)
| Asset | Trend_CAGR | Trend_Sharpe | Trend_MaxDD | BuyHold_CAGR | BuyHold_Sharpe | BuyHold_MaxDD |
| --- | --- | --- | --- | --- | --- | --- |
| QQQ | 0.0740 | 0.4251 | -0.3941 | 0.0952 | 0.3536 | -0.8296 |
| IWM | 0.0335 | 0.2121 | -0.4208 | 0.0678 | 0.2831 | -0.5949 |

## Figures

**Figure 1. SPY Equity Curves (Log Scale)**

![Figure 1. SPY Equity Curves (Log Scale)](output/figures/fig_equity_curves.png)

**Figure 2. SPY Drawdown Curves**

![Figure 2. SPY Drawdown Curves](output/figures/fig_drawdowns.png)

**Figure 3. Regime Frequency (SPY, OOS)**

![Figure 3. Regime Frequency (SPY, OOS)](output/figures/fig_regime_frequency.png)

**Figure 4. Sharpe by Volatility Regime (SPY)**

![Figure 4. Sharpe by Volatility Regime (SPY)](output/figures/fig_conditional_sharpe.png)

**Figure 5. Regime Transition Matrix (SPY)**

![Figure 5. Regime Transition Matrix (SPY)](output/figures/fig_transition_matrix.png)

**Figure 6. SMA Window Robustness (Sharpe by Regime)**

![Figure 6. SMA Window Robustness (Sharpe by Regime)](output/figures/fig_sma_sweep.png)

**Figure 7. Robustness Across Assets (CAGR and Sharpe)**

![Figure 7. Robustness Across Assets (CAGR and Sharpe)](output/figures/fig_robustness_assets.png)
