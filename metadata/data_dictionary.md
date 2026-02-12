# Data Dictionary

## `data/tables/unconditional_performance.csv`
- Index: `CAGR`, `Sharpe`, `MaxDD`, `Vol`, `Sortino`, `Calmar`, `WinRate`, `Sharpe_CI_Lower`, `Sharpe_CI_Upper`
- Columns: `Trend_Strategy`, `Buy_and_Hold`

## `data/tables/conditional_performance_by_regime.csv`
- Index: `Regime` (`Low`, `Normal`, `High`)
- Columns: `Avg_Vol_Ann`, `Strategy_Sharpe`, `Benchmark_Sharpe`, `Strategy_CAGR`, `Benchmark_CAGR`, `Strategy_WinRate`, `Count`

## `data/tables/regime_frequency.csv`
- Index: `Vol_Regime`
- Columns: `Frequency`

## `data/tables/regime_transition_matrix.csv`
- Index: source regime (`Low`, `Normal`, `High`)
- Columns: destination regime (`Low`, `Normal`, `High`)
- Values: row-normalized transition probabilities

## `data/tables/regime_transition_performance.csv`
- Index: transition label (example: `Normal->High`)
- Columns: `Mean`, `Sharpe`, `WinRate`, `CAGR`, `Count`

## `data/tables/walk_forward_oos_summary.csv`
- Columns: `OOS_CAGR`, `OOS_Sharpe`, `OOS_MaxDD`, `OOS_Vol`, `OOS_WinRate`, `Periods`

## `data/tables/sma_sweep_sharpe_by_regime.csv`
- Index: `SMA` window length
- Columns: regime-specific Sharpe values (`Low`, `Normal`, `High`, optional `Unknown`)

## `data/tables/robustness_assets_unconditional.csv`
- Index: `Asset` (`QQQ`, `IWM`)
- Columns: `Trend_CAGR`, `Trend_Sharpe`, `Trend_MaxDD`, `BuyHold_CAGR`, `BuyHold_Sharpe`, `BuyHold_MaxDD`

## `data/tables/regime_sensitivity_bootstrap.csv`
- Columns: `Metric`, `Value`
- Includes high-minus-normal Sharpe/CAGR diffs and bootstrap p-value

## `data/time_series/*_backtest_daily.csv`
- Row index: `Date`
- Shared columns:
- `Close`, `Daily_Return`, `Vol_21d`, `Vol_Regime`, `SMA_50`, `Signal_Trend`
- `Position`, `Strategy_Return`, `Cost`, `Strategy_Net_Return`
- `Equity_Benchmark`, `Equity_Strategy`, `DD_Benchmark`, `DD_Strategy`

## `metadata/summary_metrics.json`
- Backtest assumptions, sample period, primary results, and regime setup.

## `metadata/figure_index.csv`
- Mapping of figure file names to recommended captions.
