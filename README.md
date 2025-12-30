# Quantitative Momentum Dashboard

## Research Question
**"Does simple price-based momentum (trend) generate statistically meaningful excess returns compared to buy-and-hold?"**

This dashboard provides a robust, interactive tool to empirically test this question across various asset classes (Equities, Crypto, Commods) using flexible historical windows and configurable parameters.

## Features

### 1. Advanced Analytics
- **Dynamic Signal Generation**: 
  - **Trend**: Adjustable Simple Moving Average (SMA) lookback.
  - **Momentum**: Customizable "12-1" month momentum window (Jegadeesh & Titman, 1993).
  - **Mean Reversion**: RSI (Relative Strength Index) monitoring.
- **Risk-Adjusted Performance**: Automatic calculation of Sharpe Ratio, Sortino Ratio, Calmar Ratio, and Max Drawdown.

### 2. Backtesting Engine
- **Event-Driven Simulation** (Vectorized):
  - Supports Monthly or Daily rebalancing logic.
  - Configurable Transaction Costs (bps).
  - Long/Cash or Long/Short regimes.
- **Visualizations**: Interactive Equity Curves, Drawdown Charts, and Signal Overlays.

### 3. Modular Architecture
The codebase is structured for scalability:
- `src/modules/data_model.py`: Robust data fetching with caching.
- `src/modules/signals.py`: Library of technical indicators.
- `src/modules/backtester.py`: Vectorized backtesting logic and metric calculations.

## Getting Started

### Prerequisites
- Python 3.8+
- [Poetry](https://python-poetry.org/) or virtualenv recommended.

### Installation
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: dependencies include streamlit, yfinance, pandas, plotly, numpy)*

2.  Run the tests to verify integrity:
    ```bash
    python -m unittest discover tests
    ```

3.  Launch the dashboard:
    ```bash
    streamlit run src/dashboard.py
    ```

## User Guide
1.  **Sidebar Config**: Select an asset (e.g., SPY, BTC-USD) and adjust the "Lookback Window" to see how sensitivity changes performance.
2.  **Overview Tab**: View the current price action relative to the Trend SMA.
3.  **Signals Tab**: Check the current state of three distinct strategies (Trend, Momentum, Mean Reversion).
4.  **Backtest Tab**: Compare the selected strategy against a Buy & Hold benchmark. Adjust transaction costs to see the impact of friction.
5.  **Report Tab**: Download the full analysis data as a CSV for offline research.

## References
- **Jegadeesh, N., & Titman, S. (1993)**. Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. *The Journal of Finance*.
- **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012)**. Time series momentum. *Journal of Financial Economics*.

---
*Built with ❤️ by the Quantitative Research Team.*
