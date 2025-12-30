# HedgeFund Dashboard 📈
## Quantitative Momentum & Regime Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📊 Overview

The **HedgeFund Dashboard** is a professional-grade quantitative research tool designed to empirically test the efficacy of price-based momentum strategies across varying volatility regimes.

Built for seamless interactivity, this application empowers researchers to move beyond simple "Buy & Hold" comparisons. It decomposes returns to isolate specific market conditions where trend-following strategies generate statistically significant excess returns versus where they suffer from "whipsaw" decay.

## 🚀 Key Capabilities

### 1. Dynamic Signal Generation
*   **Trend Following**: Adjustable Simple Moving Average (SMA) logic (e.g., Price > 200d SMA).
*   **Momentum**: customizable lookback windows (e.g., classical 12-1 month momentum).
*   **Mean Reversion**: Integrated RSI and volatility oscillators.

### 2. Regime-Conditional Backtesting
*   **Market Segmentation**: Automatically detects Low, Normal, and High volatility regimes using rolling realized volatility quantiles.
*   **Conditional Performance**: Calculates Sharpe, Sortino, and Win Rate *per regime*, answering the critical question: *"Does this strategy survive high-volatility crashes?"*

### 3. Professional Backtest Engine
*   **Vectorized Simulation**: High-speed backtesting across daily or monthly rebalancing frequencies.
*   **Friction Modeling**: Configurable transaction costs (basis points) to simulate real-world execution drag.
*   **Long/Short Logic**: Toggle between Long-Only and Long/Short mandates.

### 4. Interactive Visualization
*   **Equity Curves**: Compare Strategy vs. Benchmark wealth accumulation in log or linear scale.
*   **Drawdown Analysis**: Visualize underwater periods to assess tail risk.
*   **Signal Overlays**: Inspect specific trade entry/exit points directly on the price chart.

## 🛠 Tech Stack

*   **Core**: Python 3.8+
*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Data Analysis**: Pandas, NumPy
*   **Data Feed**: [yfinance](https://pypi.org/project/yfinance/)
*   **Visualization**: Plotly Express & Graph Models

## ⚡ Getting Started

### Prerequisites
*   Python 3.8 or higher
*   pip (Python Package Installer)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/hedgefund-dashboard.git
    cd hedgefund-dashboard
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run src/dashboard.py
    ```

## 📖 User Guide

| Tab | Functionality |
| :--- | :--- |
| **📈 Overview** | Real-time snapshot of the asset's price, current trend status, and volatility regime. |
| **🌪️ Regime Analysis** | Deep dive into the distribution of market volatility. Visualize how often the market is in "High Stress" mode. |
| **🧪 Backtest Engine** | The core research lab. Compare your configured strategy against the benchmark. Analyze conditional statistics. |
| **📄 Report** | Summary of findings and raw data export for further analysis in Jupyter/Excel. |

## 🔬 Methodology

This project draws inspiration from seminal literature in quantitative finance, specifically investigating the **"Smile Curve"** performance of trend strategies:

*   **Jegadeesh, N., & Titman, S. (1993)**. Returns to Buying Winners and Selling Losers.
*   **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012)**. Time series momentum.

The underlying hypothesis is that price trends are persistent (autocorrelated) in normal markets but break down during mean-reverting (high volatility) shocks. This tool allows you to quantify that breakdown.

## ⚖️ Disclaimer

*This software is for educational and research purposes only. It does not constitute financial advice, investment recommendations, or trading signals. Past performance is not indicative of future results.*

---
*Maintained by Aarjava Metha.*
