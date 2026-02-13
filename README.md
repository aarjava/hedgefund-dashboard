# HedgeFund Dashboard 📈
## Quantitative Momentum & Regime Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-51%20Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-85%25-yellow)

## 📊 Overview

The **HedgeFund Dashboard** is a professional-grade quantitative research tool designed to empirically test the efficacy of price-based momentum strategies across varying volatility regimes.

Built for seamless interactivity, this application empowers researchers to move beyond simple "Buy & Hold" comparisons. It decomposes returns to isolate specific market conditions where trend-following strategies generate statistically significant excess returns versus where they suffer from "whipsaw" decay.

## 🚀 Key Capabilities

### 1. Dynamic Signal Generation
*   **Trend Following**: Adjustable Simple Moving Average (SMA) logic (e.g., Price > 200d SMA).
*   **Momentum**: Customizable lookback windows (e.g., classical 12-1 month momentum).
*   **Mean Reversion**: RSI-based signals with Bollinger Band confirmation.
*   **Volatility Breakout**: Capture momentum during volatility expansion.
*   **Dual Momentum**: Combined absolute + relative momentum (Antonacci-style).
*   **Composite Signals**: Weighted combination of multiple strategies.

### 2. Regime-Conditional Backtesting
*   **Market Segmentation**: Automatically detects Low, Normal, and High volatility regimes using rolling realized volatility quantiles.
*   **Out-of-Sample Mode**: Toggle to use expanding-window quantiles for rigorous backtesting without look-ahead bias.
*   **Conditional Performance**: Calculates Sharpe, Sortino, and Win Rate *per regime*, answering the critical question: *"Does this strategy survive high-volatility crashes?"*

### 3. Professional Backtest Engine
*   **Vectorized Simulation**: High-speed backtesting across daily, weekly, or monthly rebalancing frequencies.
*   **Friction Modeling**: Configurable transaction costs (basis points) to simulate real-world execution drag.
*   **Long/Short Logic**: Toggle between Long-Only and Long/Short mandates.
*   **Walk-Forward Validation**: Rolling out-of-sample testing for robust performance evaluation.
*   **Bootstrap Confidence Intervals**: Statistical significance testing for Sharpe ratios.

### 4. Interactive Visualization
*   **Equity Curves**: Compare Strategy vs. Benchmark wealth accumulation.
*   **Drawdown Analysis**: Visualize underwater periods with duration metrics.
*   **Signal Overlays**: Inspect specific trade entry/exit points directly on the price chart.
*   **8+ Performance Metrics**: CAGR, Sharpe (with 95% CI), Sortino, Calmar, Max DD, DD Duration, Win Rate.

### 5. Portfolio Analytics Suite
*   **Multi-Asset Portfolio Mode**: Upload a CSV of tickers/weights or build from presets.
*   **Risk Posture Score**: Volatility, drawdown, beta, and liquidity combined into a single score.
*   **Factor Attribution**: Rolling OLS betas on ETF factor proxies with contributions.
*   **Scenario Engine**: Macro shock sliders (rates, USD, oil, gold, vol) with impact estimates.
*   **Alerts & Anomalies**: Rule-based alerts and z-score anomaly detection.
*   **Client-Ready Reports**: Export deterministic Markdown/HTML summaries and CSV data.

### 6. Regime Performance Lab (New)
*   **Transition Matrix**: Regime-to-regime transition probabilities with heatmap.
*   **Transition Impact**: Performance following regime switches (e.g., Normal→High).
*   **Regime Sensitivity Score**: Sharpe/CAGR differences between High and Normal regimes.
*   **Bootstrap Significance**: P-value for regime performance differences.
*   **Robustness Sweep**: SMA window sweep to detect parameter fragility.
*   **Portfolio Cross-Section**: Rank assets by regime sensitivity (resilient vs fragile).

## 🛠 Tech Stack

| Category | Technologies |
|:---|:---|
| **Core** | Python 3.8+ |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Data Analysis** | Pandas, NumPy |
| **Data Feed** | [yfinance](https://pypi.org/project/yfinance/) |
| **Visualization** | Plotly Express & Graph Objects |
| **Testing** | pytest, pytest-cov |
| **Linting** | Ruff, Black, MyPy |
| **CI/CD** | GitHub Actions |

## ⚡ Getting Started

### Prerequisites
*   Python 3.8 or higher
*   pip (Python Package Installer)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/aarjavametha/hedgefund-dashboard.git
    cd hedgefund-dashboard
    ```

2.  **Install Dependencies**
    ```bash
    # Using pip
    pip install -r requirements.txt
    
    # Or using modern packaging (recommended)
    pip install -e ".[dev]"
    ```

3.  **Run the Application**
    ```bash
    streamlit run src/dashboard.py
    ```

4.  **Run Tests**
    ```bash
    pytest tests/ -v
    ```

## 📖 User Guide

| Tab | Functionality |
| :--- | :--- |
| **📈 Overview** | Real-time snapshot of the asset's price, current trend status, and volatility regime. |
| **🌪️ Regime Analysis** | Deep dive into the distribution of market volatility. Visualize how often the market is in "High Stress" mode. |
| **🧪 Backtest Engine** | The core research lab. Compare your configured strategy against the benchmark. Analyze conditional statistics. Includes walk-forward validation. |
| **📄 Report** | Summary of findings and raw data export for further analysis in Jupyter/Excel. |

**Portfolio Mode Tabs** (available when Mode = Portfolio):
* **📈 Overview**: Portfolio equity curve, weights, quick query panel.
* **🧪 Regime Lab**: Cross-section regime sensitivity and robustness checks.
* **🛡️ Risk & Liquidity**: Risk posture, drawdowns, correlation matrix, liquidity stress.
* **🧬 Attribution**: Rolling factor betas, factor contributions, alpha series.
* **🧪 Scenario**: Macro shock sliders and portfolio impact.
* **📡 Signals Health**: Signal decay and rolling IC on the benchmark.
* **🚨 Alerts**: Threshold alerts and anomaly detection.
* **📄 Report**: Client-ready report exports.

## 🔬 Methodology

This project draws inspiration from seminal literature in quantitative finance, specifically investigating the **"Smile Curve"** performance of trend strategies:

*   **Jegadeesh, N., & Titman, S. (1993)**. Returns to Buying Winners and Selling Losers.
*   **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012)**. Time series momentum.
*   **Antonacci, G. (2014)**. Dual Momentum Investing.

The underlying hypothesis is that price trends are persistent (autocorrelated) in normal markets but break down during mean-reverting (high volatility) shocks. This tool allows you to quantify that breakdown.

## 📁 Project Structure

```
hedgefund-dashboard/
├── src/
│   ├── dashboard.py          # Main Streamlit application
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── config.py         # Centralized configuration
│   │   ├── data_model.py     # Data fetching with caching
│   │   ├── signals.py        # Core technical indicators
│   │   ├── signals_advanced.py  # Advanced signal strategies
│   │   └── backtester.py     # Backtest engine with metrics
├── tests/
│   ├── test_backtester.py
│   ├── test_signals.py
│   ├── test_signals_advanced.py
│   └── test_data_model.py
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD
├── requirements.txt
├── pyproject.toml            # Modern Python packaging
├── research_note.md
└── README.md
```

## 🧪 Testing

The project maintains a comprehensive test suite with 44+ tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_backtester.py -v
```

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚖️ Disclaimer

*This software is for educational and research purposes only. It does not constitute financial advice, investment recommendations, or trading signals. Past performance is not indicative of future results.*

---
*Maintained by Aarjav Ametha.*
