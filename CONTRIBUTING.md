# Contributing to HedgeFund Dashboard

Thank you for your interest in contributing to the HedgeFund Dashboard! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hedgefund-dashboard.git
   cd hedgefund-dashboard
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify Setup**
   ```bash
   pytest tests/ -v
   ```

## 📝 Development Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking

Run these before committing:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports
```

### Type Hints

All new functions should include type hints:

```python
def calculate_signal(
    df: pd.DataFrame,
    window: int = 50,
    threshold: float = 0.0
) -> pd.Series:
    """Calculate trading signal.
    
    Args:
        df: DataFrame with price data.
        window: Lookback period.
        threshold: Signal threshold.
        
    Returns:
        Series with signal values.
    """
    ...
```

### Logging

Use the module logger for diagnostic output:

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Detailed calculation info")
logger.info("Operation completed successfully")
logger.warning("Potential issue detected")
logger.error("Operation failed")
```

### Testing

All new features should include tests:

1. **Unit tests** for individual functions
2. **Edge case tests** for boundary conditions
3. **Integration tests** for feature workflows

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_backtester.py::TestBacktester::test_run_backtest_daily -v
```

## 🔄 Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, documented code
- Add/update tests as needed
- Update documentation if applicable

### 3. Run Quality Checks

```bash
# Format
black src/ tests/

# Lint
ruff check src/ tests/ --fix

# Test
pytest tests/ -v

# (Optional) Type check
mypy src/ --ignore-missing-imports
```

### 4. Commit with Clear Messages

```bash
git commit -m "feat: add volatility breakout signal"
git commit -m "fix: handle empty dataframe in backtester"
git commit -m "docs: update README with new features"
git commit -m "test: add tests for walk-forward validation"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 📋 Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|:---|:---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `test` | Adding or updating tests |
| `refactor` | Code refactoring (no feature/fix) |
| `perf` | Performance improvement |
| `style` | Code style (formatting, etc.) |
| `chore` | Maintenance tasks |

## 🏗️ Architecture

### Module Overview

| Module | Purpose |
|:---|:---|
| `config.py` | Centralized constants and defaults |
| `data_model.py` | Data fetching with Streamlit caching |
| `signals.py` | Core technical indicators |
| `signals_advanced.py` | Advanced trading strategies |
| `backtester.py` | Simulation engine with metrics |
| `dashboard.py` | Streamlit UI |

### Adding a New Signal

1. Add the function to `signals.py` or `signals_advanced.py`
2. Include docstring with Args, Returns, and example
3. Add logging at appropriate levels
4. Write tests in `test_signals.py` or `test_signals_advanced.py`
5. (Optional) Add to dashboard UI

Example:

```python
# In signals_advanced.py
def generate_my_signal(df: pd.DataFrame, param: int = 10) -> pd.Series:
    """
    Generate my custom signal.
    
    Args:
        df: DataFrame with Close prices.
        param: Signal parameter.
        
    Returns:
        Series with signal values: 1 (long), -1 (short), 0 (neutral).
    """
    logger.info(f"Generating my signal with param={param}")
    # Implementation...
    return signal
```

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to Reproduce**: How to trigger the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, package versions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🙏
