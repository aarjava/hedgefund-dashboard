"""Tests for data model module."""

import unittest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modules import data_model


class TestFetchStockData(unittest.TestCase):
    """Tests for fetch_stock_data function."""

    @patch("src.modules.data_model.yf.Ticker")
    def test_successful_fetch(self, mock_ticker):
        """Test successful data fetch."""
        # Create mock data
        dates = pd.date_range("2020-01-01", periods=10)
        mock_df = pd.DataFrame(
            {
                "Open": [100] * 10,
                "High": [105] * 10,
                "Low": [95] * 10,
                "Close": [102] * 10,
                "Volume": [1000000] * 10,
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance

        # Clear cache to ensure fresh call
        data_model.fetch_stock_data.clear()

        result = data_model.fetch_stock_data("TEST", period="1y")

        self.assertFalse(result.empty)
        self.assertEqual(len(result), 10)
        self.assertIn("Close", result.columns)

    @patch("src.modules.data_model.yf.Ticker")
    def test_empty_data_handling(self, mock_ticker):
        """Test handling of empty data response."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        data_model.fetch_stock_data.clear()
        result = data_model.fetch_stock_data("INVALID", period="1y")

        self.assertTrue(result.empty)

    @patch("src.modules.data_model.yf.Ticker")
    def test_timezone_handling(self, mock_ticker):
        """Test that timezone is removed from index."""
        dates = pd.date_range("2020-01-01", periods=5, tz="America/New_York")
        mock_df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance

        data_model.fetch_stock_data.clear()
        result = data_model.fetch_stock_data("SPY", period="1y")

        # Timezone should be removed
        self.assertIsNone(result.index.tz)


class TestValidateTicker(unittest.TestCase):
    """Tests for validate_ticker function."""

    @patch("src.modules.data_model.yf.Ticker")
    def test_valid_ticker(self, mock_ticker):
        """Test validation of a valid ticker."""
        mock_info = {"regularMarketPrice": 150.0, "longName": "Apple Inc."}
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance

        result = data_model.validate_ticker("AAPL")

        self.assertTrue(result)

    @patch("src.modules.data_model.yf.Ticker")
    def test_invalid_ticker(self, mock_ticker):
        """Test validation of an invalid ticker."""
        mock_info = {"regularMarketPrice": None}
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance

        result = data_model.validate_ticker("INVALIDTICKER123")

        self.assertFalse(result)

    @patch("src.modules.data_model.yf.Ticker")
    def test_api_error_handling(self, mock_ticker):
        """Test handling of API errors."""
        mock_ticker.side_effect = Exception("API Error")

        result = data_model.validate_ticker("ERROR")

        self.assertFalse(result)


class TestGetTickerInfo(unittest.TestCase):
    """Tests for get_ticker_info function."""

    @patch("src.modules.data_model.yf.Ticker")
    def test_successful_info_fetch(self, mock_ticker):
        """Test successful ticker info fetch."""
        mock_info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "currency": "USD",
            "exchange": "NASDAQ",
        }
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance

        result = data_model.get_ticker_info("AAPL")

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Apple Inc.")
        self.assertEqual(result["sector"], "Technology")

    @patch("src.modules.data_model.yf.Ticker")
    def test_missing_info_fields(self, mock_ticker):
        """Test handling of missing info fields."""
        mock_info = {"longName": "Test Company"}  # Missing other fields
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance

        result = data_model.get_ticker_info("TEST")

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test Company")
        self.assertEqual(result["sector"], "N/A")  # Default value

    @patch("src.modules.data_model.yf.Ticker")
    def test_api_error_returns_none(self, mock_ticker):
        """Test that API errors return None."""
        mock_ticker.side_effect = Exception("API Error")

        result = data_model.get_ticker_info("ERROR")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
