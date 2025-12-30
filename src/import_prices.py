from modules import data_model, signals

def analyze_stock(ticker_symbol, period="1y"):
    """
    Demonstrates using the new modular architecture.
    """
    print(f"\n--- Analyzing {ticker_symbol} ---")
    
    # 1. Use the data model
    df = data_model.fetch_stock_data(ticker_symbol, period=period)
    
    if df.empty:
        print(f"No data found for {ticker_symbol}")
        return None

    # 2. Use the signals module
    df = signals.add_technical_indicators(df)
    
    latest = df.iloc[-1]
    
    print(f"Current Price: ${latest['Close']:.2f}")
    print(f"50-Day SMA:    ${latest['SMA_50']:.2f}")
    print(f"Momentum:      {latest['Momentum_12M_1M']:.2%}")
    print(f"RSI (14):      {latest['RSI_14']:.2f}")
    
    if latest['Close'] > latest['SMA_50']:
        print("Trend: BULLISH 🐂")
    else:
        print("Trend: BEARISH 🐻")
        
    return df

if __name__ == "__main__":
    # Add your favorite stocks here!
    portfolio = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    for ticker in portfolio:
        analyze_stock(ticker)