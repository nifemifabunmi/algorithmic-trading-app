# main.py
# This script fetches the last year's stock data for Apple Inc. (AAPL)
# and plots the closing price using yfinance and matplotlib.
# Suppress FutureWarnings from yfinance 

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def download_data(ticker="AAPL", period="1y"):
        print(f"Downloading data for {ticker}...")
        df = yf.download(ticker, period=period)
        if isinstance(df.columns, pd.MultiIndex):
              df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        print(df.columns)
        return df

def add_moving_averages(data, short_window=20, long_window=50):
    close_col = [col for col in data.columns if col.startswith('Close')][0]
    data['MA20'] = data[close_col].rolling(window=short_window).mean()
    data['MA50'] = data[close_col].rolling(window=long_window).mean()
    data['Close'] = data[close_col]  # Standardize the name for use later
    return data


def generate_signals(data):
        data = data.copy()  # avoid SettingWithCopyWarning
        data['Signal'] = 0
        data.loc[data['MA20'] > data['MA50'], 'Signal'] = 1  # Buy
        data.loc[data['MA20'] < data['MA50'], 'Signal'] = -1  # Sell 
        data['Position'] = data['Signal'].diff()
        return data

def plot_moving_averages(data, ticker="AAPL"):
    plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['MA20'], label='20-day MA')
    plt.plot(data['MA50'], label='50-day MA')
    plt.title(f"{ticker} Close Price and Moving Averages")
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def backtest_strategy(data, initial_cash=10000):
    data = data.copy()
    data = data.sort_index()

    cash = initial_cash
    shares = 0
    position = 0
    portfolio_values = []

    for i, row in data.iterrows():
        signal = row['Position']

        # Skip NaNs or invalid data
        if pd.isna(signal):
            portfolio_values.append(cash + shares * row['Close'])
            continue

        if signal == 1 and position == 0:
            # Buy
            shares = cash // row['Close']
            cash -= shares * row['Close']
            position = 1

        elif signal == -1 and position == 1:
            # Sell
            cash += shares * row['Close']
            shares = 0
            position = 0

        # Portfolio value at this step
        portfolio_value = cash + shares * row['Close']
        portfolio_values.append(portfolio_value)

    print(f"Length of portfolio_values: {len(portfolio_values)}")
    print(f"Length of data: {len(data)}")

    data['Portfolio Value'] = portfolio_values
    return data


def performance_metrics(data):
    # Calculate daily returns of the portfolio
    data['Daily Return'] = data['Portfolio Value'].pct_change()
    
    # Total return from start to end
    total_return = (data['Portfolio Value'].iloc[-1] / data['Portfolio Value'].iloc[0]) - 1
    
    # Annualized return
    trading_days = 252  # typical number of trading days in a year
    annualized_return = (1 + total_return) ** (trading_days / len(data)) - 1
    
    # Max drawdown
    cumulative_max = data['Portfolio Value'].cummax()
    drawdown = (data['Portfolio Value'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio (assuming risk-free rate ~0 for simplicity)
    sharpe_ratio = data['Daily Return'].mean() / data['Daily Return'].std() * (trading_days ** 0.5)
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

def plot_portfolio_value(data):
    plt.figure(figsize=(12,6))
    plt.plot(data['Portfolio Value'], label='Portfolio Value', color='green')
    plt.title("Backtested Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
   # Get Apple stock data for last 1 year
    data = yf.download("AAPL", period="2y")
    # Plot the closing price
    data['Close'].plot(title="AAPL Closing Price - Last 1 Year")
    plt.show() 
    

if __name__ == "__main__":
        df = download_data()
        df = add_moving_averages(df)
        df = generate_signals(df)
        
        print("Columns in DataFrame:", df.columns)  # <-- Add here
        print(df.head())                           # <-- And here
        df = backtest_strategy(df)
        
        performance_metrics(df)
        plot_moving_averages(df)
        plot_portfolio_value(df)
        
        print(df[['Close_AAPL', 'MA20', 'MA50', 'Signal', 'Position']].tail(10)) 