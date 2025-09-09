# main.py
"""
Algorithmic Trading Strategy Implementation
Moving Average Crossover Strategy with Risk Management
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class MovingAverageCrossoverStrategy:
    """
    Professional implementation of Moving Average Crossover trading strategy
    with comprehensive backtesting and risk management capabilities.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        self.data = None
        self.results = None
        
    def download_data(self, ticker: str = "AAPL", period: str = "2y") -> pd.DataFrame:
        """
        Download historical stock data from Yahoo Finance.
        
        Args:
            ticker: Stock symbol
            period: Time period for historical data
            
        Returns:
            DataFrame with historical price data
        """
        try:
            logger.info(f"Downloading data for {ticker} over {period}")
            df = yf.download(ticker, period=period, progress=False)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
                
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
            
            # Ensure we have Close price column
            close_col = [col for col in df.columns if 'Close' in col]
            if not close_col:
                raise ValueError("No Close price column found in data")
                
            # Standardize column name
            df['Close'] = df[close_col[0]]
            df['Volume'] = df[[col for col in df.columns if 'Volume' in col][0]] if any('Volume' in col for col in df.columns) else 0
            
            self.data = df
            logger.info(f"Successfully downloaded {len(df)} data points")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
    
    def calculate_technical_indicators(self) -> pd.DataFrame:
        """
        Calculate technical indicators including moving averages and additional metrics.
        
        Returns:
            DataFrame with technical indicators
        """
        if self.data is None:
            raise ValueError("No data available. Call download_data() first.")
        
        data = self.data.copy()
        
        # Moving Averages
        data[f'MA_{self.short_window}'] = data['Close'].rolling(window=self.short_window).mean()
        data[f'MA_{self.long_window}'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Volatility (20-day rolling standard deviation)
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        # Returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Bollinger Bands
        data['BB_Upper'] = data[f'MA_{self.short_window}'] + (2 * data['Volatility'])
        data['BB_Lower'] = data[f'MA_{self.short_window}'] - (2 * data['Volatility'])
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        self.data = data
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover strategy.
        
        Returns:
            DataFrame with trading signals
        """
        if self.data is None:
            raise ValueError("No data available. Calculate indicators first.")
        
        data = self.data.copy()
        
        # Initialize signal columns
        data['Signal'] = 0
        data['Position'] = 0
        
        # Generate signals
        short_ma = f'MA_{self.short_window}'
        long_ma = f'MA_{self.long_window}'
        
        # Buy signal: short MA crosses above long MA
        data.loc[data[short_ma] > data[long_ma], 'Signal'] = 1
        
        # Sell signal: short MA crosses below long MA  
        data.loc[data[short_ma] < data[long_ma], 'Signal'] = -1
        
        # Position changes (actual trade signals)
        data['Position'] = data['Signal'].diff()
        
        # Additional filter: only trade when RSI is not in extreme territory
        data.loc[(data['RSI'] > 80) & (data['Position'] == 2), 'Position'] = 0  # Don't buy when overbought
        data.loc[(data['RSI'] < 20) & (data['Position'] == -2), 'Position'] = 0  # Don't sell when oversold
        
        self.data = data
        return data
    
    def backtest_strategy(self, initial_capital: float = 100000, 
                         transaction_cost: float = 0.001,
                         position_size: float = 1.0) -> pd.DataFrame:
        """
        Backtest the trading strategy with realistic constraints.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as percentage of trade value
            position_size: Position sizing (1.0 = 100% of available capital)
            
        Returns:
            DataFrame with backtest results
        """
        if self.data is None:
            raise ValueError("No data with signals available.")
        
        data = self.data.copy()
        data = data.dropna().sort_index()
        
        # Portfolio tracking
        cash = initial_capital
        shares = 0
        portfolio_values = []
        positions = []
        trades = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            signal = row['Position']
            price = row['Close']
            
            if pd.isna(signal) or pd.isna(price):
                portfolio_values.append(cash + shares * price if not pd.isna(price) else cash)
                positions.append(shares)
                continue
            
            # Execute trades
            if signal == 2:  # Buy signal (signal changed from -1 or 0 to 1)
                if shares == 0 and cash > 0:  # Only buy if not already holding
                    # Calculate position size
                    max_shares = int((cash * position_size) // price)
                    if max_shares > 0:
                        transaction_cost_amount = max_shares * price * transaction_cost
                        if cash >= (max_shares * price + transaction_cost_amount):
                            shares = max_shares
                            cash -= (shares * price + transaction_cost_amount)
                            trades.append({
                                'Date': date,
                                'Type': 'BUY',
                                'Shares': shares,
                                'Price': price,
                                'Value': shares * price,
                                'Cost': transaction_cost_amount
                            })
                            logger.debug(f"BUY: {shares} shares at ${price:.2f} on {date}")
            
            elif signal == -2:  # Sell signal (signal changed from 1 to -1 or 0)
                if shares > 0:  # Only sell if holding shares
                    transaction_cost_amount = shares * price * transaction_cost
                    cash += (shares * price - transaction_cost_amount)
                    trades.append({
                        'Date': date,
                        'Type': 'SELL',
                        'Shares': shares,
                        'Price': price,
                        'Value': shares * price,
                        'Cost': transaction_cost_amount
                    })
                    logger.debug(f"SELL: {shares} shares at ${price:.2f} on {date}")
                    shares = 0
            
            # Calculate portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
            positions.append(shares)
        
        data['Portfolio_Value'] = portfolio_values
        data['Shares_Held'] = positions
        data['Cash'] = [pv - (sh * data.loc[data.index[i], 'Close']) 
                       for i, (pv, sh) in enumerate(zip(portfolio_values, positions))]
        
        self.results = data
        self.trades = pd.DataFrame(trades)
        
        logger.info(f"Backtest completed. {len(trades)} trades executed.")
        return data
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance and risk metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if self.results is None:
            raise ValueError("No backtest results available.")
        
        data = self.results.copy()
        initial_value = data['Portfolio_Value'].iloc[0]
        final_value = data['Portfolio_Value'].iloc[-1]
        
        # Basic performance metrics
        total_return = (final_value / initial_value) - 1
        
        # Calculate daily returns
        portfolio_returns = data['Portfolio_Value'].pct_change().dropna()
        market_returns = data['Daily_Return'].dropna()
        
        # Align returns for comparison
        aligned_portfolio = portfolio_returns.reindex(market_returns.index).dropna()
        aligned_market = market_returns.reindex(aligned_portfolio.index).dropna()
        
        # Annualized metrics
        trading_days = 252
        periods = len(aligned_portfolio)
        years = periods / trading_days
        
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        annualized_volatility = aligned_portfolio.std() * np.sqrt(trading_days)
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(data['Portfolio_Value'])
        var_95 = np.percentile(aligned_portfolio, 5)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = aligned_portfolio[aligned_portfolio < 0]
        downside_deviation = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
        
        # Beta calculation
        if len(aligned_portfolio) > 1 and len(aligned_market) > 1:
            covariance = np.cov(aligned_portfolio, aligned_market)[0][1]
            market_variance = np.var(aligned_market)
            beta = covariance / market_variance if market_variance != 0 else 0
            
            # Alpha calculation
            market_annual_return = (1 + aligned_market.mean()) ** trading_days - 1
            alpha = annualized_return - (risk_free_rate + beta * (market_annual_return - risk_free_rate))
        else:
            beta = 0
            alpha = 0
        
        # Win rate
        if hasattr(self, 'trades') and len(self.trades) > 0:
            trades_df = self.trades
            if len(trades_df) >= 2:
                # Calculate P&L for each trade pair
                buy_trades = trades_df[trades_df['Type'] == 'BUY']
                sell_trades = trades_df[trades_df['Type'] == 'SELL']
                winning_trades = sum(1 for b, s in zip(buy_trades.iterrows(), sell_trades.iterrows()) 
                                   if s[1]['Price'] > b[1]['Price'])
                total_trade_pairs = min(len(buy_trades), len(sell_trades))
                win_rate = winning_trades / total_trade_pairs if total_trade_pairs > 0 else 0
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Maximum Drawdown': max_drawdown,
            'Value at Risk (95%)': var_95,
            'Beta': beta,
            'Alpha': alpha,
            'Win Rate': win_rate,
            'Number of Trades': len(self.trades) if hasattr(self, 'trades') else 0
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def plot_results(self, save_path: str = None) -> None:
        """
        Create comprehensive visualization of backtest results.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.results is None:
            raise ValueError("No results to plot.")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithmic Trading Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        data = self.results
        
        # Plot 1: Price and Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=1.5, color='black')
        ax1.plot(data.index, data[f'MA_{self.short_window}'], label=f'{self.short_window}-day MA', 
                linewidth=1, color='blue', alpha=0.7)
        ax1.plot(data.index, data[f'MA_{self.long_window}'], label=f'{self.long_window}-day MA', 
                linewidth=1, color='red', alpha=0.7)
        
        # Mark buy/sell points
        buy_signals = data[data['Position'] == 2]
        sell_signals = data[data['Position'] == -2]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', 
                   marker='^', s=60, label='Buy Signal', alpha=0.8)
        ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', 
                   marker='v', s=60, label='Sell Signal', alpha=0.8)
        
        ax1.set_title('Price Action with Trading Signals')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Value
        ax2 = axes[0, 1]
        ax2.plot(data.index, data['Portfolio_Value'], label='Strategy Portfolio', 
                linewidth=2, color='green')
        
        # Benchmark (buy and hold)
        benchmark_value = data['Close'] / data['Close'].iloc[0] * data['Portfolio_Value'].iloc[0]
        ax2.plot(data.index, benchmark_value, label='Buy & Hold Benchmark', 
                linewidth=1, color='orange', alpha=0.7)
        
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax3 = axes[1, 0]
        peak = data['Portfolio_Value'].expanding().max()
        drawdown = (data['Portfolio_Value'] - peak) / peak * 100
        ax3.fill_between(data.index, drawdown, 0, alpha=0.3, color='red')
        ax3.plot(data.index, drawdown, color='red', linewidth=1)
        ax3.set_title('Portfolio Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rolling Returns
        ax4 = axes[1, 1]
        rolling_returns = data['Portfolio_Value'].pct_change().rolling(30).mean() * 100
        ax4.plot(data.index, rolling_returns, color='purple', linewidth=1)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('30-Day Rolling Average Returns')
        ax4.set_ylabel('Returns (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive strategy performance report."""
        if self.results is None:
            return "No backtest results available."
        
        metrics = self.calculate_performance_metrics()
        
        report = f"""
ALGORITHMIC TRADING STRATEGY PERFORMANCE REPORT
{'=' * 50}

Strategy: Moving Average Crossover ({self.short_window}-day / {self.long_window}-day)
Backtest Period: {self.results.index[0].strftime('%Y-%m-%d')} to {self.results.index[-1].strftime('%Y-%m-%d')}
Total Trading Days: {len(self.results)}

PERFORMANCE METRICS
{'=' * 20}
Total Return:           {metrics['Total Return']:.2%}
Annualized Return:      {metrics['Annualized Return']:.2%}
Annualized Volatility:  {metrics['Annualized Volatility']:.2%}
Sharpe Ratio:          {metrics['Sharpe Ratio']:.3f}
Sortino Ratio:         {metrics['Sortino Ratio']:.3f}

RISK METRICS
{'=' * 15}
Maximum Drawdown:      {metrics['Maximum Drawdown']:.2%}
Value at Risk (95%):   {metrics['Value at Risk (95%)']:.2%}
Beta:                  {metrics['Beta']:.3f}
Alpha:                 {metrics['Alpha']:.2%}

TRADING STATISTICS
{'=' * 20}
Total Number of Trades: {int(metrics['Number of Trades'])}
Win Rate:              {metrics['Win Rate']:.2%}

PORTFOLIO SUMMARY
{'=' * 18}
Initial Capital:       ${self.results['Portfolio_Value'].iloc[0]:,.2f}
Final Portfolio Value: ${self.results['Portfolio_Value'].iloc[-1]:,.2f}
Total P&L:            ${self.results['Portfolio_Value'].iloc[-1] - self.results['Portfolio_Value'].iloc[0]:,.2f}
"""
        return report


def main():
    """Main execution function for demonstration."""
    # Initialize strategy
    strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
    
    # Download and prepare data
    strategy.download_data("AAPL", "2y")
    strategy.calculate_technical_indicators()
    strategy.generate_signals()
    
    # Run backtest
    strategy.backtest_strategy(initial_capital=100000, transaction_cost=0.001)
    
    # Display results
    print(strategy.generate_report())
    strategy.plot_results()


if __name__ == "__main__":
    main()
