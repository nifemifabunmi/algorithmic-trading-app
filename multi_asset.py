# multi_asset.py
"""
Multi-Asset Portfolio Strategy Implementation
Extends the base MovingAverageCrossoverStrategy for portfolio-level optimization
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
import logging
from main import moving_avg_strategy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class multi_asset_strategy:
    """
    Multi-asset portfolio implementation with correlation analysis,
    portfolio optimization, and risk management across multiple instruments.
    """
    
    def __init__(self, tickers: List[str], 
                 short_window: int = 20, 
                 long_window: int = 50,
                 rebalance_frequency: str = 'monthly'):
        """
        Initialize multi-asset strategy.
        
        Args:
            tickers: List of stock symbols
            short_window: Short MA period
            long_window: Long MA period
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
        """
        self.tickers = tickers
        self.short_window = short_window
        self.long_window = long_window
        self.rebalance_frequency = rebalance_frequency
        
        # Strategy instances for each asset
        self.strategies = {}
        self.portfolio_data = None
        self.portfolio_results = None
        self.correlation_matrix = None
        self.optimized_weights = None
        
        # Initialize individual strategies
        for ticker in tickers:
            self.strategies[ticker] = MovingAverageCrossoverStrategy(
                short_window=short_window, 
                long_window=long_window
            )
    
    def download_multi_asset_data(self, period: str = "2y") -> pd.DataFrame:
        """
        Download data for all assets in the portfolio.
        
        Args:
            period: Time period for historical data
            
        Returns:
            Combined DataFrame with all asset data
        """
        try:
            logger.info(f"Downloading data for {len(self.tickers)} assets...")
            
            # Download all data at once for efficiency
            combined_data = yf.download(self.tickers, period=period, progress=False)
            
            if combined_data.empty:
                raise ValueError("No data downloaded for any tickers")
            
            # Process each ticker's data
            portfolio_data = {}
            
            for ticker in self.tickers:
                try:
                    # Extract single ticker data
                    if len(self.tickers) == 1:
                        ticker_data = combined_data
                    else:
                        ticker_data = combined_data.xs(ticker, level=1, axis=1) if isinstance(combined_data.columns, pd.MultiIndex) else combined_data
                    
                    # Download individual strategy data
                    self.strategies[ticker].download_data(ticker, period)
                    self.strategies[ticker].calculate_technical_indicators()
                    self.strategies[ticker].generate_signals()
                    
                    portfolio_data[ticker] = {
                        'data': self.strategies[ticker].data,
                        'close': self.strategies[ticker].data['Close'],
                        'signals': self.strategies[ticker].data['Signal'],
                        'position': self.strategies[ticker].data['Position']
                    }
                    
                    logger.info(f"Successfully processed {ticker}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process {ticker}: {str(e)}")
                    continue
            
            self.portfolio_data = portfolio_data
            logger.info(f"Successfully downloaded data for {len(portfolio_data)} assets")
            return self.create_combined_dataframe()
            
        except Exception as e:
            logger.error(f"Error in multi-asset data download: {str(e)}")
            raise
    
    def create_combined_dataframe(self) -> pd.DataFrame:
        """Create combined DataFrame with all assets' data."""
        if not self.portfolio_data:
            raise ValueError("No portfolio data available")
        
        # Create combined price matrix
        price_data = {}
        signal_data = {}
        
        for ticker, data in self.portfolio_data.items():
            price_data[f'{ticker}_Close'] = data['close']
            signal_data[f'{ticker}_Signal'] = data['signals']
        
        combined_df = pd.DataFrame(price_data)
        signal_df = pd.DataFrame(signal_data)
        
        # Combine price and signal data
        for col in signal_df.columns:
            combined_df[col] = signal_df[col]
        
        # Calculate returns
        for ticker in self.tickers:
            if f'{ticker}_Close' in combined_df.columns:
                combined_df[f'{ticker}_Return'] = combined_df[f'{ticker}_Close'].pct_change()
        
        return combined_df.dropna()
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets."""
        if not self.portfolio_data:
            raise ValueError("No data available for correlation calculation")
        
        returns_data = {}
        for ticker in self.tickers:
            if ticker in self.portfolio_data:
                returns = self.portfolio_data[ticker]['data']['Daily_Return']
                returns_data[ticker] = returns
        
        returns_df = pd.DataFrame(returns_data).dropna()
        self.correlation_matrix = returns_df.corr()
        
        logger.info("Correlation matrix calculated successfully")
        return self.correlation_matrix
    
    def optimize_portfolio_weights(self, method: str = 'equal_risk_contribution',
                                 target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights using various methods.
        
        Args:
            method: 'equal_weight', 'market_cap', 'equal_risk_contribution', 'mean_variance'
            target_return: Target return for mean-variance optimization
            
        Returns:
            Dictionary of optimized weights
        """
        if not self.portfolio_data:
            raise ValueError("No portfolio data available for optimization")
        
        n_assets = len(self.tickers)
        
        if method == 'equal_weight':
            weights = {ticker: 1.0 / n_assets for ticker in self.tickers}
        
        elif method == 'equal_risk_contribution':
            weights = self._calculate_risk_parity_weights()
        
        elif method == 'mean_variance' and target_return is not None:
            weights = self._calculate_mean_variance_weights(target_return)
        
        else:
            # Default to equal weight
            weights = {ticker: 1.0 / n_assets for ticker in self.tickers}
        
        self.optimized_weights = weights
        logger.info(f"Portfolio optimization completed using {method} method")
        
        return weights
    
    def _calculate_risk_parity_weights(self) -> Dict[str, float]:
        """Calculate equal risk contribution weights."""
        # Get returns data
        returns_data = {}
        for ticker in self.tickers:
            if ticker in self.portfolio_data:
                returns_data[ticker] = self.portfolio_data[ticker]['data']['Daily_Return']
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 50:  # Need sufficient data
            logger.warning("Insufficient data for risk parity optimization, using equal weights")
            return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # Risk parity optimization
        n_assets = len(self.tickers)
        
        def risk_budget_objective(weights, cov_matrix):
            """Objective function for risk parity."""
            weights = np.array(weights)
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib
            return np.sum((contrib - portfolio_var / n_assets) ** 2)
        
        # Constraints: weights sum to 1, all positive
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        try:
            result = minimize(
                risk_budget_objective,
                initial_guess,
                args=(cov_matrix.values,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimized_weights = {ticker: weight for ticker, weight in zip(self.tickers, result.x)}
                return optimized_weights
            else:
                logger.warning("Risk parity optimization failed, using equal weights")
                return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
                
        except Exception as e:
            logger.warning(f"Risk parity optimization error: {str(e)}, using equal weights")
            return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
    
    def _calculate_mean_variance_weights(self, target_return: float) -> Dict[str, float]:
        """Calculate mean-variance optimal weights."""
        # Implementation of Markowitz optimization
        returns_data = {}
        for ticker in self.tickers:
            if ticker in self.portfolio_data:
                returns_data[ticker] = self.portfolio_data[ticker]['data']['Daily_Return']
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 50:
            logger.warning("Insufficient data for mean-variance optimization")
            return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
        
        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        n_assets = len(self.tickers)
        
        # Optimization objective: minimize variance
        def portfolio_variance(weights, cov_matrix):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
            {'type': 'eq', 'fun': lambda x: np.dot(expected_returns, x) - target_return}  # Target return
        ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        try:
            result = minimize(
                portfolio_variance,
                initial_guess,
                args=(cov_matrix.values,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return {ticker: weight for ticker, weight in zip(self.tickers, result.x)}
            else:
                logger.warning("Mean-variance optimization failed")
                return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
                
        except Exception as e:
            logger.warning(f"Mean-variance optimization error: {str(e)}")
            return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
    
    def backtest_portfolio(self, initial_capital: float = 100000,
                          transaction_cost: float = 0.001,
                          rebalance_cost: float = 0.0005) -> pd.DataFrame:
        """
        Backtest the multi-asset portfolio strategy.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per individual trade
            rebalance_cost: Additional cost for portfolio rebalancing
            
        Returns:
            DataFrame with portfolio backtest results
        """
        if not self.portfolio_data or not self.optimized_weights:
            raise ValueError("Portfolio data and weights required for backtesting")
        
        # Get rebalancing dates
        combined_df = self.create_combined_dataframe()
        rebalance_dates = self._get_rebalance_dates(combined_df.index)
        
        # Initialize portfolio tracking
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {ticker: 0 for ticker in self.tickers}
        portfolio_values = []
        
        # Track individual asset allocations
        asset_allocations = {ticker: [] for ticker in self.tickers}
        rebalance_costs = []
        
        for i, (date, row) in enumerate(combined_df.iterrows()):
            # Check if rebalancing date
            is_rebalance_date = date in rebalance_dates or i == 0
            
            if is_rebalance_date:
                # Calculate current portfolio value
                current_value = cash
                for ticker in self.tickers:
                    if f'{ticker}_Close' in row and not pd.isna(row[f'{ticker}_Close']):
                        current_value += positions[ticker] * row[f'{ticker}_Close']
                
                # Rebalance portfolio
                rebalance_cost_total = 0
                
                for ticker in self.tickers:
                    if ticker in self.optimized_weights and f'{ticker}_Close' in row:
                        target_value = current_value * self.optimized_weights[ticker]
                        current_asset_value = positions[ticker] * row[f'{ticker}_Close'] if not pd.isna(row[f'{ticker}_Close']) else 0
                        
                        if not pd.isna(row[f'{ticker}_Close']) and row[f'{ticker}_Close'] > 0:
                            # Calculate required position change
                            target_shares = target_value / row[f'{ticker}_Close']
                            shares_diff = target_shares - positions[ticker]
                            
                            if abs(shares_diff) > 0.01:  # Only trade if significant difference
                                trade_value = abs(shares_diff * row[f'{ticker}_Close'])
                                trade_cost = trade_value * (transaction_cost + rebalance_cost)
                                rebalance_cost_total += trade_cost
                                
                                # Update positions and cash
                                positions[ticker] = target_shares
                                cash = cash - shares_diff * row[f'{ticker}_Close'] - trade_cost
                
                rebalance_costs.append(rebalance_cost_total)
            
            # Handle individual asset signals (for tactical overlays)
            for ticker in self.tickers:
                signal_col = f'{ticker}_Signal'
                if signal_col in row and not pd.isna(row[signal_col]):
                    # Implement signal-based adjustments if needed
                    pass
            
            # Calculate current portfolio value
            current_portfolio_value = cash
            for ticker in self.tickers:
                if f'{ticker}_Close' in row and not pd.isna(row[f'{ticker}_Close']):
                    asset_value = positions[ticker] * row[f'{ticker}_Close']
                    current_portfolio_value += asset_value
                    asset_allocations[ticker].append(asset_value)
                else:
                    asset_allocations[ticker].append(0)
            
            portfolio_values.append(current_portfolio_value)
        
        # Create results DataFrame
        results_df = combined_df.copy()
        results_df['Portfolio_Value'] = portfolio_values
        results_df['Cash'] = cash
        
        # Add individual asset allocations
        for ticker in self.tickers:
            results_df[f'{ticker}_Allocation'] = asset_allocations[ticker]
        
        # Calculate portfolio returns
        results_df['Portfolio_Return'] = results_df['Portfolio_Value'].pct_change()
        
        self.portfolio_results = results_df
        
        logger.info(f"Portfolio backtest completed. Total rebalancing costs: ${sum(rebalance_costs):,.2f}")
        return results_df
    
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency."""
        if self.rebalance_frequency == 'daily':
            return date_index.tolist()
        elif self.rebalance_frequency == 'weekly':
            return [date for date in date_index if date.weekday() == 0]  # Mondays
        elif self.rebalance_frequency == 'monthly':
            return [date for date in date_index if date.day == 1 or 
                   (date_index.get_loc(date) > 0 and 
                    date.month != date_index[date_index.get_loc(date) - 1].month)]
        elif self.rebalance_frequency == 'quarterly':
            return [date for date in date_index if date.month in [1, 4, 7, 10] and date.day == 1]
        else:
            return [date_index[0], date_index[-1]]  # Start and end only
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""
        if self.portfolio_results is None:
            raise ValueError("No portfolio backtest results available")
        
        results = self.portfolio_results
        portfolio_returns = results['Portfolio_Return'].dropna()
        
        # Basic metrics
        total_return = (results['Portfolio_Value'].iloc[-1] / results['Portfolio_Value'].iloc[0]) - 1
        
        # Annualized metrics
        trading_days = 252
        periods = len(portfolio_returns)
        years = periods / trading_days if periods > 0 else 1
        
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        annualized_volatility = portfolio_returns.std() * np.sqrt(trading_days) if len(portfolio_returns) > 1 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(results['Portfolio_Value'])
        
        # Sharpe ratio
        risk_free_rate = 0.02
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Portfolio-specific metrics
        diversification_ratio = self._calculate_diversification_ratio()
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Diversification Ratio': diversification_ratio,
            'Number of Assets': len(self.tickers)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown for portfolio."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def _calculate_diversification_ratio(self) -> float:
        """Calculate portfolio diversification ratio."""
        if not self.correlation_matrix is not None or not self.optimized_weights:
            return 1.0  # No diversification benefit
        
        try:
            # Weighted average of individual volatilities
            individual_vols = {}
            for ticker in self.tickers:
                if ticker in self.portfolio_data:
                    returns = self.portfolio_data[ticker]['data']['Daily_Return']
                    individual_vols[ticker] = returns.std() * np.sqrt(252)
            
            if not individual_vols:
                return 1.0
            
            # Weighted average volatility
            weighted_vol = sum(self.optimized_weights.get(ticker, 0) * individual_vols.get(ticker, 0) 
                              for ticker in self.tickers)
            
            # Portfolio volatility
            if self.portfolio_results is not None:
                portfolio_vol = self.portfolio_results['Portfolio_Return'].std() * np.sqrt(252)
                diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol != 0 else 1.0
                return max(1.0, diversification_ratio)  # Should be >= 1
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating diversification ratio: {str(e)}")
            return 1.0
    
    def plot_portfolio_analysis(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive portfolio analysis visualization."""
        if self.portfolio_results is None:
            raise ValueError("No portfolio results to plot")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Asset Portfolio Analysis', fontsize=16, fontweight='bold')
        
        results = self.portfolio_results
        
        # Plot 1: Portfolio Value Over Time
        ax1 = axes[0, 0]
        ax1.plot(results.index, results['Portfolio_Value'], linewidth=2, color='blue', label='Portfolio')
        
        # Add individual asset benchmarks
        for ticker in self.tickers[:3]:  # Limit to first 3 for readability
            if f'{ticker}_Close' in results.columns:
                initial_price = results[f'{ticker}_Close'].iloc[0]
                initial_value = results['Portfolio_Value'].iloc[0]
                benchmark = results[f'{ticker}_Close'] / initial_price * initial_value
                ax1.plot(results.index, benchmark, alpha=0.6, linestyle='--', label=f'{ticker} Benchmark')
        
        ax1.set_title('Portfolio Performance Comparison')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Asset Allocation Over Time
        ax2 = axes[0, 1]
        bottom = np.zeros(len(results))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tickers)))
        for i, ticker in enumerate(self.tickers):
            allocation_col = f'{ticker}_Allocation'
            if allocation_col in results.columns:
                allocation_pct = results[allocation_col] / results['Portfolio_Value'] * 100
                ax2.fill_between(results.index, bottom, bottom + allocation_pct, 
                               alpha=0.8, color=colors[i], label=ticker)
                bottom += allocation_pct
        
        ax2.set_title('Asset Allocation Over Time')
        ax2.set_ylabel('Allocation (%)')
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        # Plot 3: Correlation Heatmap
        ax3 = axes[1, 0]
        if self.correlation_matrix is not None:
            im = ax3.imshow(self.correlation_matrix.values, cmap='RdBu', vmin=-1, vmax=1)
            ax3.set_xticks(range(len(self.tickers)))
            ax3.set_yticks(range(len(self.tickers)))
            ax3.set_xticklabels(self.tickers, rotation=45)
            ax3.set_yticklabels(self.tickers)
            
            # Add correlation values
            for i in range(len(self.tickers)):
                for j in range(len(self.tickers)):
                    value = self.correlation_matrix.iloc[i, j]
                    ax3.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color='white' if abs(value) > 0.5 else 'black')
            
            plt.colorbar(im, ax=ax3)
            ax3.set_title('Asset Correlation Matrix')
        
        # Plot 4: Risk-Return Scatter
        ax4 = axes[1, 1]
        
        # Calculate individual asset metrics for scatter plot
        asset_returns = []
        asset_volatilities = []
        asset_names = []
        
        for ticker in self.tickers:
            if ticker in self.portfolio_data:
                returns = self.portfolio_data[ticker]['data']['Daily_Return']
                if len(returns.dropna()) > 1:
                    annual_return = returns.mean() * 252
                    annual_vol = returns.std() * np.sqrt(252)
                    asset_returns.append(annual_return)
                    asset_volatilities.append(annual_vol)
                    asset_names.append(ticker)
        
        # Plot individual assets
        ax4.scatter(asset_volatilities, asset_returns, alpha=0.7, s=100, color='lightblue')
        
        # Plot portfolio
        portfolio_metrics = self.calculate_portfolio_metrics()
        ax4.scatter(portfolio_metrics['Annualized Volatility'], 
                   portfolio_metrics['Annualized Return'],
                   color='red', s=200, alpha=0.8, label='Portfolio')
        
        # Add labels
        for i, name in enumerate(asset_names):
            ax4.annotate(name, (asset_volatilities[i], asset_returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Annualized Volatility')
        ax4.set_ylabel('Annualized Return')
        ax4.set_title('Risk-Return Profile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_portfolio_report(self) -> str:
        """Generate comprehensive portfolio analysis report."""
        if self.portfolio_results is None:
            return "No portfolio backtest results available."
        
        metrics = self.calculate_portfolio_metrics()
        
        # Individual asset performance
        asset_performance = {}
        for ticker in self.tickers:
            if ticker in self.portfolio_data:
                strategy = self.strategies[ticker]
                if strategy.results is not None:
                    individual_metrics = strategy.calculate_performance_metrics()
                    asset_performance[ticker] = individual_metrics
        
        report = f"""
MULTI-ASSET PORTFOLIO STRATEGY REPORT
{'=' * 50}

Portfolio Composition: {', '.join(self.tickers)}
Rebalancing Frequency: {self.rebalance_frequency.title()}
Optimization Method: Risk Parity
Backtest Period: {self.portfolio_results.index[0].strftime('%Y-%m-%d')} to {self.portfolio_results.index[-1].strftime('%Y-%m-%d')}

PORTFOLIO-LEVEL METRICS
{'=' * 25}
Total Return:              {metrics['Total Return']:.2%}
Annualized Return:         {metrics['Annualized Return']:.2%}
Annualized Volatility:     {metrics['Annualized Volatility']:.2%}
Sharpe Ratio:             {metrics['Sharpe Ratio']:.3f}
Maximum Drawdown:         {metrics['Maximum Drawdown']:.2%}
Diversification Ratio:    {metrics['Diversification Ratio']:.2f}

PORTFOLIO WEIGHTS
{'=' * 18}"""

        if self.optimized_weights:
            for ticker, weight in self.optimized_weights.items():
                report += f"\n{ticker:>6}: {weight:>8.1%}"

        report += f"""

INDIVIDUAL ASSET PERFORMANCE
{'=' * 30}"""

        for ticker, perf in asset_performance.items():
            report += f"""

{ticker} Performance:
  Total Return:     {perf['Total Return']:.2%}
  Sharpe Ratio:     {perf['Sharpe Ratio']:.3f}
  Max Drawdown:     {perf['Maximum Drawdown']:.2%}
  Number of Trades: {int(perf['Number of Trades'])}"""

        if self.correlation_matrix is not None:
            report += f"""

CORRELATION ANALYSIS
{'=' * 20}
Average Correlation: {self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].mean():.3f}
Highest Correlation: {self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].max():.3f}
Lowest Correlation:  {self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].min():.3f}"""

        report += f"""

PORTFOLIO SUMMARY
{'=' * 18}
Initial Capital:       ${self.portfolio_results['Portfolio_Value'].iloc[0]:,.2f}
Final Portfolio Value: ${self.portfolio_results['Portfolio_Value'].iloc[-1]:,.2f}
Total P&L:            ${self.portfolio_results['Portfolio_Value'].iloc[-1] - self.portfolio_results['Portfolio_Value'].iloc[0]:,.2f}

RISK ANALYSIS
{'=' * 14}
The portfolio demonstrates {'strong' if metrics['Diversification Ratio'] > 1.2 else 'moderate'} diversification benefits
with a diversification ratio of {metrics['Diversification Ratio']:.2f}. 
{'Excellent' if metrics['Sharpe Ratio'] > 1.5 else 'Good' if metrics['Sharpe Ratio'] > 1.0 else 'Moderate'} 
risk-adjusted returns with Sharpe ratio of {metrics['Sharpe Ratio']:.3f}.
"""
        
        return report


# Usage example and integration
def demo_multi_asset_strategy():
    """Demonstration of multi-asset strategy implementation."""
    
    # Define portfolio
    portfolio_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    # Initialize strategy
    multi_strategy = MultiAssetPortfolioStrategy(
        tickers=portfolio_tickers,
        short_window=20,
        long_window=50,
        rebalance_frequency='monthly'
    )
    
    # Download and process data
    print("Downloading multi-asset data...")
    multi_strategy.download_multi_asset_data("2y")
    
    # Calculate correlations
    print("Calculating correlations...")
    correlation_matrix = multi_strategy.calculate_correlation_matrix()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Optimize weights
    print("Optimizing portfolio weights...")
    weights = multi_strategy.optimize_portfolio_weights(method='equal_risk_contribution')
    print("Optimized Weights:")
    for ticker, weight in weights.items():
        print(f"{ticker}: {weight:.1%}")
    
    # Run backtest
    print("Running portfolio backtest...")
    results = multi_strategy.backtest_portfolio(initial_capital=100000)
    
    # Calculate and display metrics
    metrics = multi_strategy.calculate_portfolio_metrics()
    print(f"\nPortfolio Performance:")
    print(f"Total Return: {metrics['Total Return']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    print(f"Max Drawdown: {metrics['Maximum Drawdown']:.2%}")
    
    # Generate report
    report = multi_strategy.generate_portfolio_report()
    print("\n" + "="*50)
    print(report)
    
    # Plot results
    multi_strategy.plot_portfolio_analysis()


if __name__ == "__main__":
    demo_multi_asset_strategy()
