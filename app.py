# app.py
"""
Algorithmic Trading Dashboard
Built with Streamlit for interactive strategy analysis and backtesting
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from main import MovingAverageCrossoverStrategy
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Algorithmic Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 0.5rem 0;
}

.positive-metric {
    color: #28a745;
    font-weight: bold;
}

.negative-metric {
    color: #dc3545;
    font-weight: bold;
}

.info-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2196f3;
    margin: 1rem 0;
}

.stButton > button {
    background-color: #1f77b4;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 2rem;
    font-size: 16px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #1565c0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📈 Algorithmic Trading Strategy Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for strategy configuration
    with st.sidebar:
        st.header("🔧 Strategy Configuration")
        
        # Stock selection
        ticker = st.text_input(
            "Stock Symbol", 
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        # Time period
        period_options = {
            "1 Year": "1y",
            "2 Years": "2y", 
            "3 Years": "3y",
            "5 Years": "5y"
        }
        period_label = st.selectbox("Historical Data Period", list(period_options.keys()), index=1)
        period = period_options[period_label]
        
        st.subheader("Moving Average Parameters")
        short_ma = st.slider("Short MA (days)", 5, 50, 20, help="Fast moving average period")
        long_ma = st.slider("Long MA (days)", 20, 200, 50, help="Slow moving average period")
        
        if short_ma >= long_ma:
            st.error("Short MA must be less than Long MA")
            return
        
        st.subheader("Backtest Parameters")
        initial_capital = st.number_input(
            "Initial Capital ($)", 
            min_value=1000, 
            value=100000, 
            step=1000,
            help="Starting portfolio value"
        )
        
        transaction_cost = st.slider(
            "Transaction Cost (%)", 
            0.0, 1.0, 0.1, 0.1,
            help="Cost per trade as percentage of trade value"
        ) / 100
        
        position_size = st.slider(
            "Position Size (%)", 
            10, 100, 100, 10,
            help="Percentage of available capital to use per trade"
        ) / 100
        
        analyze_button = st.button("🚀 Run Strategy Analysis", use_container_width=True)
    
    # Main content area
    if analyze_button:
        try:
            with st.spinner(f"Analyzing {ticker} with {short_ma}/{long_ma} MA strategy..."):
                # Initialize and run strategy
                strategy = MovingAverageCrossoverStrategy(short_window=short_ma, long_window=long_ma)
                
                # Download and process data
                data = strategy.download_data(ticker, period)
                if data.empty:
                    st.error(f"No data found for ticker {ticker}. Please check the symbol.")
                    return
                
                strategy.calculate_technical_indicators()
                strategy.generate_signals()
                backtest_results = strategy.backtest_strategy(
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    position_size=position_size
                )
                
                # Calculate metrics
                metrics = strategy.calculate_performance_metrics()
                
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Performance Overview", 
                "📈 Charts", 
                "📋 Detailed Metrics",
                "🔍 Data Analysis",
                "📁 Export Results"
            ])
            
            with tab1:
                display_performance_overview(strategy, metrics, ticker)
            
            with tab2:
                display_charts(strategy)
            
            with tab3:
                display_detailed_metrics(metrics, strategy)
            
            with tab4:
                display_data_analysis(strategy)
            
            with tab5:
                display_export_options(strategy, ticker)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your inputs and try again.")
    
    else:
        # Display welcome message and instructions
        display_welcome_message()

def display_performance_overview(strategy, metrics, ticker):
    """Display key performance metrics in an overview format."""
    
    st.markdown(f"### 📊 Performance Summary for {ticker}")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = metrics['Total Return']
        return_color = "positive-metric" if total_return > 0 else "negative-metric"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Return</h4>
            <p class="{return_color}">{total_return:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        sharpe = metrics['Sharpe Ratio']
        sharpe_color = "positive-metric" if sharpe > 1 else "negative-metric"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Sharpe Ratio</h4>
            <p class="{sharpe_color}">{sharpe:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        max_dd = metrics['Maximum Drawdown']
        st.markdown(f"""
        <div class="metric-card">
            <h4>Max Drawdown</h4>
            <p class="negative-metric">{max_dd:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        win_rate = metrics['Win Rate']
        win_color = "positive-metric" if win_rate > 0.5 else "negative-metric"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Win Rate</h4>
            <p class="{win_color}">{win_rate:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Portfolio performance chart
    st.subheader("Portfolio Value Over Time")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Strategy performance
    ax.plot(strategy.results.index, strategy.results['Portfolio_Value'], 
            label='Strategy Portfolio', linewidth=2, color='#1f77b4')
    
    # Benchmark (Buy & Hold)
    initial_value = strategy.results['Portfolio_Value'].iloc[0]
    benchmark = strategy.results['Close'] / strategy.results['Close'].iloc[0] * initial_value
    ax.plot(strategy.results.index, benchmark, 
            label='Buy & Hold Benchmark', linewidth=2, color='#ff7f0e', alpha=0.8)
    
    ax.set_title(f'{ticker} Strategy vs Benchmark Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    st.pyplot(fig)
    
    # Performance comparison table
    st.subheader("Strategy vs Benchmark Comparison")
    
    benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
    comparison_df = pd.DataFrame({
        'Metric': ['Total Return', 'Final Value', 'Profit/Loss'],
        'Strategy': [
            f"{metrics['Total Return']:.2%}",
            f"${strategy.results['Portfolio_Value'].iloc[-1]:,.2f}",
            f"${strategy.results['Portfolio_Value'].iloc[-1] - initial_value:,.2f}"
        ],
        'Buy & Hold': [
            f"{benchmark_return:.2%}",
            f"${benchmark.iloc[-1]:,.2f}",
            f"${benchmark.iloc[-1] - initial_value:,.2f}"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True)

def display_charts(strategy):
    """Display comprehensive charts for strategy analysis."""
    
    st.subheader("📈 Technical Analysis Charts")
    
    # Chart 1: Price with Moving Averages and Signals
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    data = strategy.results
    
    # Price and moving averages
    ax1.plot(data.index, data['Close'], label='Close Price', linewidth=1.5, color='black')
    ax1.plot(data.index, data[f'MA_{strategy.short_window}'], 
             label=f'{strategy.short_window}-day MA', linewidth=1.2, color='blue', alpha=0.8)
    ax1.plot(data.index, data[f'MA_{strategy.long_window}'], 
             label=f'{strategy.long_window}-day MA', linewidth=1.2, color='red', alpha=0.8)
    
    # Bollinger Bands
    ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], 
                     alpha=0.1, color='gray', label='Bollinger Bands')
    
    # Trading signals
    buy_signals = data[data['Position'] == 2]
    sell_signals = data[data['Position'] == -2]
    
    ax1.scatter(buy_signals.index, buy_signals['Close'], 
               color='green', marker='^', s=100, label='Buy Signal', alpha=0.8, zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['Close'], 
               color='red', marker='v', s=100, label='Sell Signal', alpha=0.8, zorder=5)
    
    ax1.set_title('Price Action with Trading Signals', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    st.pyplot(fig1)
    
    # Chart 2: Portfolio Performance and Drawdown
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Portfolio value
    ax2a.plot(data.index, data['Portfolio_Value'], 
              label='Strategy Portfolio', linewidth=2, color='green')
    
    # Benchmark
    initial_value = data['Portfolio_Value'].iloc[0]
    benchmark = data['Close'] / data['Close'].iloc[0] * initial_value
    ax2a.plot(data.index, benchmark, 
              label='Buy & Hold', linewidth=2, color='orange', alpha=0.8)
    
    ax2a.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
    ax2a.set_ylabel('Portfolio Value ($)')
    ax2a.legend()
    ax2a.grid(True, alpha=0.3)
    ax2a.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Drawdown
    peak = data['Portfolio_Value'].expanding().max()
    drawdown = (data['Portfolio_Value'] - peak) / peak * 100
    ax2b.fill_between(data.index, drawdown, 0, alpha=0.3, color='red')
    ax2b.plot(data.index, drawdown, color='darkred', linewidth=1)
    ax2b.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax2b.set_xlabel('Date')
    ax2b.set_ylabel('Drawdown (%)')
    ax2b.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Chart 3: Additional Technical Indicators
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # RSI
    ax3a.plot(data.index, data['RSI'], color='purple', linewidth=1.5)
    ax3a.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax3a.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax3a.fill_between(data.index, 70, 100, alpha=0.1, color='red')
    ax3a.fill_between(data.index, 0, 30, alpha=0.1, color='green')
    ax3a.set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
    ax3a.set_ylabel('RSI')
    ax3a.legend()
    ax3a.grid(True, alpha=0.3)
    ax3a.set_ylim(0, 100)
    
    # Volume (if available)
    if 'Volume' in data.columns and data['Volume'].sum() > 0:
        ax3b.bar(data.index, data['Volume'], alpha=0.6, color='lightblue')
        ax3b.set_title('Trading Volume', fontsize=14, fontweight='bold')
        ax3b.set_ylabel('Volume')
        ax3b.grid(True, alpha=0.3)
    else:
        # Volatility instead of volume
        ax3b.plot(data.index, data['Volatility'], color='orange', linewidth=1.5)
        ax3b.set_title('Price Volatility (20-day rolling)', fontsize=14, fontweight='bold')
        ax3b.set_ylabel('Volatility')
        ax3b.grid(True, alpha=0.3)
    
    ax3b.set_xlabel('Date')
    
    plt.tight_layout()
    st.pyplot(fig3)

def display_detailed_metrics(metrics, strategy):
    """Display comprehensive performance and risk metrics."""
    
    st.subheader("📋 Comprehensive Performance Analysis")
    
    # Performance metrics table
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Return Metrics")
        performance_data = {
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Sortino Ratio'
            ],
            'Value': [
                f"{metrics['Total Return']:.2%}",
                f"{metrics['Annualized Return']:.2%}",
                f"{metrics['Annualized Volatility']:.2%}",
                f"{metrics['Sharpe Ratio']:.3f}",
                f"{metrics['Sortino Ratio']:.3f}"
            ]
        }
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 🛡️ Risk Metrics")
        risk_data = {
            'Metric': [
                'Maximum Drawdown',
                'Value at Risk (95%)',
                'Beta',
                'Alpha',
                'Win Rate'
            ],
            'Value': [
                f"{metrics['Maximum Drawdown']:.2%}",
                f"{metrics['Value at Risk (95%)']:.2%}",
                f"{metrics['Beta']:.3f}",
                f"{metrics['Alpha']:.2%}",
                f"{metrics['Win Rate']:.2%}"
            ]
        }
        risk_df = pd.DataFrame(risk_data)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    # Trading statistics
    st.markdown("#### 📊 Trading Statistics")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Total Trades", int(metrics['Number of Trades']))
    
    with col4:
        initial_value = strategy.results['Portfolio_Value'].iloc[0]
        final_value = strategy.results['Portfolio_Value'].iloc[-1]
        profit_loss = final_value - initial_value
        st.metric("Total P&L", f"${profit_loss:,.2f}", 
                 delta=f"{metrics['Total Return']:.2%}")
    
    with col5:
        days_in_market = len(strategy.results)
        st.metric("Days in Backtest", days_in_market)
    
    # Strategy interpretation
    st.markdown("#### 💡 Strategy Interpretation")
    
    interpretation_text = generate_interpretation(metrics)
    st.markdown(f"""
    <div class="info-box">
        {interpretation_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Risk-Return scatter plot
    st.markdown("#### 📊 Risk-Return Profile")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot strategy point
    ax.scatter(metrics['Annualized Volatility'] * 100, metrics['Annualized Return'] * 100,
               s=200, color='blue', alpha=0.7, label='Strategy', zorder=5)
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add Sharpe ratio lines
    for sr in [0.5, 1.0, 1.5, 2.0]:
        x_vals = np.linspace(0, 50, 100)
        y_vals = sr * x_vals
        ax.plot(x_vals, y_vals, '--', alpha=0.3, color='gray')
        ax.text(45, sr * 45, f'SR={sr}', fontsize=8, alpha=0.6)
    
    ax.set_xlabel('Annualized Volatility (%)')
    ax.set_ylabel('Annualized Return (%)')
    ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Annotate the strategy point
    ax.annotate(f'Sharpe: {metrics["Sharpe Ratio"]:.2f}',
                (metrics['Annualized Volatility'] * 100, metrics['Annualized Return'] * 100),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    st.pyplot(fig)

def display_data_analysis(strategy):
    """Display detailed data analysis and trade breakdown."""
    
    st.subheader("🔍 Detailed Data Analysis")
    
    # Trade history
    if hasattr(strategy, 'trades') and len(strategy.trades) > 0:
        st.markdown("#### 📋 Trade History")
        
        trades_df = strategy.trades.copy()
        trades_df['Date'] = trades_df['Date'].dt.strftime('%Y-%m-%d')
        trades_df['Price'] = trades_df['Price'].round(2)
        trades_df['Value'] = trades_df['Value'].round(2)
        trades_df['Cost'] = trades_df['Cost'].round(2)
        
        st.dataframe(trades_df, use_container_width=True)
        
        # Trade analysis
        buy_trades = trades_df[trades_df['Type'] == 'BUY']
        sell_trades = trades_df[trades_df['Type'] == 'SELL']
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            st.markdown("#### 📊 Trade Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Buy Orders", len(buy_trades))
            with col2:
                st.metric("Sell Orders", len(sell_trades))
            with col3:
                total_costs = trades_df['Cost'].sum()
                st.metric("Total Transaction Costs", f"${total_costs:.2f}")
    
    else:
        st.info("No trades were executed during the backtest period.")
    
    # Monthly returns analysis
    st.markdown("#### 📅 Monthly Returns Heatmap")
    
    monthly_returns = strategy.results['Portfolio_Value'].resample('M').last().pct_change().dropna()
    
    if len(monthly_returns) > 0:
        # Create monthly returns matrix
        returns_matrix = []
        years = []
        
        for year in monthly_returns.index.year.unique():
            year_data = monthly_returns[monthly_returns.index.year == year]
            month_returns = [0] * 12
            
            for month_data in year_data.iteritems():
                month_idx = month_data[0].month - 1
                month_returns[month_idx] = month_data[1]
            
            returns_matrix.append(month_returns)
            years.append(year)
        
        if returns_matrix:
            returns_df = pd.DataFrame(returns_matrix, 
                                    index=years,
                                    columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(returns_df.values, cmap='RdYlGn', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(12))
            ax.set_xticklabels(returns_df.columns)
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(years)
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Monthly Returns', rotation=270, labelpad=15)
            
            # Add text annotations
            for i in range(len(years)):
                for j in range(12):
                    value = returns_df.iloc[i, j]
                    if value != 0:
                        text = ax.text(j, i, f'{value:.1%}', 
                                     ha='center', va='center',
                                     color='white' if abs(value) > 0.02 else 'black',
                                     fontsize=8)
            
            ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
    
    # Data quality metrics
    st.markdown("#### 📊 Data Quality Metrics")
    
    data_quality = {
        'Metric': [
            'Total Data Points',
            'Missing Values',
            'Date Range',
            'Average Daily Volume',
            'Price Range'
        ],
        'Value': [
            len(strategy.results),
            strategy.results.isnull().sum().sum(),
            f"{strategy.results.index[0].strftime('%Y-%m-%d')} to {strategy.results.index[-1].strftime('%Y-%m-%d')}",
            f"{strategy.results['Volume'].mean():,.0f}" if 'Volume' in strategy.results.columns else "N/A",
            f"${strategy.results['Close'].min():.2f} - ${strategy.results['Close'].max():.2f}"
        ]
    }
    
    quality_df = pd.DataFrame(data_quality)
    st.dataframe(quality_df, use_container_width=True, hide_index=True)
    
    # Recent portfolio data
    st.markdown("#### 📋 Recent Portfolio Data")
    
    recent_data = strategy.results[['Close', f'MA_{strategy.short_window}', 
                                  f'MA_{strategy.long_window}', 'Portfolio_Value', 
                                  'Signal']].tail(10)
    
    # Style the dataframe
    styled_df = recent_data.style.format({
        'Close': '${:.2f}',
        f'MA_{strategy.short_window}': '${:.2f}',
        f'MA_{strategy.long_window}': '${:.2f}',
        'Portfolio_Value': '${:,.2f}'
    }).background_gradient(subset=['Portfolio_Value'], cmap='RdYlGn')
    
    st.dataframe(styled_df, use_container_width=True)

def display_export_options(strategy, ticker):
    """Display export options for results and reports."""
    
    st.subheader("📁 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Download Data")
        
        # Prepare data for export
        export_data = strategy.results[[
            'Close', f'MA_{strategy.short_window}', f'MA_{strategy.long_window}',
            'Portfolio_Value', 'Signal', 'Position', 'RSI', 'Volatility'
        ]].copy()
        
        export_data.columns = [
            'Close_Price', f'MA_{strategy.short_window}', f'MA_{strategy.long_window}',
            'Portfolio_Value', 'Signal', 'Position', 'RSI', 'Volatility'
        ]
        
        # CSV download
        csv_data = export_data.to_csv()
        st.download_button(
            label="📊 Download Strategy Data (CSV)",
            data=csv_data,
            file_name=f"{ticker}_strategy_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Trade history CSV
        if hasattr(strategy, 'trades') and len(strategy.trades) > 0:
            trades_csv = strategy.trades.to_csv(index=False)
            st.download_button(
                label="📋 Download Trade History (CSV)",
                data=trades_csv,
                file_name=f"{ticker}_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.markdown("#### 📄 Generate Report")
        
        # Generate comprehensive report
        report_text = strategy.generate_report()
        
        st.download_button(
            label="📄 Download Performance Report",
            data=report_text,
            file_name=f"{ticker}_performance_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # JSON export for programmatic use
        metrics = strategy.calculate_performance_metrics()
        import json
        
        json_data = {
            'ticker': ticker,
            'strategy_parameters': {
                'short_ma': strategy.short_window,
                'long_ma': strategy.long_window
            },
            'performance_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                                  for k, v in metrics.items()},
            'backtest_period': {
                'start': strategy.results.index[0].isoformat(),
                'end': strategy.results.index[-1].isoformat(),
                'total_days': len(strategy.results)
            }
        }
        
        st.download_button(
            label="🔧 Download Metrics (JSON)",
            data=json.dumps(json_data, indent=2),
            file_name=f"{ticker}_metrics_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Preview of the report
    st.markdown("#### 👀 Report Preview")
    with st.expander("View Performance Report"):
        st.text(strategy.generate_report())

def display_welcome_message():
    """Display welcome message and instructions."""
    
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Welcome to the Algorithmic Trading Dashboard</h3>
        <p>This professional-grade platform allows you to backtest and analyze moving average crossover trading strategies with comprehensive risk metrics and performance analytics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 How to Use This Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔧 Configure Strategy**
        - Select a stock ticker symbol
        - Choose historical data period
        - Set moving average parameters
        - Configure backtest settings
        """)
    
    with col2:
        st.markdown("""
        **📊 Analyze Results**
        - Review performance metrics
        - Examine interactive charts
        - Compare vs buy-and-hold
        - Assess risk measures
        """)
    
    with col3:
        st.markdown("""
        **📁 Export Data**
        - Download strategy results
        - Generate performance reports
        - Export trade history
        - Save metrics for analysis
        """)
    
    st.markdown("### 🎯 Strategy Overview")
    
    st.markdown("""
    The **Moving Average Crossover Strategy** is a trend-following approach that:
    
    - **Generates buy signals** when the short-term MA crosses above the long-term MA
    - **Generates sell signals** when the short-term MA crosses below the long-term MA
    - **Includes risk filters** using RSI to avoid extreme market conditions
    - **Accounts for transaction costs** and realistic trading constraints
    
    This implementation provides professional-grade backtesting with comprehensive risk metrics
    including Sharpe ratio, maximum drawdown, Value at Risk, and more.
    """)
    
    st.info("👆 Configure your strategy parameters in the sidebar and click 'Run Strategy Analysis' to begin!")

def generate_interpretation(metrics):
    """Generate interpretation text based on performance metrics."""
    
    total_return = metrics['Total Return']
    sharpe_ratio = metrics['Sharpe Ratio']
    max_drawdown = metrics['Maximum Drawdown']
    win_rate = metrics['Win Rate']
    
    interpretation = f"""
    <strong>Performance Analysis:</strong><br>
    """
    
    # Return analysis
    if total_return > 0.15:
        interpretation += "✅ <strong>Strong positive returns</strong> indicate the strategy outperformed significantly.<br>"
    elif total_return > 0:
        interpretation += "✅ <strong>Positive returns</strong> show the strategy was profitable.<br>"
    else:
        interpretation += "❌ <strong>Negative returns</strong> indicate the strategy underperformed.<br>"
    
    # Sharpe ratio analysis
    if sharpe_ratio > 1.5:
        interpretation += "✅ <strong>Excellent risk-adjusted returns</strong> (Sharpe > 1.5).<br>"
    elif sharpe_ratio > 1.0:
        interpretation += "✅ <strong>Good risk-adjusted returns</strong> (Sharpe > 1.0).<br>"
    elif sharpe_ratio > 0.5:
        interpretation += "⚠️ <strong>Moderate risk-adjusted returns</strong> (Sharpe > 0.5).<br>"
    else:
        interpretation += "❌ <strong>Poor risk-adjusted returns</strong> (Sharpe < 0.5).<br>"
    
    # Drawdown analysis
    if abs(max_drawdown) < 0.1:
        interpretation += "✅ <strong>Low maximum drawdown</strong> indicates good risk control.<br>"
    elif abs(max_drawdown) < 0.2:
        interpretation += "⚠️ <strong>Moderate maximum drawdown</strong> shows acceptable risk.<br>"
    else:
        interpretation += "❌ <strong>High maximum drawdown</strong> indicates significant risk exposure.<br>"
    
    # Win rate analysis
    if win_rate > 0.6:
        interpretation += f"✅ <strong>High win rate</strong> ({win_rate:.1%}) shows consistent performance.<br>"
    elif win_rate > 0.4:
        interpretation += f"⚠️ <strong>Moderate win rate</strong> ({win_rate:.1%}) is acceptable for trend-following strategies.<br>"
    else:
        interpretation += f"❌ <strong>Low win rate</strong> ({win_rate:.1%}) may indicate poor signal quality.<br>"
    
    return interpretation

if __name__ == "__main__":
    main()
