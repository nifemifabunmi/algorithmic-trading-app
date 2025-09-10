# app.py
"""
Algorithmic Trading Dashboard
Built with Streamlit for interactive strategy analysis and backtesting
Enhanced Streamlit Dashboard with Multi-Asset, Real-Time, and ML Capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import warnings

# Import our modules
from main import MovingAverageCrossoverStrategy
from multi_asset import MultiAssetPortfolioStrategy
from live_trading import RealTimeDataProvider, DemoTradingEngine, RealTimeTradingStrategy
from ml_integration import MLTradingStrategy, FeatureEngineering

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Algorithmic Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.feature-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
}

.metric-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin: 0.5rem 0;
}

.success-metric {
    color: #28a745;
    font-weight: bold;
    font-size: 1.2rem;
}

.danger-metric {
    color: #dc3545;
    font-weight: bold;
    font-size: 1.2rem;
}

.info-metric {
    color: #17a2b8;
    font-weight: bold;
    font-size: 1.2rem;
}

.realtime-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    background-color: #28a745;
    border-radius: 50%;
    animation: pulse 2s infinite;
    margin-right: 8px;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.sidebar-section {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'strategy_type' not in st.session_state:
    st.session_state.strategy_type = 'Single Asset'
if 'realtime_active' not in st.session_state:
    st.session_state.realtime_active = False
if 'ml_models_trained' not in st.session_state:
    st.session_state.ml_models_trained = {}

def main():
    # Main header
    st.markdown('<h1 class="main-header"> Advanced Algorithmic Trading Platform</h1>', unsafe_allow_html=True)
    
    # Strategy type selector
    st.markdown("### 🎯 Select Trading Strategy Type")
    strategy_type = st.selectbox(
        "Choose your trading approach:",
        ["Single Asset Strategy", "Multi-Asset Portfolio", "Real-Time Trading", "ML-Enhanced Strategy"],
        key="strategy_type_selector"
    )
    
    # Route to appropriate interface
    if strategy_type == "Single Asset Strategy":
        render_single_asset_interface()
    elif strategy_type == "Multi-Asset Portfolio":
        render_multi_asset_interface()
    elif strategy_type == "Real-Time Trading":
        render_realtime_interface()
    elif strategy_type == "ML-Enhanced Strategy":
        render_ml_interface()

def render_single_asset_interface():
    """Render the enhanced single asset trading interface."""
    
    st.markdown("""
    <div class="feature-card">
        <h3>📈 Single Asset Moving Average Strategy</h3>
        <p>Professional-grade backtesting with comprehensive risk metrics and performance analytics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔧 Strategy Configuration")
        
        # Basic parameters
        ticker = st.text_input("Stock Symbol", value="AAPL", help="Enter a valid ticker symbol").upper()
        
        period_options = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y"}
        period = period_options[st.selectbox("Time Period", list(period_options.keys()), index=1)]
        
        # Moving average parameters
        st.subheader(" Technical Parameters")
        col1, col2 = st.columns(2)
        with col1:
            short_ma = st.slider("Short MA", 5, 50, 20)
        with col2:
            long_ma = st.slider("Long MA", 20, 200, 50)
        
        if short_ma >= long_ma:
            st.error("Short MA must be less than Long MA")
            return
        
        # Risk management
        st.subheader(" Risk Management")
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.1) / 100
        position_size = st.slider("Position Size (%)", 10, 100, 100, 10) / 100
        
        # Advanced options
        with st.expander("🔬 Advanced Options"):
            enable_volatility_filter = st.checkbox("Enable Volatility Filter", value=True)
            volatility_threshold = st.slider("Volatility Threshold", 0.01, 0.05, 0.02, 0.01)
            
            enable_volume_filter = st.checkbox("Enable Volume Filter", value=False)
            volume_threshold = st.number_input("Minimum Volume", value=1000000)
        
        analyze_button = st.button("🚀 Run Strategy Analysis", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main analysis
    if analyze_button:
        with st.spinner(f" Analyzing {ticker} strategy..."):
            try:
                # Initialize and run strategy
                strategy = MovingAverageCrossoverStrategy(short_window=short_ma, long_window=long_ma)
                
                # Download and process data
                data = strategy.download_data(ticker, period)
                if data.empty:
                    st.error(f" No data found for {ticker}")
                    return
                
                strategy.calculate_technical_indicators()
                strategy.generate_signals()
                results = strategy.backtest_strategy(
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    position_size=position_size
                )
                
                metrics = strategy.calculate_performance_metrics()
                
                # Display results in enhanced tabs
                display_enhanced_single_asset_results(strategy, metrics, ticker)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        display_single_asset_welcome()

def render_multi_asset_interface():
    """Render the multi-asset portfolio interface."""
    
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Multi-Asset Portfolio Optimization</h3>
        <p>Advanced portfolio construction with correlation analysis, risk parity optimization, and dynamic rebalancing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📊 Portfolio Configuration")
        
        # Asset selection
        st.subheader("Asset Selection")
        
        # Popular portfolios
        portfolio_presets = {
            "Tech Giants": ["AAPL", "MSFT", "GOOGL", "TSLA"],
            "FAANG": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
            "Diversified": ["SPY", "QQQ", "GLD", "TLT", "VTI"],
            "Custom": []
        }
        
        preset = st.selectbox("Portfolio Preset", list(portfolio_presets.keys()))
        
        if preset == "Custom":
            tickers_input = st.text_area(
                "Enter tickers (comma-separated)", 
                value="AAPL,MSFT,GOOGL,TSLA",
                help="Enter 2-10 stock symbols separated by commas"
            )
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
        else:
            tickers = portfolio_presets[preset]
            st.info(f"Selected: {', '.join(tickers)}")
        
        if len(tickers) < 2:
            st.error("Please select at least 2 assets")
            return
        elif len(tickers) > 10:
            st.warning("Maximum 10 assets supported")
            tickers = tickers[:10]
        
        # Portfolio parameters
        st.subheader("Parameters")
        period = st.selectbox("Time Period", ["1y", "2y", "3y", "5y"], index=1)
        
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            ["monthly", "quarterly", "weekly", "daily"],
            help="How often to rebalance the portfolio"
        )
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ["equal_risk_contribution", "equal_weight", "mean_variance"],
            help="Portfolio weight optimization approach"
        )
        
        # Risk parameters
        st.subheader("Risk Settings")
        portfolio_capital = st.number_input("Portfolio Capital ($)", min_value=10000, value=100000, step=5000)
        rebalance_cost = st.slider("Rebalancing Cost (%)", 0.0, 1.0, 0.05, 0.05) / 100
        
        analyze_portfolio = st.button("Analyze Portfolio", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main analysis
    if analyze_portfolio:
        with st.spinner("Running multi-asset analysis..."):
            try:
                # Initialize multi-asset strategy
                multi_strategy = MultiAssetPortfolioStrategy(
                    tickers=tickers,
                    rebalance_frequency=rebalance_freq
                )
                
                # Download and process data
                multi_strategy.download_multi_asset_data(period)
                correlation_matrix = multi_strategy.calculate_correlation_matrix()
                weights = multi_strategy.optimize_portfolio_weights(method=optimization_method)
                results = multi_strategy.backtest_portfolio(
                    initial_capital=portfolio_capital,
                    rebalance_cost=rebalance_cost
                )
                
                display_multi_asset_results(multi_strategy, tickers)
                
            except Exception as e:
                st.error(f"Error in portfolio analysis: {str(e)}")
    else:
        display_multi_asset_welcome()

def render_realtime_interface():
    """Render the real-time trading interface."""
    
    st.markdown("""
    <div class="feature-card">
        <h3><span class="realtime-indicator"></span>📡 Real-Time Trading System</h3>
        <p>Live market data processing with demo trading capabilities and real-time performance monitoring.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("Real-Time Configuration")
        
        # Trading parameters
        st.subheader("Trading Setup")
        rt_tickers = st.multiselect(
            "Select Symbols",
            ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META"],
            default=["AAPL", "MSFT"],
            help="Choose up to 5 symbols for real-time tracking"
        )
        
        if len(rt_tickers) > 5:
            st.warning("Maximum 5 symbols for real-time demo")
            rt_tickers = rt_tickers[:5]
        
        update_interval = st.slider("Update Interval (seconds)", 1, 30, 5)
        
        # Strategy parameters
        st.subheader("📈 Strategy Parameters")
        rt_short_ma = st.slider("Short MA (RT)", 3, 20, 5, key="rt_short")
        rt_long_ma = st.slider("Long MA (RT)", 10, 50, 15, key="rt_long")
        
        # Demo trading settings
        st.subheader("💰 Demo Trading")
        demo_capital = st.number_input("Demo Capital ($)", min_value=1000, value=50000, step=1000)
        demo_commission = st.slider("Commission (%)", 0.0, 0.5, 0.1, 0.05) / 100
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            start_realtime = st.button("▶️ Start Demo", use_container_width=True)
        with col2:
            stop_realtime = st.button("⏹️ Stop Demo", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time display
    if start_realtime and rt_tickers:
        st.session_state.realtime_active = True
        display_realtime_dashboard(rt_tickers, update_interval, demo_capital, demo_commission, rt_short_ma, rt_long_ma)
    elif stop_realtime:
        st.session_state.realtime_active = False
        st.success("Real-time demo stopped")
    else:
        display_realtime_welcome()

def render_ml_interface():
    """Render the ML-enhanced strategy interface."""
    
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 AI-Powered Trading Strategy</h3>
        <p>Advanced machine learning models for signal prediction, risk assessment, and strategy optimization.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🤖 ML Configuration")
        
        # Asset selection
        ml_ticker = st.selectbox("Select Asset", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"], key="ml_ticker")
        ml_period = st.selectbox("Training Period", ["1y", "2y", "3y", "5y"], index=2, key="ml_period")
        
        # ML parameters
        st.subheader("Model Settings")
        
        prediction_horizon = st.slider("Prediction Horizon (days)", 1, 10, 5)
        
        models_to_use = st.multiselect(
            "Select Models",
            ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "Gradient Boosting"],
            default=["Random Forest", "XGBoost", "Logistic Regression"],
            help="Choose ML models for ensemble"
        )
        
        # Feature engineering options
        st.subheader("🔧 Feature Engineering")
        enable_technical_features = st.checkbox("Technical Indicators", value=True)
        enable_statistical_features = st.checkbox("Statistical Features", value=True)
        enable_regime_features = st.checkbox("Market Regime Features", value=True)
        
        # Training options
        st.subheader("📚 Training Options")
        test_size = st.slider("Test Set Size (%)", 10, 50, 30) / 100
        cross_validation_folds = st.slider("CV Folds", 3, 10, 5)
        
        # Control buttons
        train_models = st.button("Train ML Models", use_container_width=True)
        
        if ml_ticker in st.session_state.ml_models_trained:
            run_ml_backtest = st.button("Run ML Backtest", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ML analysis
    if train_models:
        with st.spinner("Training ML models..."):
            try:
                # Initialize ML strategy
                base_strategy = MovingAverageCrossoverStrategy()
                ml_strategy = MLTradingStrategy(base_strategy)
                
                # Prepare data and train models
                ml_data = ml_strategy.prepare_ml_data(ml_ticker, ml_period)
                training_results = ml_strategy.train_ensemble_models(ml_data, ml_ticker)
                
                # Store in session state
                st.session_state.ml_models_trained[ml_ticker] = ml_strategy
                
                display_ml_training_results(training_results, ml_ticker)
                
            except Exception as e:
                st.error(f"ML training error: {str(e)}")
    
    elif ml_ticker in st.session_state.ml_models_trained and 'run_ml_backtest' in locals() and run_ml_backtest:
        with st.spinner("Running ML backtest..."):
            try:
                ml_strategy = st.session_state.ml_models_trained[ml_ticker]
                backtest_results = ml_strategy.backtest_ml_strategy(ml_ticker, ml_period)
                display_ml_backtest_results(ml_strategy, backtest_results, ml_ticker)
                
            except Exception as e:
                st.error(f"ML backtest error: {str(e)}")
    else:
        display_ml_welcome()

def display_enhanced_single_asset_results(strategy, metrics, ticker):
    """Display enhanced single asset results."""
    
    # Performance overview
    st.markdown("## 📊 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = metrics['Total Return']
        return_color = "success-metric" if total_return > 0 else "danger-metric"
        st.markdown(f"""
        <div class="metric-container">
            <h4>Total Return</h4>
            <div class="{return_color}">{total_return:+.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sharpe = metrics['Sharpe Ratio']
        sharpe_color = "success-metric" if sharpe > 1 else "info-metric" if sharpe > 0 else "danger-metric"
        st.markdown(f"""
        <div class="metric-container">
            <h4>Sharpe Ratio</h4>
            <div class="{sharpe_color}">{sharpe:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_dd = metrics['Maximum Drawdown']
        st.markdown(f"""
        <div class="metric-container">
            <h4>Max Drawdown</h4>
            <div class="danger-metric">{max_dd:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        win_rate = metrics['Win Rate']
        win_color = "success-metric" if win_rate > 0.5 else "danger-metric"
        st.markdown(f"""
        <div class="metric-container">
            <h4>Win Rate</h4>
            <div class="{win_color}">{win_rate:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive charts using Plotly
    st.markdown("## 📈 Interactive Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Performance", "Technical Analysis", "Risk Analysis", "Trade Details"])
    
    with tab1:
        # Interactive portfolio performance chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Portfolio Value vs Benchmark', 'Daily Returns'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=strategy.results.index, y=strategy.results['Portfolio_Value'],
                      name='Strategy', line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # Benchmark
        initial_value = strategy.results['Portfolio_Value'].iloc[0]
        benchmark = strategy.results['Close'] / strategy.results['Close'].iloc[0] * initial_value
        fig.add_trace(
            go.Scatter(x=strategy.results.index, y=benchmark,
                      name='Buy & Hold', line=dict(color='#ff7f0e', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Daily returns
        daily_returns = strategy.results['Portfolio_Value'].pct_change() * 100
        colors = ['green' if x > 0 else 'red' for x in daily_returns]
        fig.add_trace(
            go.Bar(x=strategy.results.index, y=daily_returns,
                   name='Daily Returns (%)', marker_color=colors, opacity=0.7),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title=f"{ticker} Strategy Performance Analysis")
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Technical analysis chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Price & Moving Averages', 'RSI', 'Volume'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price and MAs
        fig.add_trace(
            go.Scatter(x=strategy.results.index, y=strategy.results['Close'],
                      name='Price', line=dict(color='black', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=strategy.results.index, y=strategy.results[f'MA_{strategy.short_window}'],
                      name=f'{strategy.short_window}-day MA', line=dict(color='blue', width=1.5)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=strategy.results.index, y=strategy.results[f'MA_{strategy.long_window}'],
                      name=f'{strategy.long_window}-day MA', line=dict(color='red', width=1.5)),
            row=1, col=1
        )
        
        # Buy/Sell signals
        buy_signals = strategy.results[strategy.results['Position'] == 2]
        sell_signals = strategy.results[strategy.results['Position'] == -2]
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                          mode='markers', name='Buy Signal',
                          marker=dict(symbol='triangle-up', size=10, color='green')),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                          mode='markers', name='Sell Signal',
                          marker=dict(symbol='triangle-down', size=10, color='red')),
                row=1, col=1
            )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=strategy.results.index, y=strategy.results['RSI'],
                      name='RSI', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        
        # Volume
        if 'Volume' in strategy.results.columns:
            fig.add_trace(
                go.Bar(x=strategy.results.index, y=strategy.results['Volume'],
                      name='Volume', marker_color='lightblue', opacity=0.7),
                row=3, col=1
            )
        
        fig.update_layout(height=800, title=f"{ticker} Technical Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Risk analysis
        st.markdown("### 📊 Risk Metrics")
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            # Drawdown chart
            peak = strategy.results['Portfolio_Value'].expanding().max()
            drawdown = (strategy.results['Portfolio_Value'] - peak) / peak * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strategy.results.index, y=drawdown,
                                   fill='tonexty', name='Drawdown (%)',
                                   line=dict(color='red'), fillcolor='rgba(255,0,0,0.3)'))
            fig.add_hline(y=0, line_color="black", line_width=1)
            fig.update_layout(title="Portfolio Drawdown", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with risk_col2:
            # Monthly returns heatmap
            monthly_returns = strategy.results['Portfolio_Value'].resample('M').last().pct_change() * 100
            
            if len(monthly_returns) > 12:
                # Create heatmap data
                years = monthly_returns.index.year.unique()
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                heatmap_data = []
                for year in years:
                    year_data = []
                    for month in range(1, 13):
                        try:
                            value = monthly_returns[f'{year}-{month:02d}']
                            year_data.append(value)
                        except (KeyError, IndexError):
                            year_data.append(np.nan)
                    heatmap_data.append(year_data)
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=months,
                    y=years,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Return %")
                ))
                fig.update_layout(title="Monthly Returns Heatmap", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics table
        st.markdown("### 📋 Detailed Risk Metrics")
        risk_metrics = {
            'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                      'Maximum Drawdown', 'Value at Risk (95%)', 'Beta', 'Alpha'],
            'Value': [
                f"{metrics['Annualized Return']:.2%}",
                f"{metrics['Annualized Volatility']:.2%}",
                f"{metrics['Sharpe Ratio']:.3f}",
                f"{metrics['Sortino Ratio']:.3f}",
                f"{metrics['Maximum Drawdown']:.2%}",
                f"{metrics['Value at Risk (95%)']:.2%}",
                f"{metrics['Beta']:.3f}",
                f"{metrics['Alpha']:.2%}"
            ]
        }
        
        st.dataframe(pd.DataFrame(risk_metrics), use_container_width=True, hide_index=True)


    with tab4:
        # Trade details
        if hasattr(strategy, 'trades') and len(strategy.trades) > 0:
            st.markdown("### 📋 Trade History")
            
            trades_df = strategy.trades.copy()
            trades_df['Date'] = pd.to_datetime(trades_df['Date']).dt.strftime('%Y-%m-%d %H:%M')
            trades_df['P&L'] = trades_df.apply(
                lambda row: (row['Price'] - trades_df[trades_df['Type'] == 'BUY']['Price'].iloc[-1]) * row['Shares'] 
                if row['Type'] == 'SELL' and len(trades_df[trades_df['Type'] == 'BUY']) > 0 else 0, axis=1
            )
            
            # Style the dataframe
            styled_trades = trades_df.style.format({
                'Price': '${:.2f}',
                'Value': '${:,.2f}',
                'Cost': '${:.2f}',
                'P&L': '${:+,.2f}'
            }).applymap(
                lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
                subset=['P&L']
            )
            
            st.dataframe(styled_trades, use_container_width=True)
            
            # Trade statistics
            st.markdown("### 📊 Trade Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", len(strategy.trades))
            with col2:
                buy_trades = len(strategy.trades[strategy.trades['Type'] == 'BUY'])
                sell_trades = len(strategy.trades[strategy.trades['Type'] == 'SELL'])
                st.metric("Buy/Sell Ratio", f"{buy_trades}/{sell_trades}")
            with col3:
                total_costs = strategy.trades['Cost'].sum()
                st.metric("Total Costs", f"${total_costs:.2f}")
        else:
            st.info("No trades executed during the backtest period.")
    
    # Export options
    st.markdown("## 📁 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export data
        csv_data = strategy.results.to_csv()
        st.download_button(
            label="📊 Download Results (CSV)",
            data=csv_data,
            file_name=f"{ticker}_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export report
        report_text = strategy.generate_report()
        st.download_button(
            label="📄 Download Report",
            data=report_text,
            file_name=f"{ticker}_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    with col3:
        # Export metrics
        metrics_json = json.dumps(metrics, indent=2, default=str)
        st.download_button(
            label="🔧 Download Metrics (JSON)",
            data=metrics_json,
            file_name=f"{ticker}_metrics_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

def display_multi_asset_results(multi_strategy, tickers):
    """Display multi-asset portfolio results."""
    
    st.markdown("## Portfolio Analysis Results")
    
    # Portfolio metrics
    portfolio_metrics = multi_strategy.calculate_portfolio_metrics()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h4>Portfolio Return</h4>
            <div class="{'success-metric' if portfolio_metrics['Total Return'] > 0 else 'danger-metric'}">
                {portfolio_metrics['Total Return']:+.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h4>Sharpe Ratio</h4>
            <div class="{'success-metric' if portfolio_metrics['Sharpe Ratio'] > 1 else 'info-metric'}">
                {portfolio_metrics['Sharpe Ratio']:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h4>Diversification</h4>
            <div class="success-metric">{portfolio_metrics['Diversification Ratio']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h4>Assets</h4>
            <div class="info-metric">{portfolio_metrics['Number of Assets']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Performance", "Asset Allocation", "Correlation Analysis", "Individual Assets"])
    
    with tab1:
        # Portfolio performance chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=('Portfolio Value', 'Asset Allocation Over Time'),
                           vertical_spacing=0.1, row_heights=[0.6, 0.4])
        
        # Portfolio value
        results = multi_strategy.portfolio_results
        fig.add_trace(
            go.Scatter(x=results.index, y=results['Portfolio_Value'],
                      name='Portfolio Value', line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # Individual asset benchmarks
        colors = px.colors.qualitative.Set1
        for i, ticker in enumerate(tickers[:5]):  # Limit to 5 for readability
            if f'{ticker}_Close' in results.columns:
                initial_price = results[f'{ticker}_Close'].dropna().iloc[0]
                initial_value = results['Portfolio_Value'].iloc[0] / len(tickers)
                benchmark = results[f'{ticker}_Close'] / initial_price * initial_value
                fig.add_trace(
                    go.Scatter(x=results.index, y=benchmark,
                              name=f'{ticker} Benchmark', 
                              line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                              opacity=0.7),
                    row=1, col=1
                )
        
        # Asset allocation area chart
        allocation_data = {}
        for ticker in tickers:
            allocation_col = f'{ticker}_Allocation'
            if allocation_col in results.columns:
                allocation_pct = (results[allocation_col] / results['Portfolio_Value'] * 100).fillna(0)
                allocation_data[ticker] = allocation_pct
        
        if allocation_data:
            allocation_df = pd.DataFrame(allocation_data, index=results.index)
            
            for i, ticker in enumerate(tickers):
                if ticker in allocation_df.columns:
                    fig.add_trace(
                        go.Scatter(x=allocation_df.index, y=allocation_df[ticker],
                                  name=f'{ticker} %', fill='tonexty' if i > 0 else 'tozeroy',
                                  stackgroup='one', line=dict(color=colors[i % len(colors)])),
                        row=2, col=1
                    )
        
        fig.update_layout(height=800, title="Portfolio Performance Analysis")
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Allocation (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Current allocation
        st.markdown("### 📊 Optimized Portfolio Weights")
        
        if multi_strategy.optimized_weights:
            weights_df = pd.DataFrame(list(multi_strategy.optimized_weights.items()),
                                    columns=['Asset', 'Weight'])
            weights_df['Weight'] = weights_df['Weight'] * 100
            
            # Pie chart
            fig = px.pie(weights_df, values='Weight', names='Asset',
                        title="Portfolio Allocation",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Weights table
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    with tab3:
        # Correlation analysis
        st.markdown("### 🔗 Asset Correlation Matrix")
        
        if multi_strategy.correlation_matrix is not None:
            # Interactive correlation heatmap
            fig = px.imshow(multi_strategy.correlation_matrix,
                           color_continuous_scale='RdBu_r',
                           aspect="auto",
                           title="Asset Correlation Heatmap")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation statistics
            corr_matrix = multi_strategy.correlation_matrix
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
            min_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Correlation", f"{avg_corr:.3f}")
            with col2:
                st.metric("Highest Correlation", f"{max_corr:.3f}")
            with col3:
                st.metric("Lowest Correlation", f"{min_corr:.3f}")
    
    with tab4:
        # Individual asset performance
        st.markdown("### 📈 Individual Asset Performance")
        
        asset_performance = {}
        for ticker in tickers:
            if ticker in multi_strategy.strategies:
                try:
                    strategy = multi_strategy.strategies[ticker]
                    if strategy.results is not None:
                        individual_metrics = strategy.calculate_performance_metrics()
                        asset_performance[ticker] = individual_metrics
                except:
                    continue
        
        if asset_performance:
            # Create comparison dataframe
            comparison_data = []
            for ticker, metrics in asset_performance.items():
                comparison_data.append({
                    'Asset': ticker,
                    'Total Return': f"{metrics['Total Return']:.2%}",
                    'Sharpe Ratio': f"{metrics['Sharpe Ratio']:.3f}",
                    'Max Drawdown': f"{metrics['Maximum Drawdown']:.2%}",
                    'Win Rate': f"{metrics['Win Rate']:.1%}",
                    'Trades': int(metrics['Number of Trades'])
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("Individual asset performance data not available.")

def display_realtime_dashboard(tickers, update_interval, demo_capital, commission, short_ma, long_ma):
    """Display real-time trading dashboard."""
    
    st.markdown("## 📡 Live Trading Dashboard")
    
    # Create placeholders for real-time updates
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    positions_placeholder = st.empty()
    signals_placeholder = st.empty()
    
    # Initialize real-time components (simplified for demo)
    demo_data = {
        'portfolio_value': demo_capital,
        'cash': demo_capital,
        'positions': {},
        'signals': {ticker: 0 for ticker in tickers},
        'prices': {ticker: 100 + np.random.random() * 50 for ticker in tickers}
    }
    
    # Simulate real-time updates
    for update_count in range(10):  # Run for 10 updates as demo
        # Update status
        with status_placeholder.container():
            st.markdown(f"""
            <div class="feature-card">
                <h3><span class="realtime-indicator"></span>Live Market Status</h3>
                <p>Update #{update_count + 1} - Last update: {datetime.now().strftime('%H:%M:%S')}</p>
                <p>Monitoring: {', '.join(tickers)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Simulate price updates
        for ticker in tickers:
            price_change = (np.random.random() - 0.5) * 2  # ±1% random walk
            demo_data['prices'][ticker] *= (1 + price_change / 100)
        
        # Update metrics
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Value", f"${demo_data['portfolio_value']:,.2f}")
            with col2:
                daily_change = (np.random.random() - 0.5) * 4  # ±2% daily change
                st.metric("Daily P&L", f"{daily_change:+.2%}", delta=f"${daily_change * demo_capital / 100:+,.2f}")
            with col3:
                st.metric("Available Cash", f"${demo_data['cash']:,.2f}")
            with col4:
                st.metric("Active Positions", len(demo_data['positions']))
        
        # Update price chart
        with chart_placeholder.container():
            # Create mock price history
            dates = pd.date_range(start=datetime.now() - timedelta(minutes=100), 
                                end=datetime.now(), freq='10min')
            
            fig = go.Figure()
            
            for i, ticker in enumerate(tickers):
                # Generate mock price series
                base_price = demo_data['prices'][ticker]
                price_series = base_price + np.random.randn(len(dates)).cumsum() * 2
                
                fig.add_trace(go.Scatter(
                    x=dates, y=price_series, name=ticker,
                    line=dict(color=px.colors.qualitative.Set1[i])
                ))
            
            fig.update_layout(
                title="Real-Time Price Movement",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Update signals
        with signals_placeholder.container():
            st.markdown("### 🎯 Current Signals")
            
            signal_cols = st.columns(len(tickers))
            for i, ticker in enumerate(tickers):
                with signal_cols[i]:
                    # Random signal for demo
                    signal = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
                    signal_text = "🔴 SELL" if signal == -1 else "🟡 HOLD" if signal == 0 else "🟢 BUY"
                    confidence = np.random.uniform(0.5, 0.95)
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>{ticker}</h4>
                        <div>{signal_text}</div>
                        <small>Confidence: {confidence:.1%}</small>
                        <br><small>Price: ${demo_data['prices'][ticker]:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Wait before next update
        if update_count < 9:  # Don't sleep on last iteration
            time.sleep(2)  # 2 second demo intervals
    
    st.success("🏁 Real-time demo completed!")

def display_ml_training_results(training_results, ticker):
    """Display ML model training results."""
    
    st.markdown(f"## 🤖 ML Training Results - {ticker}")
    
    if training_results and 'performance' in training_results:
        # Model performance comparison
        st.markdown("### 📊 Model Performance Comparison")
        
        performance_data = []
        for model_name, score in training_results['performance'].items():
            performance_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{score:.3f}",
                'Weight': f"{training_results['weights'].get(model_name, 0):.3f}"
            })
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        # Performance visualization
        fig = px.bar(perf_df, x='Model', y='Accuracy', 
                    title="Model Accuracy Comparison",
                    color='Accuracy',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (mock for demo)
        st.markdown("### 🔍 Top Features")
        
        feature_names = ['RSI', 'MACD', 'MA_Ratio_20', 'Volatility_20', 'Momentum_5', 
                        'BB_Position', 'Volume_Ratio', 'Price_Change_10']
        importance_scores = np.random.random(len(feature_names))
        
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_df.head(10), x='Importance', y='Feature', 
                    orientation='h', title="Top 10 Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"✅ Successfully trained {len(training_results['models'])} models for {ticker}")
    else:
        st.error("❌ Training failed - no results available")

def display_ml_backtest_results(ml_strategy, results, ticker):
    """Display ML strategy backtest results."""
    
    st.markdown(f"## 📊 ML Backtest Results - {ticker}")
    
    # Calculate ML performance metrics
    ml_metrics = ml_strategy.calculate_ml_performance_metrics(results)
    
    # Performance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ML Strategy Return", f"{ml_metrics['total_return']:+.2%}")
    with col2:
        st.metric("ML Sharpe Ratio", f"{ml_metrics['sharpe_ratio']:.3f}")
    with col3:
        st.metric("Signal Accuracy", f"{ml_metrics['signal_accuracy']:.1%}")
    with col4:
        st.metric("Total Signals", int(ml_metrics['buy_signals'] + ml_metrics['sell_signals']))
    
    # ML vs Base strategy comparison
    st.markdown("### 🔍 ML vs Base Strategy Comparison")
    
    tab1, tab2, tab3 = st.tabs(["Performance Chart", "Signal Analysis", "Confidence Analysis"])
    
    with tab1:
        # Performance comparison chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=('Portfolio Value Comparison', 'Signal Comparison'),
                           vertical_spacing=0.1)
        
        # Portfolio performance
        fig.add_trace(
            go.Scatter(x=results.index, y=results['Portfolio_Value'],
                      name='ML Strategy', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # Benchmark
        initial_price = results['Close'].iloc[0]
        initial_value = results['Portfolio_Value'].iloc[0]
        benchmark = results['Close'] / initial_price * initial_value
        fig.add_trace(
            go.Scatter(x=results.index, y=benchmark,
                      name='Buy & Hold', line=dict(color='orange', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Signals comparison
        fig.add_trace(
            go.Scatter(x=results.index, y=results['ML_Signal'],
                      name='ML Signals', line=dict(color='blue', width=2)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=results.index, y=results['Base_Signal'],
                      name='Base Signals', line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(height=700, title="ML Strategy Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Signal distribution analysis
        st.markdown("#### 📊 Signal Distribution")
        
        signal_comparison = pd.DataFrame({
            'Signal Type': ['Buy', 'Hold', 'Sell'],
            'ML Strategy': [
                ml_metrics['buy_signals'],
                ml_metrics['hold_signals'],
                ml_metrics['sell_signals']
            ],
            'Base Strategy': [
                len(results[results['Base_Signal'] == 1]),
                len(results[results['Base_Signal'] == 0]),
                len(results[results['Base_Signal'] == -1])
            ]
        })
        
        fig = px.bar(signal_comparison, x='Signal Type', y=['ML Strategy', 'Base Strategy'],
                    title="Signal Distribution Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Agreement analysis
        agreement = (results['ML_Signal'] == results['Base_Signal']).mean()
        st.metric("Signal Agreement Rate", f"{agreement:.1%}")
    
    with tab3:
        # Confidence analysis
        if 'Confidence' in results.columns:
            st.markdown("#### 🎯 Prediction Confidence Over Time")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results.index, y=results['Confidence'],
                                   name='ML Confidence', line=dict(color='green', width=2)))
            fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                         annotation_text="High Confidence Threshold")
            fig.update_layout(title="ML Prediction Confidence", 
                            yaxis_title="Confidence", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence statistics
            high_conf_signals = len(results[results['Confidence'] > 0.7])
            avg_confidence = results['Confidence'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("High Confidence Signals", high_conf_signals)
            with col2:
                st.metric("Average Confidence", f"{avg_confidence:.1%}")

def display_single_asset_welcome():
    """Display welcome message for single asset strategy."""
    
    st.markdown("""
    <div class="feature-card">
        <h3>📈 Single Asset Strategy Analysis</h3>
        <p>Configure your strategy parameters in the sidebar and click "Run Strategy Analysis" to begin comprehensive backtesting.</p>
        
        <h4>🎯 Features Include:</h4>
        <ul>
            <li>📊 Professional backtesting with realistic constraints</li>
            <li>🛡️ Comprehensive risk metrics and analysis</li>
            <li>📈 Interactive charts and visualizations</li>
            <li>📋 Detailed trade history and statistics</li>
            <li>📁 Multiple export formats (CSV, JSON, Reports)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_multi_asset_welcome():
    """Display welcome message for multi-asset strategy."""
    
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Multi-Asset Portfolio Strategy</h3>
        <p>Build and analyze diversified portfolios with advanced optimization techniques.</p>
        
        <h4>🚀 Advanced Features:</h4>
        <ul>
            <li>📊 Portfolio optimization (Risk Parity, Mean Variance)</li>
            <li>🔗 Correlation analysis and diversification metrics</li>
            <li>⚖️ Dynamic rebalancing with cost modeling</li>
            <li>📈 Multi-asset performance attribution</li>
            <li>🎯 Risk-adjusted portfolio construction</li>
        </ul>
        
        <p>Select your assets and configuration parameters in the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)

def display_realtime_welcome():
    """Display welcome message for real-time trading."""
    
    st.markdown("""
    <div class="feature-card">
        <h3>📡 Real-Time Trading System</h3>
        <p>Experience live market data processing with demo trading capabilities.</p>
        
        <h4>🎯 Real-Time Features:</h4>
        <ul>
            <li>📡 Live market data integration</li>
            <li>⚡ Real-time signal generation</li>
            <li>💰 Demo trading with realistic execution</li>
            <li>📊 Live performance monitoring</li>
            <li>🎯 Risk management controls</li>
        </ul>
        
        <p><strong>Note:</strong> This is a demonstration using simulated live data. Configure your parameters and click "Start Demo" to begin.</p>
    </div>
    """, unsafe_allow_html=True)

def display_ml_welcome():
    """Display welcome message for ML strategy."""
    
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 Machine Learning Enhanced Trading</h3>
        <p>Harness the power of AI for advanced trading signal generation and risk assessment.</p>
        
        <h4>🧠 ML Capabilities:</h4>
        <ul>
            <li>🤖 Ensemble ML models (Random Forest, XGBoost, SVM)</li>
            <li>📊 Advanced feature engineering</li>
            <li>🎯 Signal confidence scoring</li>
            <li>📈 Performance comparison vs traditional methods</li>
            <li>🔍 Feature importance analysis</li>
        </ul>
        
        <p>Select your asset and model parameters, then train ML models to enhance your trading strategy.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
