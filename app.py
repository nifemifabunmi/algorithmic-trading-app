import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from main import add_moving_averages, generate_signals, backtest_strategy, performance_metrics

# Use a pretty matplotlib style
style.use('ggplot')

# Set page config
st.set_page_config(
    page_title="algo trading dashboard ✨",
    page_icon="📈",
    layout="wide"
)

st.title("💖 Algorithmic Trading Strategy Dashboard 💖")

# Custom button styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ff69b4;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    transition: background-color 0.3s ease;
}
div.stButton > button:first-child:hover {
    background-color: #ff1493;
}
</style>
""", unsafe_allow_html=True)

# Select ticker and period
ticker = st.text_input("Enter stock ticker, babe:", value="AAPL")
period = st.selectbox("Select period (years):", options=["1y", "2y", "5y"], index=1)

# Customize MAs
st.markdown("#### 🎯 Customize Moving Averages")
short_ma = st.slider("Short MA (days)", 5, 50, 20)
long_ma = st.slider("Long MA (days)", 20, 200, 50)

if st.button("Download & Analyze 📊"):
    st.success(f"Downloading data for {ticker}...")
    data = yf.download(ticker, period=period)

    # Fix multiindex if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(filter(None, col)).strip() for col in data.columns.values]

    # Add MAs and signals
    data = add_moving_averages(data, short_ma, long_ma)
    data = generate_signals(data)
    data = backtest_strategy(data)

    # CSV download
    st.subheader("📥 Download Your Strategy Results")
    csv = data.to_csv().encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f'{ticker}_strategy_results.csv',
        mime='text/csv'
    )

    # Tabs for content
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Chart", "📊 Backtest Results"])

    with tab1:
        st.dataframe(data.tail().style.background_gradient(cmap='pink'))

    with tab2:
        st.line_chart(data["Close"])

    with tab3:
        st.subheader("💰 Portfolio Value Over Time")
        st.line_chart(data['Portfolio Value'])

        # Performance Metrics
        st.subheader("🚀 Performance Metrics")
        import io, sys
        buffer = io.StringIO()
        sys.stdout = buffer
        performance_metrics(data)
        sys.stdout = sys.__stdout__
        st.text(buffer.getvalue())

        # Price + MA Plot
        st.subheader("📈 Price & Moving Averages")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Close'], label='Close Price', color='#ff69b4', linewidth=2)
        ax.plot(data['MA20'], label='20-day MA', color='#ff1493', linestyle='--')
        ax.plot(data['MA50'], label='50-day MA', color='#c71585', linestyle='-.')
        ax.set_title(f"{ticker} Price & Moving Averages", fontsize=16, color='#d147a3')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.balloons()