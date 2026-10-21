# Algorithmic Trading App
 
A learning project for backtesting simple trading strategies in Python, with a Streamlit web interface. It exists to teach me quantitative finance, not to trade real money.
 
> **Not financial advice.** Backtests on historical data say little about future results, and the results below come from a single stock over about two years.
 
## Status
 
| Part | Status | Notes |
|---|---|---|
| Single-asset backtest (`main.py`) | Working and checked | 20/50-day moving-average crossover with an RSI filter on buys and sells, whole-share sizing, a 0.1% fee per trade, and fills at the next bar's close. Cross-checked against an independent vectorised calculation (`check_vec.py`). |
| Streamlit app, single-asset tab | Working | `streamlit run app.py` |
| Multi-asset portfolio (`multi_asset.py`) | Written, not yet verified | Risk-parity and mean-variance weights with periodic rebalancing. |
| Machine learning (`ml_integration.py`) | Experimental | Trains several classifiers on technical features. The validation has known flaws (overlapping targets, filled missing values, non-stationary features), so the accuracy figures should not be trusted. The "Top Features" chart in the app is placeholder data. |
| Real-Time tab | Demo only | The prices, P&L and signals on this tab are randomly generated, not market data, and no orders are placed. The data and order classes in `live_trading.py` are not connected to it. |
 
## Results so far
 
AAPL, about two years of daily data to 2026-10-21, 20/50-day crossover, 0.1% fee per trade, $100,000 starting capital.
 
| Version | Total return | Trades | Max drawdown |
|---|---|---|---|
| Fills at the signal day's close | +19.92% | 13 | -16.72% |
| Fills one bar later (current default) | **+16.25%** | 13 | -18.57% |
| Buy and hold | +42.09% | n/a | -30.22% |
 
What these numbers do and don't say:
 
- This is one stock and 13 trades, which is too little data to conclude anything about the strategy.
- There is no slippage model, and fills happen at the next bar's close rather than its open, so the result is still somewhat generous.
- Prices come from Yahoo Finance via `yfinance` and can be revised slightly between downloads, so the analysis scripts save a CSV (`appl_2y.csv`) and reuse it.

## What I've learned so far
 
- **A backtest can be quietly wrong.** The original code skipped a buy whenever the fee didn't fit in the leftover cash, and it skipped the matching sell too. Two of seven buys never happened, which turned a +19.9% result into -15.0%. Cross-checking against a second, independent calculation is how this kind of bug shows up.
- **Same-bar fills flatter results.** Acting one bar after the signal, as a real trader must, cut the return from +19.92% to +16.25% here.
- **Win rate alone misleads.** Only 2 of 6 closed trades won, yet the strategy made money, because the winners were much larger than the losers.
- **Drawdown is measured from the running peak,** not from the starting capital.
## Setup
 
Developed on Windows with Python 3.14 and pandas 3.0.
 
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```
 
## Run
 
```bash
python main.py          # text report and charts for AAPL
python explore.py       # trades, drawdowns, and a drawdown plot on saved data
python compare.py       # no delay vs one-bar delay
python check_vec.py     # loop backtest vs vectorised cross-check
streamlit run app.py    # web interface
```
 
`explore.py`, `compare.py` and `check_vec.py` read `appl_2y.csv`. `explore.py` downloads and saves it the first time it runs.
 
## Files
 
| File | Purpose |
|---|---|
| `main.py` | Strategy class: data download, indicators, signals, loop backtest, metrics, plots |
| `explore.py` | Trade list, volatility and drawdown comparison on saved data |
| `compare.py` | Compares results with and without the one-bar signal delay |
| `check_vec.py` | Independent vectorised backtest used to cross-check `main.py` |
| `app.py` | Streamlit interface |
| `multi_asset.py` | Portfolio weights, rebalancing and portfolio metrics |
| `ml_integration.py` | Feature engineering and ML models |
| `live_trading.py` | Data provider and simulated order engine (not wired into the app) |
 
## Known issues and next steps
 
- [ ] Standard Sharpe ratio (mean of daily excess returns) and a proper beta/alpha regression
- [ ] Wilder-smoothed RSI instead of a simple rolling mean
- [ ] A slippage model, and fills at the next open
- [ ] Walk-forward testing, and results across many tickers and periods
- [ ] Unit tests for indicators and metrics
- [ ] Fix ML validation (purged cross-validation, stationary features, no `fillna(0)`)
- [ ] Replace the Real-Time demo with real data, or label it clearly as simulated
- [ ] Unused sidebar controls in the single-asset tab (volatility and volume filters) do nothing yet
## Tools
 
Python, pandas, NumPy, SciPy, yfinance, Matplotlib, Plotly, scikit-learn, XGBoost, Streamlit.