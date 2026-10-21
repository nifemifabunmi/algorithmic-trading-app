import os
import pandas as pd
import matplotlib.pyplot as plt
from main import MovingAverageCrossoverStrategy

s = MovingAverageCrossoverStrategy(20, 50)

#to download once, and reuse so all runs are consistent with same data
if os.path.exists("appl_2y.csv"):
    s.data = pd.read_csv("appl_2y.csv", index_col=0, parse_dates=True)
else:
    s.download_data("AAPL", "2y") . to_csv("appl_2y.csv")

s.calculate_technical_indicators()
s.generate_signals()
s.backtest_strategy(initial_capital=100000, transaction_cost=0.001)

print(s.trades[['Date', 'Type', 'Price', 'Shares']])

first = s.results["Close"].iloc[0]
last = s.results["Close"].iloc[-1]
print(f"\nBuy & Hold: {last / first - 1 :.2%}")
print(f"First day return: {s.results.index[0].date()} | Last day return: {s.results.index[-1].date()}")

r = s.results["Portfolio_Value"].pct_change().dropna()
bh = s.results["Close"].pct_change().dropna()

def max_dd(values):
    return (values / values.cummax() - 1).min()

print(f"Strategy annualized vol: {r.std() * (252 ** 0.5):.2%}")
print(f"Buy & Hold annualized vol: {bh.std() * (252 ** 0.5):.2%}")
print(f"Strategy max drawdown: {max_dd(s.results['Portfolio_Value']):.2%}")
print(f"Buy & Hold max drawdown: {max_dd(s.results['Close']):.2%}")
print(f"Share of days holding no shares: {(s.results['Shares_Held'] == 0).mean(): .0%}")
print(s.results["Portfolio_Value"].max())

dd_strat = s.results["Portfolio_Value"] / s.results["Portfolio_Value"].cummax() - 1
dd_bh = s.results["Close"] / s.results["Close"].cummax() - 1

plt.figure(figsize=(10, 4))
plt.plot(dd_strat.index, dd_strat * 100, label="Strategy")
plt.plot(dd_bh.index, dd_bh * 100, label="Buy & Hold")
plt.ylabel("Drawdown (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()