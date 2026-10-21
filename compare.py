import pandas as pd
from main import MovingAverageCrossoverStrategy

def run(delay):
    s = MovingAverageCrossoverStrategy(20, 50)
    s.data = pd.read_csv("appl_2y.csv", index_col=0, parse_dates=True)    
    s.calculate_technical_indicators()
    s.generate_signals(delay=delay)
    s.backtest_strategy(initial_capital=100000, transaction_cost=0.001)
    pv = s.results['Portfolio_Value']
    ret = pv.iloc[-1] / 100000 - 1
    mdd = (pv / pv.cummax() - 1).min()
    return s, ret, mdd

for delay in (0, 1):
    s, ret, mdd = run(delay)
    print(f"delay={delay}:  return {ret:+.2%}   trades {len(s.trades)}   max drawdown {mdd:.2%}")
    print(s.trades[["Date", "Type", "Price", "Shares"]].to_string(), "\n")