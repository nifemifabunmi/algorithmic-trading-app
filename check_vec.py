import pandas as pd
from main import MovingAverageCrossoverStrategy

def vectorised(s, initial_capital=100000, fee=0.001):

    d = s.data.dropna().sort_index()
    state = pd.Series(float('nan'), index=d.index)
    state [d['Position'] == 2] = 1.0  # buy signal -> holding
    state [d['Position'] == -2] = 0.0  # sell signal -> flat
    state = state.ffill().fillna(0.0)  # keep the last state until the next signal
    ret = d['Close'].pct_change().fillna(0.0)  # daily price change
    turnover = state.diff().abs().fillna(0.0)  # 1 on days we trade, else 0
    strat_ret = state.shift(1).fillna(0.0) * ret - turnover * fee
    return initial_capital * (1 + strat_ret).cumprod()

s = MovingAverageCrossoverStrategy(20, 50)
s.data = pd.read_csv("appl_2y.csv", index_col=0, parse_dates=True)
s.calculate_technical_indicators()
s.generate_signals()    #default delay is 1
s.backtest_strategy(initial_capital=100000, transaction_cost=0.001)

loop = s.results["Portfolio_Value"]
vec = vectorised(s)

print(f"loop:       {loop.iloc[-1] / 100000 - 1:+.2%}")
print(f"vectorised: {vec.iloc[-1] / 100000 - 1:+.2%}")
print(f"worst daily gap between the two curves: {(vec / loop - 1).abs().max():.3%}")