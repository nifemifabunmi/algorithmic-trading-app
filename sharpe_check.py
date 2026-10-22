import pandas as pd
from main import MovingAverageCrossoverStrategy

RF = 0.02          # annual risk-free rate, same as main.py
DAYS = 252


def sharpe_report_style(values):
    """What main.py does: compounded annual return over annualised volatility."""
    returns = values.pct_change().dropna()
    total = values.iloc[-1] / values.iloc[0] - 1
    years = len(returns) / DAYS
    annual_return = (1 + total) ** (1 / years) - 1
    annual_vol = returns.std() * DAYS ** 0.5
    return (annual_return - RF) / annual_vol


def sharpe_standard(values):
    """Standard: mean daily excess return over its std, scaled by sqrt(252)."""
    returns = values.pct_change().dropna()
    excess = returns - RF / DAYS
    return excess.mean() / excess.std() * DAYS ** 0.5


s = MovingAverageCrossoverStrategy(20, 50)
s.data = pd.read_csv("appl_2y.csv", index_col=0, parse_dates=True)
s.calculate_technical_indicators()
s.generate_signals()
s.backtest_strategy(initial_capital=100000, transaction_cost=0.001)

strategy = s.results["Portfolio_Value"]
buy_hold = s.results["Close"]

print(f"main.py report Sharpe:         {s.calculate_performance_metrics()['Sharpe Ratio']:.3f}")
print(f"my report-style replica:       {sharpe_report_style(strategy):.3f}   (should match the line above)")
print()
print(f"{'':16}{'report-style':>14}{'standard':>10}")
print(f"{'Strategy':16}{sharpe_report_style(strategy):>14.3f}{sharpe_standard(strategy):>10.3f}")
print(f"{'Buy & Hold':16}{sharpe_report_style(buy_hold):>14.3f}{sharpe_standard(buy_hold):>10.3f}")