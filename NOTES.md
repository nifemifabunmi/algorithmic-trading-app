\# Notes



\## Max drawdown

What it measures, and what it is measured from:



My AAPL example (the portfolio peaked at about $101,487 during the first trade; what happened to the drawdown figure afterwards?):



\## Daily vs log returns

Simple return (P\_t / P\_{t-1} - 1) vs log return (ln(P\_t / P\_{t-1})): A price goes from 100 up to 110, then back down to 100. You end up exactly where you started.



The rise is +10% (10 is 10% of 100).

The fall is -9.09%, not -10%, because the fall is measured from 110, a bigger starting number.

Add the two percentages and you get +0.9%, even though you made no money. Simple percentage returns don’t add up over time.



Log returns fix this. The rise is +0.0953 and the fall is -0.0953, so they add to exactly zero. That’s why quants use them when they add returns across many days.



When would I prefer one over the other?



\## Sharpe ratio (optional)

Why the report's version came out lower than the standard version:

