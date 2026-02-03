import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V", "UNH", "PG"]
BENCHMARK = "SPY"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
TOP_N = 3

def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data.dropna(axis=1)

prices = load_prices(TICKERS, START_DATE, END_DATE)
benchmark_prices = load_prices([BENCHMARK], START_DATE, END_DATE)

def momentum(prices, window=252):
    return prices.pct_change(window)

def volatility(prices, window=60):
    return prices.pct_change().rolling(window).std()

momentum_rank = momentum(prices).rank(axis=1, ascending=False)
vol_rank = volatility(prices).rank(axis=1, ascending=True)
scores = momentum_rank + vol_rank

def construct_portfolio(scores, top_n):
    weights = scores.apply(lambda x: (x <= top_n).astype(int), axis=1)
    return weights.div(weights.sum(axis=1), axis=0)

weights = construct_portfolio(scores, TOP_N)
returns = prices.pct_change()
strategy_returns = (weights.shift(1) * returns).sum(axis=1).dropna()
benchmark_returns = benchmark_prices.pct_change().dropna().squeeze()

df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
df.columns = ["Strategy", "Benchmark"]

def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    return ((cumulative - peak) / peak).min()

summary = pd.DataFrame({
    "Strategy": {
        "Annual Return": df["Strategy"].mean() * 252,
        "Annual Volatility": df["Strategy"].std() * np.sqrt(252),
        "Sharpe Ratio": sharpe_ratio(df["Strategy"]),
        "Max Drawdown": max_drawdown(df["Strategy"]),
    },
    "Benchmark": {
        "Annual Return": df["Benchmark"].mean() * 252,
        "Annual Volatility": df["Benchmark"].std() * np.sqrt(252),
        "Sharpe Ratio": sharpe_ratio(df["Benchmark"]),
        "Max Drawdown": max_drawdown(df["Benchmark"]),
    }
})

print(summary)

cumulative = (1 + df).cumprod()
cumulative.plot(title="Equity Curve: Strategy vs Benchmark")
plt.show()
