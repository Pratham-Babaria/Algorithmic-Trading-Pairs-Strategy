import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
import yfinance as yf
from scipy.optimize import minimize

start_date = datetime.datetime(2018, 1, 1)
end_date = datetime.datetime(2023, 7, 1)
stock_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Replace with your desired stock symbols

df = yf.download(stock_symbols, start=start_date, end=end_date)['Adj Close']
df = df.dropna()  # Remove any missing values

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            stock1 = data[keys[i]]
            stock2 = data[keys[j]]
            result = sm.OLS(stock1, stock2).fit()
            score = result.params[0]
            score_matrix[i, j] = score
            pvalue = sm.tsa.stattools.adfuller(result.resid)[1]
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(df)



def calculate_spread(pair, data):
    stock1 = data[pair[0]]
    stock2 = data[pair[1]]
    spread = stock1 - stock2
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std  # Normalize the spread
    return zscore, spread_mean, spread_std

spreads = {}
for pair in pairs:
    zscore, spread_mean, spread_std = calculate_spread(pair, df)
    spreads[pair] = (zscore, spread_mean, spread_std)

threshold = 1.0  # Example threshold for entry/exit
stop_loss = 2.0  # Example stop-loss threshold

capital = 100000  # Starting capital
position = {}  # Track position for each pair
trades = []  # Track trade history

def calculate_portfolio_value(position, df):
    portfolio_value = 0
    for pair in position:
        stock1_price = df.loc[pair['exit_date'], pair['stock1']]
        stock2_price = df.loc[pair['exit_date'], pair['stock2']]
        portfolio_value += pair['quantity'] * (stock1_price + stock2_price)
    return portfolio_value

for date in df.index:
    for pair in pairs:
        zscore, spread_mean, spread_std = spreads[pair]
        stock1_price = df.loc[date, pair[0]]
        stock2_price = df.loc[date, pair[1]]

        if zscore.loc[date] > threshold and pair not in position:
            pair_spread = stock1_price - stock2_price
            quantity = capital // (2 * pair_spread)  # Allocate equal capital to each pair
            position[pair] = {'entry_date': date, 'stock1': pair[0], 'stock2': pair[1],
                              'stock1_price': stock1_price, 'stock2_price': stock2_price,
                              'quantity': quantity}

        if pair in position and (zscore.loc[date] < threshold or zscore.loc[date] > stop_loss):
            entry_price = position[pair]['stock1_price'] - position[pair]['stock2_price']
            exit_price = stock1_price - stock2_price
            profit = (exit_price - entry_price) * position[pair]['quantity']
            trades.append({'entry_date': position[pair]['entry_date'], 'entry_price': entry_price,
                           'exit_date': date, 'exit_price': exit_price, 'profit': profit})
            del position[pair]


portfolio_value = calculate_portfolio_value(trades, df)
roi = (portfolio_value - capital) / capital
profits = [trade['profit'] for trade in trades]
max_drawdown = min(profits) / capital
sharpe_ratio = (roi - 0.05) / np.std(profits)


def objective(weights):
    return -np.mean(profits) / np.std(profits)


constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

result = minimize(objective, x0=np.ones(len(pairs)) / len(pairs), method='SLSQP', constraints=constraints)
optimal_weights = result.x

print(f"ROI: {roi:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Optimal Weights: {optimal_weights}")
