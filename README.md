# Algorithmic-Trading-Pairs-Strategy
This repository contains the implementation of a pairs trading strategy using Python. The pairs trading strategy aims to identify pairs of stocks with a historically close relationship and take advantage of temporary divergences in their prices

# Pairs Trading Strategy

This repository contains an implementation of a pairs trading strategy using Python. The pairs trading strategy aims to identify pairs of stocks with a historically close relationship and take advantage of temporary divergences in their prices. The strategy involves data acquisition, pair selection using cointegration analysis, spread calculation, entry and exit rules, backtesting, risk management, and portfolio optimization.

## Features

- Data acquisition: Historical price data for a set of stocks is gathered using the Yahoo Finance API.
- Pair selection: Suitable pairs of stocks are identified using cointegration analysis.
- Spread calculation: The spread between the prices of the selected pairs is calculated and normalized.
- Entry and exit rules: Defined entry and exit rules determine the trading strategy.
- Backtesting and evaluation: The strategy is backtested using historical data, and performance metrics such as ROI and maximum drawdown are calculated.
- Risk management: Risk management techniques are implemented to manage portfolio exposure and control potential losses.
- Portfolio optimization: The portfolio weights are optimized using mean return to standard deviation ratio as the objective function.

## Usage

1. Install the required dependencies listed in the `requirements.txt` file using pip:

2. Customize the parameters and settings in the Python script `pairs_trading_strategy.py` to fit your needs.

3. Run the script

4. View the calculated performance metrics in the console output.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open issue or submit a pull request.




