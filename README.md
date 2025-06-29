# Pairs Trading Project

## Description
A project analyzing the profitability and risk associated with a simple pairs trading strategy from 2020 to 2024. 
For information about the methods, results, and conclusion of this project, please refer to [this paper](pairs_trading.pdf).

## File Descriptions

- **[result](results/)**  
  Folder that contains all the results from the analysis of the pairs trading strategy. It includes:
  - [Monthly returns](results/monthly_returns.csv)
  - [1% Value at Risk](results/VaR_01.csv) and [5% Value at Risk](results/VaR_05.csv)
  - [Top-20 pairs](results/top_pairs/)
  - [Account’s total capital history](results/capital_history/): the history of the account's total value during the trading period
  - [Account’s asset history](results/asset_history): the history of the account's value broken down by individual stocks during the trading period
  - [Account’s transaction history](results/transaction_history/): the history of all the transactions in the account during the trading period

- **[data.py](data.py)**  
  The database for ​​the Historical Daily Time Series Stock Market Data of NYSE, NYSE American, and NASDAQ exchanges.

- **[data](data/)**  
  Raw `.csv` files from CRSP (not included as per CRSP’s proprietary data policy)

- **[find_top_pairs.py](find_top_pairs.py)**  
  Script to identify the top-20 cointegrated pairs.

- **[simulate_pairs.py](simulate_pairs.py)**  
  Script to backtest the pairs trading strategy with the top-20 cointegrated pairs selected.

- **[VaR_analysis.py](VaR_analysis.py)**  
  Script to analyze the pairs trading value at risk profile.

- **[simulation.py](simulation.py)**  
  A class modeling a trading account that can hold assets, make transactions, and liquidate assets based on open prices.

- **[trading_model.py](trading_model.py)**  
  A class that implements the pairs trading strategy by making decisions using close prices.

- **[simulation_test.py](simulation_test.py)** and **[trading_model_test.py](trading_model_test.py)**  
  Test files for their respective classes.
