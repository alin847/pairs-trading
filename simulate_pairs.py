import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from trading_model import TradingModel
from simulation import Account


nyse = mcal.get_calendar("NYSE")
windows = pd.read_csv("windows.csv")
returns = []


for i in range(len(windows)):
    print("Processing window:", i+1, "of", len(windows))
    train_start, train_end, test_start, test_end = windows.iloc[i]

    # Load the top pairs for the given test start date
    df = pd.read_csv(f"results/top_pairs/top_pairs_for_{test_start}.csv", index_col=[0,1])
    
    # Initialize the trading model
    pairs = df.index.tolist()
    OLS_coeff = {}
    threshold = {}
    stop_loss = {}
    for PERMNO1, PERMNO2 in pairs:
        OLS_coeff[(PERMNO1, PERMNO2)] = df.loc[(PERMNO1, PERMNO2), ['alpha', 'beta']].values
        spread_sd = df.loc[(PERMNO1, PERMNO2), 'spread_sd']
        threshold[(PERMNO1, PERMNO2)] = 2 * spread_sd
        stop_loss[(PERMNO1, PERMNO2)] = 4 * spread_sd

    model = TradingModel(pairs, OLS_coeff, threshold, stop_loss)

    # simulate trading for a month
    schedule = nyse.schedule(start_date=train_end, end_date=test_end)
    trading_days = schedule.index.strftime("%Y-%m-%d").tolist()

    account = Account(trading_days[0], 20)
    unique_trades = set()
    for i in range(1, len(trading_days)-1):  
        previous_day = trading_days[i-1]
        current_day = trading_days[i]

        # make decision based on previous day's closing prices
        decisions = model.make_decisions(previous_day)
        # update unique trades
        unique_trades.update([pair for pair, position in decisions if position != 0])

        # make trades on current day opening prices
        trades = model.trade(decisions, current_day)
        
        # update account with trades
        account.update_date(current_day)
        account.make_transaction(trades, negative_balance=True)
        account.update_capital()
    
    # liquidate on last day
    account.update_date(trading_days[-1])
    account.liquidate(negative_balance=True)
    account.update_capital()

    # save results
    returns.append((20*account.calc_total_return())/len(unique_trades) if len(unique_trades) > 0 else 0)
    asset_history = account.get_asset_history()
    asset_history.to_csv(f"results/asset_history/asset_history_for_{test_start}.csv")
    transaction_history = account.get_transaction_history()
    transaction_history.to_csv(f"results/transaction_history/transaction_history_for_{test_start}.csv")
    capital_history = account.get_capital_history()
    capital_history.to_csv(f"results/capital_history/capital_history_for_{test_start}.csv")


# Save the results to DataFrames
test_start_dates = windows['test_end']
returns_df = pd.DataFrame(returns, index=test_start_dates, columns=["Returns"])
returns_df.to_csv("results/monthly_returns.csv")
