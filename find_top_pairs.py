import pandas as pd
import numpy as np
from data import StockDatabase
import pandas_market_calendars as mcal
from statsmodels.tsa.stattools import coint

windows = pd.read_csv("windows.csv", parse_dates=["train_start", "train_end", "test_start", "test_end"])
sd = StockDatabase()

for index, row in windows.iterrows():
    # Step 1: preprocess the data to find top 1000 stocks
    start_date = row["train_start"].strftime("%Y-%m-%d")
    end_date = row["train_end"].strftime("%Y-%m-%d")
    test_date = row["test_start"].strftime("%Y-%m-%d")
    num_trading_days = len(mcal.get_calendar("NYSE").schedule(start_date=start_date, end_date=end_date))
    activePERMNOs = sd.get_active_PERMNOs(start_date, end_date)
    # print(f"Found {len(activePERMNOs)} active PERMNOs between {start_date} and {end_date}.")


    liquidPERMNOs = []
    for PERMNO in activePERMNOs:
        data = sd.get_metrics(PERMNO, ("Cap", "C"), start_date, end_date)
        cap = data["DlyCap"]
        close = data["DlyClose"]
        if len(close) == num_trading_days and not close.isna().any():
            liquidPERMNOs.append((PERMNO, cap.mean()))
    # pick top 1000 liquid stocks by average market cap
    liquidPERMNOs = sorted(liquidPERMNOs, key=lambda x: x[1], reverse=True)[:1000]
    liquidPERMNOs = [x[0] for x in liquidPERMNOs]  # extract PERMNOs only


    # Step 2: find pairs to test by correlation
    def get_pairs_by_correlation(PERMNOs, log_prices, threshold):
        """
        Returns pairs of PERMNOs that have a correlation above the given threshold.

        Parameters:
        PERMNOs : list of PERMNOs
        log_prices : DataFrame of log prices for the given PERMNOs
        threshold : float, correlation threshold
        """
        pairs = []
        correlation_matrix = log_prices[PERMNOs].corr()
        for i in range(len(PERMNOs)):
            for j in range(i + 1, len(PERMNOs)):
                if correlation_matrix.iloc[i, j] > threshold:
                    pairs.append((PERMNOs[i], PERMNOs[j]))
        return pairs

    log_price_list = []
    for PERMNO in liquidPERMNOs:
        data = sd.get_metrics(PERMNO, ("C"), start_date, end_date)
        log_price_list.append(np.log(data["DlyClose"]).rename(PERMNO))
    log_prices = pd.concat(log_price_list, axis=1)
    pairs = get_pairs_by_correlation(liquidPERMNOs, log_prices, 0.95)
    print(f"Found {len(pairs)} pairs with correlation above threshold.")


    # Step 3: Cointegration test
    cointegrated_pairs = {}
    for PERMNO1, PERMNO2 in pairs:
        _, p_value, _ = coint(log_prices[PERMNO1], log_prices[PERMNO2])
        if p_value < 0.05:
            cointegrated_pairs[(PERMNO1, PERMNO2)] = {"p_value": p_value}
    

    # Step 4: Choose top 20 pairs
    sorted_pairs = sorted(cointegrated_pairs.items(), key=lambda x: x[1]["p_value"], reverse=False)[:20]
    top_pairs = {pair: metrics for pair, metrics in sorted_pairs}
    
    
    # Step 5: Calculate alpha, beta, and spread std
    def calc_spread(PERMNO1, PERMNO2, log_prices):
        """
        returns the log spread between the two PERMNOs over the given date range, alpha, and beta. 
        PERMNO1 is y and PERMNO2 is x in the regression model.
        
        The spread is calculated as:
        spread = y - alpha * x - beta
        where alpha and beta are the coefficients from the OLS regression.
        """
        y = log_prices[PERMNO1]
        x = log_prices[PERMNO2]
        
        # OLS regression to find the spread
        alpha, beta = np.polyfit(x, y, 1)
        spread = y - alpha * x - beta
        return spread, alpha, beta

    for (PERMNO1, PERMNO2) in top_pairs.keys():
        spread, alpha, beta = calc_spread(PERMNO1, PERMNO2, log_prices)
        spread_sd = np.std(spread)
        top_pairs[(PERMNO1, PERMNO2)].update({"alpha": alpha, "beta": beta, "spread_sd": spread_sd})


    # Step 6: Save the pairs and their metrics to a csv file
    df = pd.DataFrame.from_dict(top_pairs, orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index, names=["PERMNO1", "PERMNO2"])
    df.to_csv(f"results/top_pairs/top_pairs_for_{test_date}.csv")
