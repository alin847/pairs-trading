import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from data import StockDatabase


class Account(StockDatabase):
    """
    A class modeling a trading account that can hold assets, make transactions, and liquidate assets.
    All functionalities are made on the open prices of the market. 
    So far, it supports: all NYSE, NYSE American, and NASDAQ stock exchanges (2010-01-01 to 2024-12-31), 
    buying and selling stocks, and shorting stocks (does not consider transaction fees).

    Methods:
    update_date(date: str) -> None:
        Updates the date of the account to the given date.
    make_transaction(transactions: list) -> None:
        Makes a transaction given a list of transactions as a tuple of (PERMNO, quantity).
    liquidate() -> None:
        Liquidates all assets in the account on the current date.
    update_capital() -> None:
        Updates the capital and history based on current date. Always call every day after making
        trasactions/liquidating assets.
    get_transaction_history() -> pd.DataFrame:
        Returns the transaction history as a pandas DataFrame.
    get_capital_history() -> pd.DataFrame:
        Returns the capital history as a pandas DataFrame.
    """
    def __init__(self, start_date, buying_power):
        super().__init__()
        self.date = start_date
        self.buying_power = buying_power
        self.assets = {}
        self.capital = buying_power
        self.transaction_history = []
        self.capital_history = []
        self.asset_history = []


    def update_date(self, date: str) -> None:
        # check if date is moving forward
        if pd.to_datetime(date) < pd.to_datetime(self.date):
            raise ValueError(f"Date {date} must be after the current date {self.date}.")
        
        # check if date is a trading day
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=date, end_date=date)
        if schedule.empty:
            raise ValueError(f"{date} is not a trading day.")
        
        self.date = date
    

    def make_transaction(self, transactions: list, negative_balance = False) -> None:
        """
        Makes a transaction given a list of transactions. Takes in a list of tuples like
        [(PERMNO, quantity), (PERMNO, quantity), ...] where quantity can be negative for selling.
        If negative_balance is True, allows the buying power to go negative.
        """
        for PERMNO, quantity in transactions:
            price = self.get_price(PERMNO)

            # updating buying power and assets
            self.buying_power -= quantity * price    
            if PERMNO in self.assets:
                self.assets[PERMNO] += quantity
                if np.isclose(self.assets[PERMNO], 0, atol=1e-10):
                    del self.assets[PERMNO]
            else:
                self.assets[PERMNO] = quantity
            
            # record transaction
            self.transaction_history.append({"timestamp": self.date,
                                             "PERMNO": PERMNO,
                                             "quantity": quantity,
                                             "price": price})
    
        
        # check if buying power is still positive, otherwise bad transaction
        if not negative_balance and self.buying_power < 0:
            raise ValueError(f"Buying power is negative after transaction: {self.buying_power}")


    def liquidate(self, negative_balance = False) -> None:
        """
        Liquidates all assets in the account based on current asset prices, 
        converting them to cash. If negative_balance is True, allows the buying power to go negative.
        """
        for PERMNO, quantity in self.assets.items():
            price = self.get_price(PERMNO)
            self.buying_power += quantity * price
            self.transaction_history.append({"timestamp": self.date,
                                             "PERMNO": PERMNO,
                                             "quantity": -quantity,
                                             "price": price})
        # clear assets
        self.assets.clear()

        if not negative_balance and self.buying_power < 0:
            raise ValueError(f"Buying power is negative after liquidation: {self.buying_power}")
        

    def update_capital(self) -> None:
        """
        Updates the capital and histories based on the current assets price and buying power.
        """
        asset_value = 0
        for PERMNO, quantity in self.assets.items():
            price = self.get_price(PERMNO)
            value = quantity * price
            asset_value += value
            self.asset_history.append({"timestamp": self.date, "PERMNO": PERMNO, "quantity": quantity, "value": value})
        self.asset_history.append({"timestamp": self.date, "PERMNO": "CASH", "quantity": 1, "value": self.buying_power})
        self.capital = self.buying_power + asset_value
        self.capital_history.append({"timestamp": self.date, "capital": self.capital})


    def get_transaction_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.transaction_history)
    

    def get_capital_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.capital_history)
    

    def get_asset_history(self) -> pd.DataFrame:
        """
        Returns the asset history as a pandas DataFrame.
        """
        return pd.DataFrame(self.asset_history)


    def calc_total_return(self) -> float:
        """Calculates the return of the account from start to end date"""
        return (self.capital_history[-1]["capital"] - self.capital_history[0]["capital"]) / self.capital_history[0]["capital"]


    # HELPER
    def get_price(self, PERMNO: int) -> float:
        """
        Returns the price of the PERMNO on the current date.
        """
        data = self.get_metrics(PERMNO, ("O"), self.date, self.date)

        if data.empty or pd.isna(data.iloc[0, 0]):
            raise ValueError(f"Not a trading day for {PERMNO} on {self.date}")
        return data.iloc[0, 0]
    
