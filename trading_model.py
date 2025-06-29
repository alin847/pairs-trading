import pandas as pd
import numpy as np
from data import StockDatabase

class TradingModel(StockDatabase):
    """
    TradingModel for pairs trading strategy.
    
    This model implements a pairs trading strategy based on your selected pairs, 
    OLS coefficients, thresholds, and stop loss levels. The OLS coefficients should
    correspond to the linear regression of the log prices of the pairs (PERMNO1, PERMNO2)
    in the format (alpha, beta), where the log spread is calculated as:
        log_spread = log(PERMNO1) - alpha * log(PERMNO2) - beta
    The thresholds and stop loss levels are used to determine when to enter and exit positions.
    They should correspond to the levels of the log spread at which the model decides to enter 
    or exit trades. For example, if the threshold for a pair is 0.5, the model will enter a
    short position when the log spread exceeds 0.5.

    Attributes:
    pairs: list of tuples of PERMNO pairs to trade, e.g. [(PERMNO1, PERMNO2), ...]
    OLS_coeff: dict of OLS coefficients for each pair, e.g. {(PERMNO1, PERMNO2): (alpha, beta)}
    threshold: dictionary of thresholds for each pair, e.g. {(PERMNO1, PERMNO2): threshold}
    stop_loss: dictionary of stop loss levels for each pair, e.g. {(PERMNO1, PERMNO2): stop_loss}
    dollar_per_trade (float): amount of dollar exposure to trade per pair (by default is 1)
    """

    def __init__(self, pairs: list, OLS_coeff: dict, threshold: dict, stop_loss: dict, dollar_per_trade: float = 1):
        """
        Initializes the paramaters for the TradingModel by setting up the pairs, 
        OLS coefficients, thresholds, stop loss levels, and dollar exposure per trade.
        
        Parameters:
        pairs: list of tuples of PERMNO pairs to trade, e.g. [(PERMNO1, PERMNO2), ...]
        OLS_coeff: dict of OLS coefficients for each pair, e.g. {(PERMNO1, PERMNO2): (alpha, beta)}
        threshold: dictionary of thresholds for each pair, e.g. {(PERMNO1, PERMNO2): threshold}
        stop_loss: dictionary of stop loss levels for each pair, e.g. {(PERMNO1, PERMNO2): stop_loss}
        dollar_per_trade: amount of dollar exposure to trade per pair (by default is 1)
        """
        super().__init__()
        self.pairs = pairs
        self.unique_PERMNOs = set([PERMNO for pair in pairs for PERMNO in pair])
        self.OLS_coeff = OLS_coeff
        self.threshold = threshold
        self.stop_loss = stop_loss
        self.dollar_per_trade = dollar_per_trade

        # initialize positions & quantities for each pair
        # 0 for no position, 1 for long, -1 for short
        self.positions = {pair: 0 for pair in pairs}
        # number of shares to buy/sell for each PERMNO in the pair
        self.quantities = {pair: (0,0) for pair in pairs}


    def calc_spreads(self, close_prices: dict) -> dict:
        """
        Calculates the log spread for each pair based on the close prices.
        
        Parameters:
        close_prices: dict of close prices for the unique PERMNOs in all pairs. For example,
        {PERMNO1: close_price_1, PERMNO2: close_price_2, ...}
        
        Returns:
        spreads: dict of log spreads for each pair, e.g. {(PERMNO1, PERMNO2): spread}
        """
        spreads = {}
        for pair in self.pairs:
            PERMNO1, PERMNO2 = pair
            alpha, beta = self.OLS_coeff[pair]
            spread = np.log(close_prices[PERMNO1]) - alpha * np.log(close_prices[PERMNO2]) - beta
            spreads[pair] = spread
        return spreads
    

    def make_decisions(self, close_date: str) -> list:
        """
        Returns a list of decisions to take based on the current close prices.
        
        Parameters:
        close_date: date for which to make decisions, in the format 'YYYY-MM-DD'.

        Returns:
        decisions: a list of decisions in the format [((PERMNO1, PERMNO2), position), ...],
        where position = 1 to enter a long position, -1 to enter a short position, and 0 to
        exit a position.
        """
        # get the last close date and price
        last_close_dates = {}
        close_prices = {}
        for PERMNO in self.unique_PERMNOs:
            last_close_dates[PERMNO] = self.get_PERMNO(PERMNO).index[-2].strftime("%Y-%m-%d")

            data = self.get_metrics(PERMNO, ("C"), close_date, close_date)["DlyClose"]
            if data.empty:
                close_prices[PERMNO] = np.nan
            else:
                close_prices[PERMNO] = data.iloc[0]

        return self.make_decisions_helper(close_date, close_prices, last_close_dates)
    

    def trade(self, decisions: list, open_date: str) -> list:
        """
        Returns a list of trades based on the given decisions and how much to trade based
        on the current open prices.

        Parameters:
        decisions: list of decisions to take from make_decisions method in the format
        [((PERMNO1, PERMNO2), position), ...], where position = 1 to enter a long position,
        -1 to enter a short position, and 0 to exit a position.
        open_date: date for which to make trades, in the format 'YYYY-MM-DD'. 

        Returns:
        trades: list of trades in the format [(PERMNO, quantity), ...], where quantity is 
        the number of shares to buy/sell for PERMNO.
        """
        open_prices = {}
        for PERMNO in self.unique_PERMNOs:
            data = self.get_metrics(PERMNO, ("O"), open_date, open_date)["DlyOpen"]
            if data.empty:
                open_prices[PERMNO] = np.nan
            else:
                open_prices[PERMNO] = data.iloc[0]

        return self.trade_helper(decisions, open_prices)


    # HELPER
    def make_decisions_helper(self, close_date: str, close_prices: dict, last_close_dates: dict) -> list:
        """
        Helper method to make decisions based on the current close prices and last close dates.
        """
        decisions = []
        spreads = self.calc_spreads(close_prices)

        for pair in self.pairs:
            if pd.to_datetime(close_date) >= pd.to_datetime(last_close_dates[pair[0]]) or pd.to_datetime(close_date) >= pd.to_datetime(last_close_dates[pair[1]]):
                if self.positions[pair] != 0:
                    # exit position if one of the stocks is delisted
                    decisions.append((pair, 0))
                    self.positions[pair] = 0
                continue
            
            current_spread = spreads[pair]

            if self.positions[pair] == 1:
                if current_spread >= 0 or current_spread <= -self.stop_loss[pair]:
                    # Exit long position (either spread crosses zero or hits stop loss)
                    decisions.append((pair, 0))
                    self.positions[pair] = 0
    
            if self.positions[pair] == -1:
                if current_spread <= 0 or current_spread >= self.stop_loss[pair]:
                    # Exit short position (either spread crosses zero or hits stop loss)
                    decisions.append((pair, 0))
                    self.positions[pair] = 0
            
            if self.positions[pair] == 0:
                if current_spread > self.threshold[pair] and current_spread < self.stop_loss[pair]:
                    # Enter short position
                    decisions.append((pair, -1))
                    self.positions[pair] = -1
                    
                elif current_spread < -self.threshold[pair] and current_spread > -self.stop_loss[pair]:
                    # Enter long position
                    decisions.append((pair, 1))
                    self.positions[pair] = 1

        return decisions


    def trade_helper(self, decisions: list, open_prices: dict) -> list:
        """
        Helper method to execute trades based on the decisions and current open prices.
        """
        trades = []

        for pair, position in decisions:
            PERMNO1, PERMNO2 = pair
            
            if position == 0:
                # exit position
                trades.append((PERMNO1, -self.quantities[pair][0]))
                trades.append((PERMNO2, -self.quantities[pair][1]))
                self.quantities[pair] = (0, 0)  # reset quantities
            else:
                # enter long or short position
                ratio1 = 1 / (1 + self.OLS_coeff[pair][0])
                ratio2 = self.OLS_coeff[pair][0] / (1 + self.OLS_coeff[pair][0])
                quantity1 = (position * ratio1 * self.dollar_per_trade) / open_prices[PERMNO1]
                quantity2 = -(position * ratio2 * self.dollar_per_trade) / open_prices[PERMNO2]
                trades.append((PERMNO1, quantity1))
                trades.append((PERMNO2, quantity2))
                self.quantities[pair] = (quantity1, quantity2)
        return trades
    