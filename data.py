import pandas as pd


class StockDatabase():
    """
    Historical Daily Time Series Stock Market Data (OHLCV and Market Capitalization) of NYSE, NYSE American, and
    NASDAQ listed securities from 2010-01-01 to 2024-12-31.

    Methods:
    search_PERMNO(ticker: str, date: str) -> int:
        Returns the PERMNO of the ticker present on the given date. Raises ValueError if not found.
    get_active_PERMNOs(start_date: str, end_date: str) -> list:
        Returns a list of active PERMNOs between start_date and end_date.
    get_security_name(PERMNO: int) -> str:
        Returns the security name for the given PERMNO.
    get_metrics(PERMNO: int, metrics: tuple, start_date: str, end_date: str) -> pd.DataFrame:
        Returns the metrics (O/H/L/C/V/Cap) of the PERMNO from start_date to end_date.
    """
    def __init__(self):
        self.identifers = pd.read_csv("data/identifiers.csv")
        self.metrics = {}


    def search_PERMNO(self, ticker: str, date: str) -> int:
        """
        Returns the PERMNO (identifier) of the ticker on the given date.
        Returns None if can't find the ticker.
        """
        mask = (self.identifers["Ticker"] == ticker) & (self.identifers["SecurityBegDt"] <= date) & (self.identifers["SecurityEndDt"] >= date)
        row = self.identifers[mask]
        if row.empty:
            raise ValueError(f"Ticker {ticker} not found on date {date}")
        return row["PERMNO"].values[0]
  

    def range_PERMNO(self, PERMNO: int) -> tuple:
        """
        Returns the start and end dates of the PERMNO.
        """
        row = self.identifers.loc[self.identifers["PERMNO"] == PERMNO]
        if row.empty:
            raise ValueError(f"PERMNO {PERMNO} not found in the dataset.")
        return row["SecurityBegDt"].values[0], row["SecurityEndDt"].values[0]
    

    def get_active_PERMNOs(self, start_date: str, end_date: str) -> list:
        """
        Returns a list of active PERMNOs between start_date and end_date.
        """
        mask = (self.identifers["SecurityBegDt"] <= start_date) & (self.identifers["SecurityEndDt"] >= end_date)
        return self.identifers.loc[mask, "PERMNO"].tolist()


    def get_security_name(self, PERMNO: int) -> str:
        """
        Returns the security name for the given PERMNO.
        """
        row = self.identifers.loc[self.identifers["PERMNO"] == PERMNO]
        if row.empty:
            raise ValueError(f"PERMNO {PERMNO} not found in the dataset.")
        return row["SecurityNm"].values[0]


    def get_metrics(self, PERMNO: int, metrics: tuple, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Returns the metrics (O/H/L/C/V/Cap) of PERMNO from start_date to end_date (inclusive) as csv. 
        """
        KEY = {"O": "DlyOpen",
               "H": "DlyHigh",
               "L": "DlyLow",
               "C": "DlyClose",
               "V": "DlyVol",
               "Cap": "DlyCap"}
        
        data = self.get_PERMNO(PERMNO)
        return data.loc[start_date:end_date, [KEY[metric] for metric in metrics]]


    # HELPER
    def get_PERMNO(self, PERMNO: int) -> pd.DataFrame:
        """
        Return the OHLCVs csv for the PERMNO. If does not exist, then raises Error.
        """
        if PERMNO in self.metrics:
            return self.metrics[PERMNO]

        try:
            df = pd.read_csv(f"data/{PERMNO}.csv")
            df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"], format="%Y-%m-%d")
            df.set_index("DlyCalDt", inplace=True)
            df.sort_index(inplace=True)
            self.metrics[PERMNO] = df
        except FileNotFoundError:
            raise FileNotFoundError(f"The PERMNO {PERMNO} does not exist in the database.")
        
        return self.metrics[PERMNO]

