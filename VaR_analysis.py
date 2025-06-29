import pandas as pd
import numpy as np
from data import StockDatabase
import pandas_market_calendars as mcal


# VaR Analysis   
def calculate_portfolio_variance(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """
    Calculate the portfolio variance given weights and covariance matrix.
    
    Parameters:
    weights (np.ndarray): Array of asset weights in the portfolio.
    cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
    
    Returns:
    float: Portfolio variance.
    """
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_VaR(weights: np.ndarray, cov_matrix: pd.DataFrame, alpha: float = 0.01) -> float:
    """
    Calculate the Value at Risk (VaR) of a portfolio.
    
    Parameters:
    weights (np.ndarray): Array of asset weights in the portfolio.
    cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
    alpha (float): Significance level for VaR calculation (default is 0.01).
    
    Returns:
    float: Value at Risk of the portfolio.
    """
    portfolio_variance = calculate_portfolio_variance(weights, cov_matrix)
    portfolio_std_dev = np.sqrt(portfolio_variance)
    
    # Calculate VaR using the normal distribution quantile function
    VaR = -portfolio_std_dev * np.percentile(np.random.normal(0, 1, 100000), 100 * alpha)
    
    return VaR

def calculate_weights(PERMNOs: list, asset_history: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the weights of assets in the portfolio based on transaction history.
    
    Parameters:
    PERMNOs (list): List of asset identifiers (PERMNOs) in the portfolio.
    asset_history (pd.DataFrame): DataFrame containing asset history with columns "timestamp",
    "PERMNO", "quantity", and "value".
    
    Returns:
    weights (pd.DataFrame): DataFrame of weights for each asset in the portfolio for each date.
    """
    weights_df = pd.DataFrame(index=asset_history['timestamp'].unique(), columns=PERMNOs)

    for PERMNO in PERMNOs:
        value = asset_history[asset_history['PERMNO'] == PERMNO].set_index('timestamp')["value"]
        weights_df[PERMNO] = value
    
    weights_df = weights_df.infer_objects(copy=False)
    weights_df = weights_df.fillna(0)
    # Normalize weights to sum to 1
    weights_df = weights_df.div(weights_df.abs().sum(axis=1), axis=0)

    return weights_df


sd = StockDatabase()
nyse = mcal.get_calendar("NYSE")
windows = pd.read_csv("windows.csv")
VaR_df = pd.DataFrame()

for i in range(len(windows)):
    print("Processing window:", i+1, "of", len(windows))
    train_start, train_end, test_start, test_end = windows.iloc[i]

    # Load the top pairs for the given test start date
    df = pd.read_csv(f"results/top_pairs/top_pairs_for_{test_start}.csv", index_col=[0,1])
    PERMNOs = list({str(PERMNO) for pairs in df.index.tolist() for PERMNO in pairs})

    # get the returns data for the PERMNOs
    returns = pd.DataFrame()
    for PERMNO in PERMNOs:
        data = sd.get_metrics(PERMNO, ("C",), train_start, train_end)
        returns[PERMNO] = data['DlyClose'].pct_change().dropna()
    
    # calculate the VaR
    cov_matrix = returns.cov()
    asset_history = pd.read_csv(f"results/asset_history/asset_history_for_{test_start}.csv")
    weights_df = calculate_weights(PERMNOs, asset_history)

    for date in weights_df.index:
        weight = weights_df.loc[date].values
        VaR = calculate_VaR(weight, cov_matrix, alpha=0.05)
        VaR_df.loc[date, 'VaR'] = VaR
    
# save the VaR data
VaR_df.to_csv("results/VaR_05.csv", index=True)
