import pandas as pd
import numpy as np
from trading_model import TradingModel
from data import StockDatabase
import pytest

def test_calc_spreads_1():
    # Test 1: Basic spread calculation
    pairs = [(1, 2), (3, 4)]
    OLS_coeff = {pairs[0]: (1, 0.5), 
                 pairs[1]: (1.5, 0.3)}
    threshold = {pair: 1 for pair in pairs}
    stop_loss = {pair: 2 for pair in pairs}
    model = TradingModel(pairs, OLS_coeff, threshold, stop_loss)
    close_prices = {1: 10,
                    2: 15,
                    3: 20,
                    4: 30}

    spread = model.calc_spreads(close_prices)
    expected_spread = {pairs[0]: np.log(10) - 1 * np.log(15) - 0.5,
                       pairs[1]: np.log(20) - 1.5 * np.log(30) - 0.3}
    for pair in pairs:
        assert np.isclose(spread[pair], expected_spread[pair]), f"Failed for pair {pair}: {spread[pair]} != {expected_spread[pair]}"

def test_calc_spreads_2():
    # Test 2: spread with real data
    db = StockDatabase()
    pairs = [(db.search_PERMNO("AAPL", "2023-01-01"), db.search_PERMNO("MSFT", "2023-01-01")), 
             (db.search_PERMNO("GOOGL", "2023-01-01"), db.search_PERMNO("AMZN", "2023-01-01"))]
    OLS_coeff = {pairs[0]: (1.2, 0.8),
                 pairs[1]: (1.5, -0.5)}
    threshold = {pair: 1 for pair in pairs}
    stop_loss = {pair: 2 for pair in pairs}
    model = TradingModel(pairs, OLS_coeff, threshold, stop_loss)
    close_prices = {}
    for permno in set([p for pair in pairs for p in pair]):
        close_prices[permno] = db.get_metrics(permno, ("C"), "2024-01-02", "2024-01-02")["DlyClose"].iloc[0]

    spread = model.calc_spreads(close_prices)
    for pair in pairs:
        PERMNO1, PERMNO2 = pair
        alpha, beta = OLS_coeff[pair]
        expected_spread = np.log(close_prices[PERMNO1]) - alpha * np.log(close_prices[PERMNO2]) - beta
        assert np.isclose(spread[pair], expected_spread), f"Failed for pair {pair}: {spread[pair]} != {expected_spread}"

def test_calc_spreads_3():
    # Test 3: nan input
    pairs = [(1, 2), (3, 4)]
    OLS_coeff = {pairs[0]: (1, 0.5), 
                 pairs[1]: (1.5, 0.3)}
    threshold = {pair: 1 for pair in pairs}
    stop_loss = {pair: 2 for pair in pairs}
    model = TradingModel(pairs, OLS_coeff, threshold, stop_loss)
    close_prices = {1: np.nan,
                    2: 15,
                    3: 20,
                    4: 30}

    spread = model.calc_spreads(close_prices)
    expected_spread = {pairs[0]: np.nan,
                       pairs[1]: np.log(20) - 1.5 * np.log(30) - 0.3}
    
    for pair in pairs:
        if np.isnan(expected_spread[pair]):
            assert np.isnan(spread[pair]), f"Expected NaN for pair {pair}, got {spread[pair]}"
        else:
            assert np.isclose(spread[pair], expected_spread[pair]), f"Failed for pair {pair}: {spread[pair]} != {expected_spread[pair]}"


def test_make_decisions_1():
    # Create the TradingModel
    pairs = [(1, 2), (3, 4)]
    OLS_coeff = {pairs[0]: (1, 0), 
                 pairs[1]: (2, 0)}
    threshold = {pair: 1 for pair in pairs}
    stop_loss = {pair: 2 for pair in pairs}
    model = TradingModel(pairs, OLS_coeff, threshold, stop_loss)

    # Test 1: no trades because no threshold crossed
    close_prices = {1: 10,
                    2: 10,
                    3: 20,
                    4: 5}
    last_close_dates = {1: "2024-01-01",
                       2: "2024-01-01",
                       3: "2024-01-01",
                       4: "2024-01-01"}
    
    decisions = model.make_decisions_helper("2023-01-01", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when no thresholds are crossed"
    
    # Test 2: entered short and long positions
    close_prices = {1: 10,
                    2: 3,
                    3: 20,
                    4: 10}
    decisions = model.make_decisions_helper("2023-01-02", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], -1), (pairs[1], 1)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    # Test 3: no trades because zero not crossed
    decisions = model.make_decisions_helper("2023-01-03", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when zero is not crossed"

    # Test 4: exit position by crossing zero and stop loss
    close_prices = {1: 10,
                    2: 1,
                    3: 50,
                    4: 5}
    decisions = model.make_decisions_helper("2023-01-04", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 0), (pairs[1], 0)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    # Test 5: no trades if above/below stop loss
    close_prices = {1: 10,
                    2: 0.5,
                    3: 50,
                    4: 20}
    decisions = model.make_decisions_helper("2023-01-05", close_prices, last_close_dates)
    assert decisions == [], f"Expected {expected_decisions}, got {decisions}"

    # Test 6: open short and long positions again
    close_prices = {1: 10,
                    2: 40,
                    3: 75,
                    4: 5}
    decisions = model.make_decisions_helper("2023-01-06", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 1), (pairs[1], -1)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    # Test 7: close long, open short and close short open long
    close_prices = {1: 50,
                    2: 10,
                    3: 10,
                    4: 7}
    decisions = model.make_decisions_helper("2023-01-07", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 0), (pairs[0], -1), (pairs[1], 0), (pairs[1], 1)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    # Test 8: close short, close long but closed above stop loss in other direction
    # so no new positions opened
    close_prices = {1: 10,
                    2: 75,
                    3: 50,
                    4: 2}
    decisions = model.make_decisions_helper("2023-01-08", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 0), (pairs[1], 0)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

def test_make_decisions_2():
    # Create the TradingModel
    pairs = [(1, 2)]
    OLS_coeff = {pairs[0]: (0.5, 0)}
    threshold = {pair: 1 for pair in pairs}
    stop_loss = {pair: 2 for pair in pairs}
    model = TradingModel(pairs, OLS_coeff, threshold, stop_loss)

    # Test 1
    close_prices = {1: 10,
                    2: 20}
    last_close_dates = {1: "2024-01-01",
                       2: "2024-01-01",}
    decisions = model.make_decisions_helper("2023-01-01", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when no thresholds are crossed"

    # Test 2
    close_prices = {1: 10,
                    2: 21}
    decisions = model.make_decisions_helper("2023-01-02", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when no thresholds are crossed"

    # Test 3
    close_prices = {1: 10,
                    2: 5}
    decisions = model.make_decisions_helper("2023-01-03", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], -1)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    # Test 4
    close_prices = {1: 10,
                    2: 1}
    decisions = model.make_decisions_helper("2023-01-04", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 0)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    # Test 5
    close_prices = {1: 10,
                    2: 3}
    decisions = model.make_decisions_helper("2023-01-05", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], -1)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    # Test 6
    close_prices = {1: 10,
                    2: 20}
    decisions = model.make_decisions_helper("2023-01-06", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when no thresholds are crossed"

    # Test 7
    close_prices = {1: 3,
                    2: 100}
    decisions = model.make_decisions_helper("2023-01-07", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 0), (pairs[0], 1)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    # Test 8
    close_prices = {1: 1,
                    2: 100}
    decisions = model.make_decisions_helper("2023-01-08", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 0)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    # Test 9
    close_prices = {1: 5,
                    2: 100}
    decisions = model.make_decisions_helper("2023-01-09", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when no thresholds are crossed"

    # test 10
    close_prices = {1: 50,
                    2: 5}
    decisions = model.make_decisions_helper("2023-01-10", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when no thresholds are crossed"

def test_make_decisions_3():
    # Create the TradingModel
    pairs = [(1, 2)]
    OLS_coeff = {pairs[0]: (0.5, 0)}
    threshold = {pair: 1 for pair in pairs}
    stop_loss = {pair: 2 for pair in pairs}
    model = TradingModel(pairs, OLS_coeff, threshold, stop_loss)
    
    # Test Delisted Stocks (both delisted)
    close_prices = {1: 20,
                    2: 20}
    last_close_dates = {1: "2024-01-01",
                        2: "2024-01-01"}
    decisions = model.make_decisions_helper("2024-01-01", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when stock is delisted" 

    # Test Delisted Stocks (one delisted)
    close_prices = {1: 20,
                    2: 20}
    last_close_dates = {1: "2024-01-01",
                        2: "2024-06-01"}
    decisions = model.make_decisions_helper("2024-01-01", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when stock is delisted" 

    # Test Delisted Stocks After Shorting
    close_prices = {1: 20,
                    2: 20}
    decisions = model.make_decisions_helper("2023-12-30", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], -1)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    close_prices = {1: 20,
                    2: 20}
    decisions = model.make_decisions_helper("2024-01-01", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 0)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"
    
    # Test Delisted Stocks After Longing
    close_prices = {1: 2,
                    2: 50}
    decisions = model.make_decisions_helper("2023-12-29", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 1)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    close_prices = {1: 2,
                    2: 50}
    decisions = model.make_decisions_helper("2024-01-01", close_prices, last_close_dates)
    expected_decisions = [(pairs[0], 0)]
    assert decisions == expected_decisions, f"Expected {expected_decisions}, got {decisions}"

    close_prices = {1: np.nan,
                    2: 30}
    decisions = model.make_decisions_helper("2024-01-02", close_prices, last_close_dates)
    assert decisions == [], "Expected no decisions when stock is delisted"


def test_trade_1():
    # Create the TradingModel
    pairs = [(1, 2), (3, 4)]
    OLS_coeff = {pairs[0]: (1, 0), 
                 pairs[1]: (2, 0)}
    threshold = {pair: 1 for pair in pairs}
    stop_loss = {pair: 2 for pair in pairs}
    model = TradingModel(pairs, OLS_coeff, threshold, stop_loss)

    # Test 1: no trades because no threshold crossed
    open_prices = {1: 10,
                   2: 10,
                   3: 20,
                   4: 5}

    decisions = []
    trades = model.trade_helper(decisions, open_prices)
    assert trades == [], "Expected no trades when no thresholds are crossed"
    
    # Test 2: entered short and long positions
    open_prices = {1: 10,
                   2: 3,
                   3: 20,
                   4: 10}
    decisions = [(pairs[0], -1), (pairs[1], 1)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], -0.5/10), (pairs[0][1], 0.5/3), 
                       (pairs[1][0], (1/3)/20), (pairs[1][1], -(2/3)/10)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"

    # Test 3: no trades because zero not crossed
    decisions = []
    trades = model.trade_helper(decisions, open_prices)
    assert trades == [], "Expected no trades when zero is not crossed"

    # Test 4: exit position by crossing zero and stop loss
    open_prices = {1: 10,
                   2: 1,
                   3: 50,
                   4: 5}
    decisions = [(pairs[0], 0), (pairs[1], 0)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], 0.5/10), (pairs[0][1], -0.5/3),
                       (pairs[1][0], -(1/3)/20), (pairs[1][1], (2/3)/10)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"

    # Test 5: no trades if above/below stop loss
    open_prices = {1: 10,
                   2: 0.5,
                   3: 50,
                   4: 20}
    decisions = []
    trades = model.trade_helper(decisions, open_prices)
    assert trades == [], "Expected no trades when no thresholds are crossed"

    # Test 6: open short and long positions again
    open_prices = {1: 10,
                   2: 40,
                   3: 75,
                   4: 5}
    decisions = [(pairs[0], 1), (pairs[1], -1)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], 0.5/10), (pairs[0][1], -0.5/40),
                       (pairs[1][0], -(1/3)/75), (pairs[1][1], (2/3)/5)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"

    # Test 7: close long, open short and close short open long
    open_prices = {1: 50,
                   2: 10,
                   3: 10,
                   4: 7}
    decisions = [(pairs[0], 0), (pairs[0], -1), (pairs[1], 0), (pairs[1], 1)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], -0.5/10), (pairs[0][1], 0.5/40),
                       (pairs[0][0], -0.5/50), (pairs[0][1], 0.5/10),
                       (pairs[1][0], (1/3)/75), (pairs[1][1], -(2/3)/5),
                       (pairs[1][0], (1/3)/10), (pairs[1][1], -(2/3)/7)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"

    # Test 8: close short, close long but closed above stop loss in other direction
    # so no new positions opened
    open_prices = {1: 10,
                   2: 75,
                   3: 50,
                   4: 2}
    decisions = [(pairs[0], 0), (pairs[1], 0)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], 0.5/50), (pairs[0][1], -0.5/10),
                       (pairs[1][0], -(1/3)/10), (pairs[1][1], (2/3)/7)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"

def test_trade_2():
    # Create the TradingModel
    pairs = [(1, 2)]
    OLS_coeff = {pairs[0]: (0.5, 0)}
    threshold = {pair: 1 for pair in pairs}
    stop_loss = {pair: 2 for pair in pairs}
    model = TradingModel(pairs, OLS_coeff, threshold, stop_loss, dollar_per_trade=15)

    # Test 1
    open_prices = {1: 11,
                   2: 19}
    decisions = []
    trades = model.trade_helper(decisions, open_prices)
    assert trades == [], "Expected no trades when no thresholds are crossed"

    # Test 2
    open_prices = {1: 11,
                   2: 20}
    decisions = []
    trades = model.trade_helper(decisions, open_prices)
    assert trades == [], "Expected no trades when no thresholds are crossed"

    # Test 3
    open_prices = {1: 11,
                   2: 4}
    decisions = [(pairs[0], -1)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], -10 / 11), (pairs[0][1], 5 / 4)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"

    # Test 4
    open_prices = {1: 10,
                   2: 2}
    decisions = [(pairs[0], 0)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], 10 / 11), (pairs[0][1], -5 / 4)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"

    # Test 5
    open_prices = {1: 9,
                   2: 3}
    decisions = [(pairs[0], -1)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], -10 / 9), (pairs[0][1], 5 / 3)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"

    # Test 6
    open_prices = {1: 11,
                   2: 17}
    decisions = []
    trades = model.trade_helper(decisions, open_prices)
    assert trades == [], "Expected no trades when no thresholds are crossed"

    # Test 7
    open_prices = {1: 5,
                   2: 100}
    decisions = [(pairs[0], 0), (pairs[0], 1)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], 10 / 9), (pairs[0][1], -5 / 3),
                       (pairs[0][0], 10 / 5), (pairs[0][1], -5 / 100)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"


    # Test 8
    open_prices = {1: 1,
                   2: 105}
    decisions = [(pairs[0], 0)]
    trades = model.trade_helper(decisions, open_prices)
    expected_trades = [(pairs[0][0], -10 / 5), (pairs[0][1], 5 / 100)]
    assert trades == expected_trades, f"Expected {expected_trades}, got {trades}"

    # Test 9
    open_prices = {1: 5,
                   2: 100}
    decisions = []
    trades = model.trade_helper(decisions, open_prices)
    assert trades == [], "Expected no trades when no thresholds are crossed"

    # Test 10
    open_prices = {1: 50,
                   2: 5}
    decisions = []
    trades = model.trade_helper(decisions, open_prices)
    assert trades == [], "Expected no trades when no thresholds are crossed"

