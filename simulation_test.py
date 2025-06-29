import pandas as pd
import numpy as np
from simulation import Account
import pytest

def test_update_date():
    account = Account("2024-01-01", 1000)
    account.update_date("2024-01-02")
    assert account.date == "2024-01-02"
    account.update_date("2024-01-03")
    assert account.date == "2024-01-03"
    account.update_date("2024-01-03")
    assert account.date == "2024-01-03"

    # Test with a date that is not moving forward
    with pytest.raises(ValueError):
        account.update_date("2024-01-02")

    # Test with a non-trading day
    with pytest.raises(ValueError):
        account.update_date("2024-01-06")  # Assuming this is not a trading day


def test_make_transaction_1():
    # Testing buying and selling a stock
    account = Account("2016-07-01", 100)
    account.update_date("2016-07-01")
    account.make_transaction([(93418, 1)])  # Buying 1 share
    assert account.assets == {93418: 1}
    assert np.isclose(account.buying_power, 100 - 5.11)


    account.update_date("2016-07-05")
    account.make_transaction([(93418, 10)])  # Buying 10 shares
    assert account.assets == {93418: 11}
    assert np.isclose(account.buying_power, 100 - 5.11 * 11)

    account.update_date("2016-07-07")
    account.make_transaction([(93418, -5)])  # Selling 5 shares
    assert account.assets == {93418: 6}
    assert np.isclose(account.buying_power, 100 - 5.11 * 11 + 5.12 * 5)

    account.update_date("2016-07-08")
    account.make_transaction([(93418, -6)])  # Selling all shares
    assert account.assets == {}
    assert np.isclose(account.buying_power, 100 - 5.11 * 11 + 5.12 * 11)

    df = account.get_transaction_history()
    expected_df = pd.DataFrame({
        "timestamp": ["2016-07-01", "2016-07-05", "2016-07-07", "2016-07-08"],
        "PERMNO": [93418, 93418, 93418, 93418],
        "quantity": [1, 10, -5, -6],
        "price": [5.11, 5.11, 5.12, 5.12]
    })
    pd.testing.assert_frame_equal(df, expected_df)

def test_make_transaction_2():
    # Testing buying and selling multiple stocks
    account = Account("2016-07-01", 100)
    account.update_date("2016-07-01")
    account.make_transaction([(93418, 1), (93416, -1)])
    assert account.assets == {93418: 1, 93416: -1}
    assert np.isclose(account.buying_power, 100 - 5.11 + 12.34)

    account.update_date("2016-07-05")
    account.make_transaction([(93418, 10), (93416, -5)])  # Buying 10 shares
    assert account.assets == {93418: 11, 93416: -6}
    assert np.isclose(account.buying_power, 100 - 5.11 * 11 + 12.34 + 12.24 * 5)

    account.update_date("2016-07-07")
    account.make_transaction([(93418, -5)])  # Selling 5 shares
    assert account.assets == {93418: 6, 93416: -6}
    assert np.isclose(account.buying_power, 100 - 5.11 * 11 + 5.12 * 5 + 12.34 + 12.24 * 5)

    account.update_date("2016-07-08")
    account.make_transaction([(93418, -6), (93416, 6)])  # Selling all shares
    assert account.assets == {}
    assert np.isclose(account.buying_power, 100 - 5.11 * 11 + 5.12 * 11 + 12.34 + 12.24 * 5 - 12.28 * 6)

    df = account.get_transaction_history()
    expected_df = pd.DataFrame({
        "timestamp": ["2016-07-01", "2016-07-01", "2016-07-05", "2016-07-05", "2016-07-07", "2016-07-08", "2016-07-08"],
        "PERMNO": [93418, 93416, 93418, 93416, 93418, 93418, 93416],
        "quantity": [1, -1, 10, -5, -5, -6, 6],
        "price": [5.11, 12.34, 5.11, 12.24, 5.12, 5.12, 12.28]
    })
    pd.testing.assert_frame_equal(df, expected_df)

def test_make_transaction_3():
    # Testing buying with insufficient buying power
    account = Account("2016-07-01", 10)
    account.update_date("2016-07-01")
    account.make_transaction([(93418, 1)])  # Buying 1 share
    assert account.assets == {93418: 1}
    assert np.isclose(account.buying_power, 10 - 5.11)

    # Attempt to buy more than available buying power
    with pytest.raises(ValueError):
        account.make_transaction([(93418, 20)])  # Trying to buy 20 shares


def test_liquidate_1():
    # Test 1
    account = Account("2016-07-01", 100)
    account.update_date("2016-07-01")
    account.make_transaction([(93418, 1)])  # Buying 1 share
    account.liquidate()
    assert account.assets == {}
    assert np.isclose(account.buying_power, 100)

    df = account.get_transaction_history()
    expected_df = pd.DataFrame({
        "timestamp": ["2016-07-01", "2016-07-01"],
        "PERMNO": [93418, 93418],
        "quantity": [1, -1],
        "price": [5.11, 5.11]
    })
    pd.testing.assert_frame_equal(df, expected_df)


    # Test 2
    account = Account("2016-07-01", 100)
    account.update_date("2016-07-01")
    account.make_transaction([(93418, 1), (93416, -1)])

    account.update_date("2016-07-05")
    account.make_transaction([(93418, 10), (93416, -5)])  # Buying 10 shares

    account.update_date("2016-07-07")
    account.make_transaction([(93418, -5)])  # Selling 5 shares

    account.update_date("2016-07-08")
    account.liquidate()  # Selling all shares
    assert account.assets == {}
    assert np.isclose(account.buying_power, 100 - 5.11 * 11 + 5.12 * 11 + 12.34 + 12.24 * 5 - 12.28 * 6)

    df = account.get_transaction_history()
    expected_df = pd.DataFrame({
        "timestamp": ["2016-07-01", "2016-07-01", "2016-07-05", "2016-07-05", "2016-07-07", "2016-07-08", "2016-07-08"],
        "PERMNO": [93418, 93416, 93418, 93416, 93418, 93418, 93416],
        "quantity": [1, -1, 10, -5, -5, -6, 6],
        "price": [5.11, 12.34, 5.11, 12.24, 5.12, 5.12, 12.28]
    })
    pd.testing.assert_frame_equal(df, expected_df)

def test_liquidate_2():
    # Negative buying power after liquidation
    account = Account("2016-07-01", 0)
    account.update_date("2016-07-01")
    account.make_transaction([(93418, -10)])  # Selling 10 share

    account.update_date("2016-07-07")
    with pytest.raises(ValueError):
        account.liquidate()


def test_update_capital():
    # Testing buying and selling multiple stocks
    account = Account("2016-07-01", 100)
    account.update_date("2016-07-01")
    account.make_transaction([(93418, 1), (93416, -1)])
    account.update_capital()  # Update capital after transaction
    assert np.isclose(account.capital, 100)

    account.update_date("2016-07-05")
    account.make_transaction([(93418, 10), (93416, -5)])  # Buying 10 shares
    account.update_capital()  # Update capital after transaction
    assert np.isclose(account.capital, 100 + 0.10)

    account.update_date("2016-07-07")
    account.make_transaction([(93418, -5)])  # Selling 5 shares
    account.update_capital()  # Update capital after transaction
    assert np.isclose(account.capital, 100 + 0.10 + 0.11 - 0.24)

    account.update_date("2016-07-08")
    account.make_transaction([(93418, -6), (93416, 6)])  # Selling all shares
    account.update_capital()  # Update capital after transaction
    assert np.isclose(account.capital, 100 + 0.10 + 0.11 - 0.24)

    df = account.get_capital_history()
    expected_df = pd.DataFrame({
        "timestamp": ["2016-07-01", "2016-07-05", "2016-07-07", "2016-07-08"],
        "capital": [100, 100 + 0.10, 100 + 0.10 + 0.11 - 0.24, 100 + 0.10 + 0.11 - 0.24]
    })
    pd.testing.assert_frame_equal(df, expected_df)

