import yfinance as yf
import pandas as pd

def get_stock_data(name, start_date, end_date):
    stock_data = yf.download(name, start=start_date, end=end_date)
    return stock_data