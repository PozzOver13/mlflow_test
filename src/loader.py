import yfinance

from src.config import TICKERS


class Loader:
    def __init__(self):
        self.tickers_list = ", ".join(TICKERS)

    def load_data(self):
        raw_data = yfinance.download(tickers=self.tickers_list, interval="1d", group_by='ticker',
                                     auto_adjust=True, treads=True)

        return raw_data