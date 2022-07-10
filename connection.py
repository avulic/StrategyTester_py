from binance.client import Client
import json
import numpy as np
import datetime
from matplotlib import dates

class Binance:
    def __init__(self):
        creditentails = json.load(open('secret_binance.json'))
        self.api_key = creditentails['key']
        self.api_secret = creditentails['secret']
        self.client = Client(self.api_key, self.api_secret)

    def setInterval(self, interval):
        if interval == '1m':
            return Client.KLINE_INTERVAL_1MINUTE
        elif interval == '3m':
            return Client.KLINE_INTERVAL_3MINUTE
        elif interval == '5m':
            return Client.KLINE_INTERVAL_5MINUTE
        elif interval == '15m':
            return Client.KLINE_INTERVAL_15MINUTE
        elif interval == '30m':
            return Client.KLINE_INTERVAL_30MINUTE
        elif interval == '1h':
            return Client.KLINE_INTERVAL_1HOUR
        elif interval == '2h':
            return Client.KLINE_INTERVAL_2HOUR
        elif interval == '4h':
            return Client.KLINE_INTERVAL_4HOUR
        elif interval == '6h':
            return Client.KLINE_INTERVAL_6HOUR
        elif interval == '8h':
            return Client.KLINE_INTERVAL_8HOUR
        elif interval == '12h':
            return Client.KLINE_INTERVAL_12HOUR
        elif interval == '1d':
            return Client.KLINE_INTERVAL_1DAY
        elif interval == '3d':
            return Client.KLINE_INTERVAL_3DAY
        elif interval == '1w':
            return Client.KLINE_INTERVAL_1WEEK
        elif interval == '1M':
            return Client.KLINE_INTERVAL_1MONTH

    def getData(self, coin):
        self.coin = coin
        interval = self.setInterval(coin.interval)
        data = self.client.get_historical_klines(coin.symbol, interval, coin.start, coin.end)
        dataFixed = self.fixData(data)
        return dataFixed
    
    def fixData(self, data):
        date_data = []
        open_data = []
        high_data = []
        close_data = []
        low_data = []
        volumen_data = []
        for row in data:
            if self.coin.interval == '1d':
                date = datetime.datetime.utcfromtimestamp(int(row[0])/1000)
            elif self.coin.interval == '4h':
                date = datetime.datetime.utcfromtimestamp(int(row[0])/1000)
            elif self.coin.interval == '1h':
                date = datetime.datetime.utcfromtimestamp(int(row[0])/1000)
            date_data.append(date)
            open_data.append(float(row[1]))
            high_data.append(float(row[2]))
            low_data.append(float(row[3]))
            close_data.append(float(row[4]))
            volumen_data.append(float(row[5]))
            #turn_val = np.array(turn[1:], dtype= np.float64)

        return date_data, open_data, high_data, low_data, close_data, volumen_data

class Bittrex:
    pass