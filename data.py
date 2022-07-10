import csv
import numpy as np
import datetime
from matplotlib import dates
import pandas as pd

class CSV:
    def __init__(self, coin, location = "csv/"):
        self.location = location
        self.fname = location + coin.symbol + "_" + coin.interval + '.csv'

    def saveCSV(self, coin, klines):
        row = []
        with open(self.fname, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(("Date","Open","High","Low","Close","Volume"))
            for i in range(0, len(klines[0])):
                row.append(klines[0][i])
                row.append(klines[1][i])
                row.append(klines[2][i])
                row.append(klines[3][i])
                row.append(klines[4][i])
                row.append(klines[5][i])
                writer.writerow(row)
                row = []
        csvfile.close()

    def loadData(self):
        try:
            df = pd.read_csv(self.fname,index_col=0, date_parser=lambda col: pd.to_datetime(col, format='%Y-%m-%d %H:%M'))
            return df
        except IOError:
            return None
