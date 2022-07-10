import os
os.environ["MODIN_ENGINE"] = "ray"
import matplotlib 
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import dates
import matplotlib.ticker as mticker
from matplotlib import style
import urllib.request
import numpy as np
import urllib
import datetime as dt
from data import CSV
from coinPair import Pair
import matplotlib.cbook as cbook
import talib
from talib import abstract
import pandas as pd
from broker import Broker, Portfolio, Trade
from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, DayLocator, MONDAY, num2date, date2num, datestr2num
import copy 
from connection import Binance
import csv
import itertools
import matplotlib.cm as cm
from datetime import datetime, timezone, timedelta 

from indicators.LazyBear import WaveTrend, FR_Component, InverseFisherRSI
import time
from dateutil.parser import parse



class Broker(Broker):
    def __init__(self, equity, leverage, risk, data, signals, indicators, timeCandle):
        super().__init__(equity, leverage, risk, data, signals, indicators, timeCandle)
        self.i = 0

    def Next(self):
        entry = self.price[0]['Open']
        date = self.price[0]['Date']
        cost = self.portfolio.available_balance * self.risk #exposure
        sizeUSD = cost * entry * self.leverage 
        
        timeSpan = date >= datetime.strptime("2020-12-01 00:00:00", "%Y-%m-%d %H:%M:%S") and date <= datetime.strptime("2021-01-01 08:00:00", "%Y-%m-%d  %H:%M:%S")
        #lastSignal = self.signal[1] if self.signal[1] else self.signal[0]
        #signals4H = self.signals4[self.signals4.apply(lambda row: (row['Date'].date() >= lastSignal['Date'].date()) and (row['Date'].date() <= self.signal[0]['Date'].date()), axis=1)]
        
        if self.signal[0]['Type'] == 'Short' and timeSpan: #and signals4H.iloc[-1]['Type'] == 'Short'
            if self.in_trade:
                self.Close()
            self.i += 1
            self.Short(date, entry, self.leverage, sizeUSD)
            #self.trades.append(trade if trade != 0 else 0)
        elif self.signal[0]['Type'] == 'Long' and timeSpan: #and signals4H.iloc[-1]['Type'] == 'Long'
            if self.in_trade:
                self.Close()
            self.i += 1
            self.Long(date, entry,  self.leverage, sizeUSD)
            #self.trades.append(trade if trade != 0 else 0)

def HA(df):
    df['HA_Close']=(df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    idx = df.index.name
    df.reset_index(inplace=True)

    for i in range(0, len(df)):
        if i == 0:
            df.set_value(i, 'HA_Open', ((df.get_value(i, 'Open') + df.get_value(i, 'Close')) / 2))
        else:
            df.set_value(i, 'HA_Open', ((df.get_value(i - 1, 'HA_Open') + df.get_value(i - 1, 'HA_Close')) / 2))

    if idx:
        df.set_index(idx, inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','High']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','Low']].min(axis=1)
    data = {'Date':df['Date'],'Open':df['HA_Open'],'High':df['HA_High'],'Low':df['HA_Low'],'Close':df['HA_Close']}
    heikana = pd.DataFrame(data=data)
    heikana.index = df['Date']
    return heikana

def GetData(coin):
    file = CSV(coin)
    data = file.loadData()
    if data is not None and not data.empty and (data.index[0].date() == parse(coin.start).date()) and (data.index[-1].date() == parse(coin.end).date()):
        return data
    else:
        #contec to Exchange
        connection = Binance()
        data = connection.getData(coin)
        file.saveCSV(coin, data)
        data = file.loadData()
        return data

def variads(lst, lstsofar):
    param = []
    offset = len(lstsofar)
    outerlen = len(lst)
    innerLst = lst[offset]
    printit = False
    if offset == (outerlen - 1):
        printit = True
    for item in innerLst:
        if printit:
            return(lstsofar + [item])
        else:
            param.append(variads(lst, lstsofar + [item]))
    return param


def CalculateInidactor(data, parameters):
    n1 = 10
    n2 = 21
    crossover_sma_len = 3
    obLevel = 75
    osLevel = -obLevel
    obExtremeLevel = 100
    osExtremeLevel = -obExtremeLevel
    stratMultiplier = 1
    
    params = {"priceData":data,"n_channel":n1, "n_average":n2, "timeMultiplier":stratMultiplier, "crossover_sma_len":crossover_sma_len}
    FR = FR_Component(params)
    paramsWT = {"priceData":data,"chanelLenght":n1, "averageLenghtn":n2}
    WT = WaveTrend(paramsWT)
    paramsRSI = {"priceData":data,"rsi_lenght":14, "smoothing_lenght":9, "crossover_sma_len":4}
    IFRSI = InverseFisherRSI(paramsRSI)
    return WT, FR, IFRSI

if __name__ == '__main__':
    timeCandledf4h = '4h'
    coin = Pair("BTCUSDT", timeCandledf4h, "17 Aug, 2017", "17 May, 2020")
    df4h = GetData(coin)
    #heikenData = copy.deepcopy(dataPanda)
    #heikencandle = HA(heikenData)    
    
    timeCandledf1d = '1d'
    coin = Pair("BTCUSDT", timeCandledf1d, "17 Aug, 2017", "01 Jan, 2021")
    df1d = GetData(coin)
    
    timeCandledf1h = '1h'
    coin = Pair("BTCUSDT", timeCandledf1h, "17 Aug, 2017", "25 May, 2020")
    df1h = GetData(coin)
    
    timeCandle = timeCandledf1d
    
    parametersRanegFR = [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,8,10,12,15],[2,5,7,8,12,15,21,22,25]]
    parametersRanegFR1 = [[1,3,10,21]]
    paramsa = list(itertools.product(*parametersRanegFR))

    lvgrangeRange = [[2,3,5,10,20,25,40,50],[0.03,0.05,0.1,0.15,0.20,0.30]]
    lvgparameters = list(itertools.product(*lvgrangeRange))
    
    broker = None
    i = 0
    parameters = [[1,3,10,20],[1,4,10,20],[1,7,10,20],[2,5,10,20],[2,7,10,20]]
    alltSeriesTrade = {}
    
    f = open( 'params.txt', 'w' )
    for parameter in paramsa:
        f.write(str(parameter)+'\n')
    f.close()
    
    
    cycles = {
        "2013" : {
            "bull" : "13.06.2011",
            "bear": "21.11.2011",
            "accumulation": "11.6.2012",
            "expansion": "20.08.2012",
            "reaccumulation": "07.01.2013",
        },
        "2016" : {
            "bull" : "02.12.2013",
            "bear": "12.01.2015",
            "accumulation": "12.10.2015",
            "expansion": "21.12.2015",
            "reaccumulation": "16.5.2016",
        },
        "2020" : {
            "bull" : "18.12.2017",
            "bear": "10.12.2018",
            "accumulation": "25.03.2019",
            "expansion": "08.07.1019",
            "reaccumulation": "31.12.2020",
        }
    }
    
    for parameter in parametersRanegFR1:
        startTimer = time.time()

        params = {"priceData":df1d,"n_channel":parameter[2], "n_average":parameter[3], "timeMultiplier":parameter[0], "crossover_sma_len":parameter[1], "timeFrame":timeCandle}
        FR = FR_Component(params)
        # paramsWT = {"priceData":df1d,"chanelLenght":parameter[2], "averageLenghtn":parameter[3]}
        # WT = WaveTrend(paramsWT)
        # paramsRSI = {"priceData":df1d,"rsi_lenght":14, "smoothing_lenght":9, "crossover_sma_len":4}
        # IFRSI = InverseFisherRSI(paramsRSI)
        #FR4H = CalculateInidactor(df4h)

        
        indicators = [FR.GetSignals()]
        if i == 0:
            broker = Broker(1, 10, 0.1, df1d, FR.GetSignals(), indicators, timeCandle)
            broker.GetPortfolio().SetGraph()
            
            portfolioBefore = broker.GetPortfolio().available_balance
            broker.Start6()
            printTrades = pd.DataFrame([x.split(' ') for x in broker.print.split('\n')], columns=["Date", "Time", "Type", "Entry", "Exit", "Balance", "P\L%", "P\Lbtc", "Size", "Cost"])
            printTrades.replace(np.nan, "", inplace=True)
            printTrades['Date'] = printTrades.apply(lambda row: pd.to_datetime(row.Date, format='%Y-%m-%d %H:%M'), axis=1)
            printTrades1 = pd.DataFrame({'Balance':pd.to_numeric(printTrades['Balance'], errors='coerce').values}, index=printTrades['Date']).rename(columns={'Balance':str(parameter)})
            alltSeriesTrade[str(i) + " " + str(parameter)] = printTrades1
            portfolioAfter = broker.GetPortfolio().available_balance
            i = i+1
        else:
            broker1 = Broker(1, 10, 0.1, df1d, FR.GetSignals(), indicators, timeCandle, timeCandle)
            portfolioBefore = broker1.GetPortfolio().available_balance
            broker1.Start6()
            printTrades = pd.DataFrame([x.split(' ') for x in broker1.print.split('\n')], columns=["Date", "Type", "Entry", "Exit", "Balance", "P\L%", "P\Lbtc", "Size", "Cost"])
            printTrades.replace(np.nan, "", inplace=True)
            printTrades['Date'] = printTrades.apply(lambda row: pd.to_datetime(row.Date, format='%Y-%m-%d %H:%M'), axis=1)
            printTrades1 = pd.DataFrame({'Balance':pd.to_numeric(printTrades['Balance'], errors='coerce').values}, index=printTrades['Date']).rename(columns={'Balance':str(parameter)})
            alltSeriesTrade[str(i) + " " + str(parameter)] = printTrades1
            portfolioAfter = broker1.GetPortfolio().available_balance
            i = i+1
            
        endtimer = time.time()
        printTrades.to_csv("trades" + timeCandle + "start6" +".csv")
        #print(f"Runtime of full iteration is {endtimer - startTimer}")
        print(portfolioBefore)
        print(portfolioAfter)
        #print(printTrades)
    broker.GetPortfolio().PlotReport(alltSeriesTrade)
        #broker.signals4 = FR4H.GetSignals()
        #broker.tickTime = "H"
        #broker.period = 4
        
        #portfolioBefore = broker.GetPortfolio().available_balance
        #broker1.Start6()
        #portfolioAfter = broker.GetPortfolio().available_balance
        #print(portfolioBefore)
        #print(portfolioAfter)

        
    #fig = broker.GetAxes()   
    #fig.autofmt_xdate()
        
        # f = open( "log\\" + str([1,3,10,21]) + str(coin.symbol) + "_" + str(coin.interval) + 'log.txt', 'w' )
        # f.write('{:<10} {:<20} {:^10} {:^15} {:^25} {:^15} {:^30} {:^20} {:^25}\n'.format("Date", "Type", "Entry", "Exit", "Balance", "P\L%", "P\Lbtc", "Size", "Cost"))
        # f.write( str(portfolioBefore)+"\n" )
        # f.write( broker.print )
        # f.write( str(portfolioAfter) )
        # f.close()

    #fig.axes[1].legend()    
    #broker.PlotTrades(fig.axes[1])
    #df1d = df1d.reset_index()
    #df1d['Date'] = df1d.apply(lambda r: date2num(r['Date']), axis=1)
    #broker.PlotPrice(df1d)

        
        #plt.savefig('line_plot1.pdf')  
    plt.show()

