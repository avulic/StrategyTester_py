import os

from numpy.core.numeric import NaN

os.environ["MODIN_ENGINE"] = "ray"
from matplotlib import dates
import numpy as np
import copy 
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
from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, DayLocator, MONDAY, num2date, date2num
from datetime import datetime, timezone, timedelta
import copy 
import time
from mplfinance.original_flavor import candlestick_ohlc
import math
from matplotlib.lines import Line2D
from matplotlib import cm


class Portfolio:
    def __init__(self, start_balance):
        self.startBalance = start_balance
        self.available_balance = start_balance #in BTC
        self.profit_btc = 0
        self.drawDown = 0
        self.numerOfTrades = 0
        self.trades = []
        self.orderBook = []

    def recalculate(self):
        trade = self.trades[0]
        if trade.liquidated:
            self.available_balance = self.available_balance - trade.initial_margine
        else:
            if trade.type == 'Long':
                self.profit_btc = (trade.quantity * ( (1 / trade.entry_price) - (1 / trade.exit_price) ))
                self.available_balance = self.available_balance +  self.profit_btc
            elif trade.type == 'Short':
                self.profit_btc = (trade.quantity * ( (1 / trade.exit_price) - (1 / trade.entry_price) ))
                self.available_balance = self.available_balance + self.profit_btc
        trade.trade_profit = self.profit_btc / trade.initial_margine * 100
        trade.profit_btc = self.profit_btc
    
    def SetGraph(self):
        graph_height = 4
        self.ax5 = plt.subplot()
        plt.title("Report")
        #self.ax1 = plt.axes()
        #self.ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=graph_height, colspan=1)
        plt.ylabel('Balance')
        plt.setp(self.ax5.get_xticklabels(), visible=False)
        plt.subplots_adjust(left=0.11, bottom=0.095, right=0.90, top=0.960, wspace=0.2, hspace=0)

    def PlotReport(self, data = None):       
        color_idx  = np.linspace(0, 1, len(data))
        #colors = [cm.rainbow(x) for x in evenly_spaced_interval]
        colors = [plt.cm.ocean(i) for i in np.linspace(0, 1, len(data))]
        self.ax5.set_prop_cycle('color',colors)
        #self.ax1.set_prop_cycle(cycler('color', colors))
        #for shift in phase_shift:
        #     plt.plot(x, np.sin(x - shift), lw=3)
        for key in data:
            data[key].plot(ax=self.ax5, lw=1, label=key)
            self.ax5.text(y=data[key].iloc[-3].values[0], x=data[key].index[-3], s=key.split(" ")[0], fontsize=8)
            print(key)


class Trade:
    def __init__(self):
        self.symbol = ''
        self.date = ''
        self.entry_price = None                    #in $
        self.exit_price = 0                     #in $
        self.value = 0                          #in BTC
        self.initial_margine = 0                #in BTC, estimete in $ ##COST
        self.quantity = 0                       #in $
        # self.latest_market_price
        self.liquidation_price = np.float64(0)  #in $
        self.liquidation_difference = 0
        self.leverage = 1
        self.trade_profit = 0                   #in %
        self.liquidated = False
        self.type = ''
        self.profit_btc = 0

class Broker:
    def __init__(self, equity, leverage, risk, data, signals, indicators, timeCandle):
        self.data = data
        self.signals = signals
        self.leverage = leverage
        self.risk = risk
        self.portfolio = Portfolio(equity)
        self.in_trade = False
        self.print = ""
        self.signals4 = None
        self.tickTime = timeCandle
        self.plotIndicator = indicators
        self.SetGraph()

    def Long(self, date, entry_price, leverage, quantity_usd):
        initial_margine = (quantity_usd / entry_price) / leverage
        if initial_margine <= self.portfolio.available_balance:
            trade = Trade()
            trade.symbol = 'BTCUSD'
            trade.initial_margine = initial_margine
            trade.date = date
            trade.entry_price = entry_price
            trade.leverage = leverage
            trade.quantity = quantity_usd
            trade.value = quantity_usd / entry_price
            trade.liquidation_price = (trade.entry_price * trade.leverage) / (trade.leverage + 1 - (0.005 * trade.leverage))
            trade.liquidation_difference = 100 / leverage
            self.in_trade = True
            trade.type = 'Long'
            self.portfolio.trades.insert(0, trade) 
            self.Print(date, trade.type, trade.entry_price, "---", self.portfolio.available_balance, "---", "---", trade.quantity, trade.initial_margine)           
        else:
            print("Margine error")
            return 0

    def Short(self, date, entry_price, leverage, quantity_usd):
        initial_margine = quantity_usd / (entry_price * leverage)
        if initial_margine <= self.portfolio.available_balance:
            trade = Trade()
            trade.symbol = 'BTCUSD'
            trade.initial_margine = initial_margine
            trade.date = date
            trade.entry_price = entry_price
            trade.leverage = leverage
            trade.quantity = quantity_usd
            trade.value = quantity_usd / trade.entry_price
            trade.liquidation_price = (trade.entry_price * trade.leverage) / (trade.leverage - 1 + (0.005 * trade.leverage))
            trade.liquidation_difference = 100 / leverage
            self.in_trade = True
            trade.type = 'Short'
            self.portfolio.trades.insert(0, trade) 
            self.Print(date, trade.type, trade.entry_price, "---", self.portfolio.available_balance, "---", "---", trade.quantity, trade.initial_margine)            
        else:
            print("Margine error")
            return 0

    def Close(self, price = None, date = None):
        self.in_trade = False
        self.portfolio.trades[0].exit_price = self.price[0]['Open'] if price == None else price
        self.portfolio.recalculate()
        #trade = Trade()
        trade = copy.deepcopy(self.portfolio.trades[0]) 
        trade.type = 'Close'
        trade.date = self.price[0]['Date'] if date == None else date
        self.portfolio.trades.insert(0,trade)
        if trade.liquidated:
            self.Print(trade.date, "Liquidated_"+trade.type, trade.entry_price, trade.exit_price, self.portfolio.available_balance, self.leverage, -1*trade.initial_margine, "---", "---")
        else:
            self.Print(trade.date, trade.type, trade.entry_price, trade.exit_price, self.portfolio.available_balance, trade.trade_profit, trade.profit_btc, "---", "---")            

    def CalcualteTimeDelta(self, time, delta):
        if self.tickTime == "1d":
            return time + timedelta(days=delta)
        elif self.tickTime == "1h":
            return time + timedelta(hours=delta)



    
    def CheckMargine5(self, start, end, lastrade):
        valueMin = self.data.loc[start:end]['Low']
        minPrice = valueMin.min()
        valueMax = self.data.loc[start:end]['High']
        maxPrice = valueMax.max()
        
        if lastrade.type == 'Long':
            if minPrice <= lastrade.liquidation_price:
                lastrade.liquidated = True
                self.portfolio.trades[0] = lastrade
                self.Close(maxPrice, pd.to_datetime(valueMax.loc[valueMax.values == maxPrice].index.values[0], format='%Y-%m-%d %H:%M'))
        elif lastrade.type == 'Short':
            if maxPrice >= lastrade.liquidation_price:
                lastrade.liquidated = True
                self.portfolio.trades[0] = lastrade
                self.Close(minPrice, pd.to_datetime(valueMin.loc[valueMin.values == minPrice].index.values[0], format='%Y-%m-%d %H:%M'))

    def CalculateRow3(self, trade): 
        [Type,Date,Value,O,H,L,C,PriceDate] = trade
        
        self.price[0] = {'Date':PriceDate,'Open':O,'High':H,'Low':L,'Close':C,'Volumen':True}
        self.signal[0]["Type"] = Type
        
        index = trade.name
        lastrade = self.portfolio.trades[0] if len(self.portfolio.trades) > 0 else None
        if lastrade != None:
            self.CheckMargine5(lastrade.date, PriceDate, lastrade)
            if not lastrade.liquidated:
                self.Close(O, PriceDate)
        
        #entry = O
        #date = PriceDate
        #cost = self.portfolio.available_balance * self.risk #exposure
        #sizeUSD = cost * entry * self.leverage 
        
        
        #timeSpan = date >= datetime.strptime("2020-12-12 00:00:00", "%Y-%m-%d %H:%M:%S") and date <= datetime.strptime("2021-01-02 08:00:00", "%Y-%m-%d  %H:%M:%S")
        #if Type == "Long" :
        #    self.Long(date, entry, self.leverage, sizeUSD)
        #elif Type == "Short":
        #    self.Short(date, entry, self.leverage, sizeUSD)
        self.Next()

    def Start5(self):
        #all = self.data.set_index('Date').join(self.signals.set_index('Date'))
        #self.data = self.data.set_index('Date')
        #allTrades = all.loc[self.signals['Date']].reset_index()
        startTimer = time.time()
        
        self.price = [None] * 2
        self.signal = [{}] * 2
        
        self.signals.apply(lambda row: self.CalculateRow3(row), axis=1)
        
        endtimer = time.time()
        print(f"Runtime of Start5 is {endtimer - startTimer}")



    def CalculateRow2(self, trade): 
        [Date,Open,High,Low,Close,Volumen,Type,Value,O,H,L,C,PriceDate] = trade
        index = trade.name
        lastrade = self.portfolio.trades[0] if len(self.portfolio.trades) > 0 else None
        if lastrade != None:
            self.Close(self.data.loc[self.CalcualteTimeDelta(lastrade.date, 1)]['Open'], self.CalcualteTimeDelta(lastrade.date, 1))
        if len(self.portfolio.orderBook) != 0:
            firsOrder = self.portfolio.orderBook.pop()
            entry = self.data.loc[self.CalcualteTimeDelta(firsOrder.date, 1)]['Open']
            date = self.CalcualteTimeDelta(firsOrder.date, 1)
            cost = self.portfolio.available_balance * self.risk #exposure
            sizeUSD = cost * entry * self.leverage 
            if firsOrder.type == "Long":
                self.Long(date, entry, self.leverage, sizeUSD)
            elif firsOrder.type == "Short":
                self.Short(date, entry, self.leverage, sizeUSD)
        tradeFirst = Trade()
        tradeFirst.date = trade[0]
        tradeFirst.type = trade[6]
        self.portfolio.orderBook.append(tradeFirst)
        
        if index != 0 and len(self.portfolio.trades):
            last = self.portfolio.trades[0]
            start = last.date
            end = Date
            valueMin = self.data.loc[start:self.CalcualteTimeDelta(end,-1)]['Low']
            minPrice = valueMin.min()
            valueMax = self.data.loc[start:self.CalcualteTimeDelta(end, -1)]['High']
            maxPrice = valueMax.max()
            if last.type == 'Long':
                if minPrice <= last.liquidation_price:
                    last.liquidated = True
                    self.portfolio.trades[0] = last
                    self.Close(maxPrice, pd.to_datetime(valueMax.loc[valueMax.values == maxPrice].index.values[0], format='%Y-%m-%d %H:%M'))
            elif last.type == 'Short':
                if maxPrice >= last.liquidation_price:
                    last.liquidated = True
                    self.portfolio.trades[0] = last
                    self.Close(minPrice, pd.to_datetime(valueMin.loc[valueMin.values == minPrice].index.values[0], format='%Y-%m-%d %H:%M'))

    def Start2(self):
        #trades = self.signals.ilpc[1].apply(lambda signal:self.data[:signal])
        #self.signals.iloc[0:len(self.signals)-1].apply(lambda signal:self.data.loc[self.signals.iloc[signal.index-1]['Date']:self.signals.iloc[signal.index]['Date']], axis=1)
        #self.data['Date'] = self.data['Date'].apply(lambda x: num2date(x).date())
        all = self.data.join(self.signals.set_index('Date'))
        #self.data = self.data.set_index('Date')
        allTrades = all.loc[self.signals['Date']].reset_index()
        startTimer = time.time()
        
        allTrades.apply(lambda row: self.CalculateRow2(row), axis=1)
        
        endTimer = time.time()
        print(f"Runtime of the program is {endTimer - startTimer}")

    def MakeTrade(self, trade): 
        lastrade = self.portfolio.trades[0] if len(self.portfolio.trades) > 0 else None
        if lastrade != None:
            self.Close(self.data.loc[self.CalcualteTimeDelta(lastrade.date, 1)]['Open'], self.CalcualteTimeDelta(lastrade.date, 1))
        if len(self.portfolio.orderBook) != 0:
            firsOrder = self.portfolio.orderBook.pop()
            entry = self.data.loc[self.CalcualteTimeDelta(firsOrder.date, 1)]['Open']
            date = self.CalcualteTimeDelta(firsOrder.date, 1)
            cost = self.portfolio.available_balance * self.risk #exposure
            sizeUSD = cost * entry * self.leverage 
            if firsOrder.type == "Long":
                self.Long(date, entry, self.leverage, sizeUSD)
            elif firsOrder.type == "Short":
                self.Short(date, entry, self.leverage, sizeUSD)
        tradeFirst = Trade()
        tradeFirst.date = trade[1]
        tradeFirst.type = trade[7]
        self.portfolio.orderBook.append(tradeFirst)

    def Start4(self):
        all = self.data.set_index('Date').join(self.signals.set_index('Date'))
        self.data = self.data.set_index('Date')
        allTrades = all.loc[self.signals['Date']].reset_index()
        startTimer = time.time()
        for (index,Date,Open,High,Low,Close,Volumen,Type,Value) in allTrades.itertuples(name=None):
            self.MakeTrade([index,Date,Open,High,Low,Close,Volumen,Type,Value])
            if index != 0 and len(self.portfolio.trades):
                last = self.portfolio.trades[0]
                start = last.date
                end = Date
                valueMin = self.data.loc[start:self.CalcualteTimeDelta(end,-1)]['Low']
                minPrice = valueMin.min()
                valueMax = self.data.loc[start:self.CalcualteTimeDelta(end, -1)]['High']
                maxPrice = valueMax.max()
                if last.type == 'Long':
                    if minPrice <= last.liquidation_price:
                        last.liquidated = True
                        self.portfolio.trades[0] = last
                        self.Close(maxPrice, pd.to_datetime(valueMax.loc[valueMax.values == maxPrice].index.values[0], format='%Y-%m-%d %H:%M'))
                elif last.type == 'Short':
                    if maxPrice >= last.liquidation_price:
                        last.liquidated = True
                        self.portfolio.trades[0] = last
                        self.Close(minPrice, pd.to_datetime(valueMin.loc[valueMin.values == minPrice].index.values[0], format='%Y-%m-%d %H:%M'))
        endTimer = time.time()
        print(f"Runtime of the program is {endTimer - startTimer}")

    def CalculateRow(self, dataRow):
        Date,Open,High,Low,Close,Volumen = dataRow
        index = dataRow.name
        self.price[0] = dataRow
        if index < len(self.data['Date'])-1:
            self.price[0] = self.data.iloc[index+1]
        if True in np.where(self.signals['Date'] == Date, True, False):
            self.signal[0] = self.signals.loc[self.signals['Date'] == Date].to_dict('records')[0]
            self.Next()
            self.signal[1] = self.signal[0]
        if self.in_trade:
            liqidated = self.CheckMargine()
        self.price[1] = self.price[0]
        
    def Start3(self):
        liqidated = False
        self.price = [None] * 2
        self.signal = [{}] * 2
        data = self.data.copy()
        start = time.time()
        
        data.apply(lambda row: self.CalculateRow(row), axis=1)

        end = time.time()
        print(f"Runtime of the program is {end - start}")
        return self.portfolio.available_balance

    def CalcuateRow7(self,Date,Open,High,Low,Close,Volumen,Type,Value):  
        self.price[0] = {'Date':Date,'Open':Open,'High':High,'Low':Low,'Close':Close,'Volumen':Volumen}
        if type(Type) == str:
            if self.signal[0]:
                if len(self.portfolio.orderBook) != 0: 
                    firsOrder = self.portfolio.orderBook.pop(0)
                    self.Next()
                    self.signal[1] = self.signal[0]
                    self.signal[0] = {}
            tradeFirst = Trade()
            tradeFirst.date = Date
            tradeFirst.type = Type
            self.portfolio.orderBook.append(tradeFirst)
            self.signal[0] = {'Date':Date,'Type':Type,'Value':Value}
        elif type(Type) != str:
            if len(self.portfolio.orderBook) != 0: 
                firsOrder = self.portfolio.orderBook.pop(0)
                self.Next()
                if self.signal[0]:
                    self.signal[1] = self.signal[0]
                    self.signal[0] = {}
        if self.in_trade:
            liqidated = self.CheckMargine()
        self.price[1] = self.price[0]

    def Start7(self):
        startTimer = time.time()
        
        allTrades = self.data.set_index('Date').join(self.signals.set_index('Date')).reset_index()
        self.data = self.data.set_index('Date')

        liqidated = False
        self.price = [None] * 2
        self.signal = [{}] * 2

        self.CalcuateRow7(allTrades['Date'],allTrades['Open'],allTrades['High'],allTrades['Low'],allTrades['Close'],allTrades['Volume'],allTrades['Type'],allTrades['Value'])

        endtimer = time.time()
        print(f"Runtime of the program is {endtimer - startTimer}")
        return self.portfolio.available_balance

    def Start6(self):
        startTimer = time.time()
        
        allTrades = self.data.join(self.signals.set_index('Date')).reset_index()
        liqidated = False
        self.price = [None] * 2
        self.signal = [{}] * 2

        for (index,Date,Open,High,Low,Close,Volumen,Type,Value, *args) in allTrades.itertuples(name=None):        
            self.price[0] = {'Date':Date,'Open':Open,'High':High,'Low':Low,'Close':Close,'Volumen':Volumen}
            if type(Type) == str:
                if self.signal[0]:
                    if len(self.portfolio.orderBook) != 0: 
                        firsOrder = self.portfolio.orderBook.pop(0)
                        self.Next()
                        self.signal[1] = self.signal[0]
                        self.signal[0] = {}
                tradeFirst = Trade()
                tradeFirst.date = Date
                tradeFirst.type = Type
                self.portfolio.orderBook.append(tradeFirst)
                self.signal[0] = {'Date':Date,'Type':Type,'Value':Value}
            elif type(Type) != str:
                if len(self.portfolio.orderBook) != 0: 
                    firsOrder = self.portfolio.orderBook.pop(0)
                    self.Next()
                    if self.signal[0]:
                        self.signal[1] = self.signal[0]
                        self.signal[0] = {}
            if self.in_trade:
                liqidated = self.CheckMargine()
            self.price[1] = self.price[0]

        endtimer = time.time()
        print(f"Runtime of Start6 is {endtimer - startTimer}")
        return self.portfolio.available_balance

    def Start(self):
        liqidated = False
        self.price = [None] * 2
        self.signal = [{}] * 2
        #signals #= self.signals.set_index("Date", drop=False)
        #self.data['Date'] = self.data['Date'].apply(lambda x: num2date(x).date())
        data = self.data.reset_index(drop=True)
        start = time.time()
        #self.signals['Date'] = self.signals['Date'].apply(lambda x: x.tz_convert('UTC').date())

        for (index,Date,Open,High,Low,Close,Volumen) in data.itertuples(name=None):
            self.price[0] = data.iloc[index]
            if index < len(data['Date'])-1:
                self.price[0] = data.iloc[index+1]
            if True in np.where(self.signals['Date'] == Date, True, False):
                self.signal[0] = self.signals.loc[self.signals['Date'] == Date].to_dict('records')[0]
                self.Next()
                self.signal[1] = self.signal[0]
            if self.in_trade:
                liqidated = self.CheckMargine()
            self.price[1] = self.price[0]

        end = time.time()
        print(f"Runtime of the program is {end - start}")
        return self.portfolio.available_balance

    def Next(self):
        pass
    
    def GetPortfolio(self):
        return self.portfolio

    #TODO multiple open trades
    def CheckMargine(self):
        trade = self.portfolio.trades[0]

        if trade.type == 'Long':
            if self.price[0]['Low'] <= trade.liquidation_price:
                trade.liquidated = True
                self.portfolio.trades[0] = trade
                self.Close(self.price[0]['Open'], self.price[0]['Date'])
        else:
            if self.price[0]['High'] >= trade.liquidation_price:
                trade.liquidated = True
                self.portfolio.trades[0] = trade
                self.Close(self.price[0]['Open'], self.price[0]['Date'])
        return trade.liquidated

    def Print(self, date, type, entry, exit, balance, pl, pl_btc, size, cost):
        entry = format(entry) if not isinstance(entry, str)  else entry
        exit = format(exit) if not isinstance(exit, str) else exit
        balance = format(balance, '.16f') if not isinstance(balance, str) else balance
        pl = format(pl, '.2f') if not isinstance(pl, str) else pl
        pl_btc = format(pl_btc, '.16f') if not isinstance(pl_btc, str) else pl_btc
        size = format(size, '.2f') if not isinstance(size, str) else size
        cost = format(cost) if not isinstance(cost, str) else cost
        self.print += '{} {} {} {} {} {} {} {} {}\n'.format(date, type, entry, exit, balance, pl, pl_btc, size, cost)

    def PlotTrades(self, ax2):       
        trades = pd.DataFrame.from_records([trade.__dict__ for trade in self.portfolio.trades]).rename(columns=
                                                                                                    {
                                                                                                        "symbol": "Symbol", 
                                                                                                        "date": "Date",
                                                                                                        'entry_price':'Entry_price',
                                                                                                        'exit_price':'Exit_price',
                                                                                                        'value':'Value',  
                                                                                                        'initial_margine':'Initial_margine',     
                                                                                                        'quantity':'Quantity',    
                                                                                                        'liquidation_difference':'Liquidation_difference',
                                                                                                        'leverage':'Leverage',  
                                                                                                        'trade_profit':'Trade_profit',  
                                                                                                        'liquidated':'Liquidated',   
                                                                                                        'type':'Type'
                                                                                                    })
        data = self.data.merge(trades, right_index=False, left_index=False, left_on='Date', right_on='Date')
        shortSig = data[data['Type']=='Short']
        longSig = data[data['Type']=='Long']
        closeSig = data[data['Type']=='Close']
        ax2.plot(longSig['Date'], longSig['Low'], 'g^', linewidth=2, markersize=7)
        ax2.plot(shortSig['Date'], shortSig['High'], 'rv', linewidth=2, markersize=7)
        ax2.plot(closeSig['Date'], closeSig['High'], 'bx', linewidth=2, markersize=7)

    def SetGraph(self):
        graph_height = 2
        self.axes = []
        if len(self.axes) != 0:
            self.fig = plt.figure(facecolor='#f0f0f0')
            plt.title("Title")
            
            for index in range(0,len(self.plotIndicator)):
                self.axes.append(plt.subplot2grid((len(self.plotIndicator)*graph_height, 1), (index*graph_height, 0), rowspan=graph_height, colspan=1))
                self.plotIndicator[index].Plot(self.axes[index])
                self.plotIndicator[index].PlotSignals(self.axes[index])
                plt.title(self.plotIndicator[index].title)
            
            # self.ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=graph_height, colspan=1)
            # plt.ylabel('H-L')
            
            # self.ax2 = plt.subplot2grid((10, 1), (2, 0), rowspan=graph_height*2, colspan=1, sharex=self.ax1)
            # plt.ylabel('Price')
            # self.ax2.grid(True)
            
            # self.ax3 = plt.subplot2grid((10, 1), (6, 0), rowspan=graph_height, colspan=1, sharex=self.ax1)
            # plt.ylabel('FR')

            # self.ax4 = plt.subplot2grid((10, 1), (8, 0), rowspan=graph_height, colspan=1, sharex=self.ax1)
            # plt.ylabel('FR-4h')
            #self.ax2.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            
            plt.setp(self.axes[1].get_xticklabels(), visible=False)
            plt.subplots_adjust(left=0.11, bottom=0.095, right=0.90, top=0.960, wspace=0.2, hspace=0)
            plt.rcParams["date.autoformatter.minute"] = "%Y-%m-%d %H:%M:%S"
            self.fig.tight_layout()

    def PlotPrice(self, cnadleData):
        candlestick_ohlc(self.axes[1], cnadleData.values, width=0.1, colorup='#77d879', colordown='#db3f3f')

    def GetAxes(self):
        return self.fig

