import pandas as pd
import numpy as np
import talib
from datetime import datetime, timezone, timedelta

def hlc3(high, low, close):
    value = (high + low + close) / 3
    return value

class FR_Component():
    def __init__(self, params={}):
        self.title = "FR_Component"
        priceData, n_channel, n_average, timeMultiplier, crossover_sma_len, self.timeFrame = params.values()
        
        self.ohlc = priceData
        hlc3Data = list(map(hlc3, priceData['High'], priceData['Low'], priceData['Close']))
        hlc3Data = pd.DataFrame({'Value': hlc3Data, 'Date': priceData.index.values})
        hlc3Data['Date'] = hlc3Data['Date'].apply(lambda col: pd.to_datetime(col, format='%Y-%m-%d %H:%M'))
        hlc3Data = hlc3Data.set_index('Date', drop=True)
        
        self.dataSet = hlc3Data
        esa = self.dataSet['Value'].ewm(span=(n_channel * timeMultiplier), adjust=False,min_periods=(n_channel * timeMultiplier)-1).mean()
        d = self.dataSet['Value'].sub(esa.values)
        d = pd.Series(d).abs().ewm(span=(n_channel * timeMultiplier), adjust=False, min_periods=(n_channel * timeMultiplier)-1).mean()
        ci = 100* (self.dataSet['Value'] - esa) / d
        tci = ci.ewm(span=(n_average * timeMultiplier), adjust=False, min_periods=(n_average * timeMultiplier)-1).mean()
        self.wtA = tci
        self.wtB = self.wtA.rolling(window=(crossover_sma_len * timeMultiplier),min_periods=(crossover_sma_len * timeMultiplier)-1).mean()
        self.wtDiff = self.wtA-self.wtB
        
        self.dataSet = pd.DataFrame({'wtA': self.wtA.values, 'wtB': self.wtB.values, 'diff': self.wtDiff.values, 'Date': priceData.index.values})
        self.dataSet.index = priceData.index

    def EWM(self, data, span):
        sma = data.rolling(window=span, min_periods=span).mean()[:span]
        rest = data[span:]
        return pd.concat([sma, rest]).ewm(span=span, adjust=False).mean()
    
    def GetSignals(self):
        return self.findCross(self.dataSet)

    def findCross(self, data):
        trading_positions_raw = data['wtA'] - data['wtB']
        trading_positions = trading_positions_raw.apply(np.sign)
        signals = trading_positions.diff()[trading_positions.diff() != 0].index.values
        signal = {'Type': [], 'Date': [], 'Value': [], 'O': [], 'H': [], 'L': [], 'C': [], 'PriceDate':[]}
        for elem in signals:
            if data['wtA'].loc[elem] != None and data['wtB'].loc[elem] != None and data['wtA'].loc[elem] < data['wtB'].loc[elem]:
                signal['Type'].append('Short')
                signal['Date'].append(pd.to_datetime(elem, format='%Y-%m-%d %H:%M'))
                signal['Value'].append(data['wtB'][elem])
                signal['PriceDate'].append(pd.to_datetime(elem + self.TimeDelta(), format='%Y-%m-%d %H:%M'))
                price = self.ohlc.loc[self.ohlc.index == elem + self.TimeDelta()]
                if(price.empty):
                    signal['O'].append(signal['O'][-1])
                    signal['H'].append(signal['H'][-1])
                    signal['L'].append(signal['L'][-1])
                    signal['C'].append(signal['C'][-1])
                else:
                    signal['O'].append(price['Open'][0])
                    signal['H'].append(price['High'][0])
                    signal['L'].append(price['Low'][0])
                    signal['C'].append(price['Close'][0])
            if data['wtA'].loc[elem] != None and data['wtB'].loc[elem] != None and data['wtA'].loc[elem] > data['wtB'].loc[elem]:
                signal['Type'].append('Long')
                signal['Date'].append(pd.to_datetime(elem, format='%Y-%m-%d %H:%M'))
                signal['Value'].append(data['wtA'][elem])
                signal['PriceDate'].append(pd.to_datetime(elem + self.TimeDelta(), format='%Y-%m-%d %H:%M'))
                price = self.ohlc.loc[self.ohlc.index == elem + self.TimeDelta()]
                if(price.empty):
                    signal['O'].append(signal['O'][-1])
                    signal['H'].append(signal['H'][-1])
                    signal['L'].append(signal['L'][-1])
                    signal['C'].append(signal['C'][-1])
                else:
                    signal['O'].append(price['Open'][0])
                    signal['H'].append(price['High'][0])
                    signal['L'].append(price['Low'][0])
                    signal['C'].append(price['Close'][0])
        return pd.DataFrame.from_dict(signal)

    def TimeDelta(self):
        if(self.timeFrame == "1h"):
            return np.timedelta64(1,'h')
        elif(self.timeFrame == "1d"):
            return np.timedelta64(1,'D')

    def Plot(self, axe):
        axe.plot(self.dataSet['Date'][:], self.dataSet['wtA'][:], color='g', linewidth=0.5)
        axe.plot(self.dataSet['Date'][:], self.dataSet['wtB'][:], color='r', linewidth=0.5)

    def PlotSignals(self, ax):
        signals = self.GetSignals()
        colors = np.where(signals["Type"] == 'Long', 'g', 'r')
        ax.scatter(signals['Date'], signals['Value'],  c=colors, s=6)

class WaveTrend():
    def __init__(self, params={}):
        self.title = "WaveTrend"
        priceData, chanelLenght, averageLenghtn = params.values()
        
        hlc3Data = list(map(hlc3, priceData['High'], priceData['Low'], priceData['Close']))
        hlc3Data = pd.DataFrame({'Value': hlc3Data, 'Date': priceData.index.values})
        hlc3Data['Date'] = hlc3Data['Date'].apply(lambda col: pd.to_datetime(col, format='%Y-%m-%d %H:%M'))
        hlc3Data = hlc3Data.set_index('Date', drop=True)
        
        self.dataSet = hlc3Data
        esa = self.dataSet['Value'].ewm(span=(chanelLenght), adjust=False,min_periods=(chanelLenght)-1).mean()
        d = self.dataSet['Value'].sub(esa.values)
        d = pd.Series(d).abs().ewm(span=(chanelLenght), adjust=False, min_periods=(chanelLenght)-1).mean()
        ci = (self.dataSet['Value'] - esa) / (0.015 * d)
        tci = ci.ewm(span=(averageLenghtn), adjust=False, min_periods=(averageLenghtn)-1).mean()
        self.wtA = tci
        self.wtB = self.wtA.rolling(window=(4),min_periods=(4)-1).mean()
        self.dataSet = pd.DataFrame({'wtA': self.wtA.values, 'wtB': self.wtB.values, 'Date': priceData.index.values})
        self.dataSet.index = priceData.index

    def EWM(self, data, span):
        sma = data.rolling(window=span, min_periods=span).mean()[:span]
        rest = data[span:]
        return pd.concat([sma, rest]).ewm(span=span, adjust=False).mean()
        
    def GetSignals(self):
        return self.findCross(self.dataSet)

    def findCross(self, data):
        trading_positions_raw = data['wtA'] - data['wtB']
        trading_positions = trading_positions_raw.apply(np.sign)
        signals = trading_positions.diff()[trading_positions.diff() != 0].index.values
        signal = {'Type': [], 'Date': [], 'Value': []}
        for elem in signals:
            if data['wtA'].loc[elem] != None and data['wtB'].loc[elem] != None and data['wtA'].loc[elem] < data['wtB'].loc[elem]:
                signal['Type'].append('Short')
                signal['Date'].append(pd.to_datetime(elem, format='%Y-%m-%d %H:%M'))
                signal['Value'].append(data['wtB'][elem])
            if data['wtA'].loc[elem] != None and data['wtB'].loc[elem] != None and data['wtA'].loc[elem] > data['wtB'].loc[elem]:
                signal['Type'].append('Long')
                signal['Date'].append(pd.to_datetime(elem, format='%Y-%m-%d %H:%M'))
                signal['Value'].append(data['wtA'][elem])
        return pd.DataFrame.from_dict(signal)

    def Plot(self, axe):
        #ax4.plot(date, wtdiff*10, color=yellow)
        axe.plot(self.dataSet['Date'][:], self.dataSet['wtA'][:], color='g', linewidth=0.5)
        axe.plot(self.dataSet['Date'][:], self.dataSet['wtB'][:], color='r', linewidth=0.5)

    def PlotSignals(self, ax):
        signals = self.GetSignals()
        colors = np.where(signals["Type"] == 'Long', 'g', 'r')
        ax.scatter(signals['Date'], signals['Value'],  c=colors, s=6)

class InverseFisherRSI():
    def __init__(self, params={}):
        self.title = "InverseRSI"
        priceData, rsi_lenght, smoothing_lenght, crossover_sma_len = params.values()
        
        length=rsi_lenght
        lengthwma=smoothing_lenght
        
        self.dataSet = pd.DataFrame(talib.RSI(np.asarray(priceData['Close'])), index=priceData.index)
        weights = np.arange(1,lengthwma+1) #this creates an array with integers 1 to 10 included
        v1 = 0.1 * (self.dataSet - 50)
        v2 = v1.rolling(lengthwma).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
        v2 = np.round(v2, decimals=3)
        ifish = (np.exp(2*v2)-1) / (np.exp(2*v2)+1)
        fast = ifish
        slow = fast.rolling(window=(crossover_sma_len),min_periods=(crossover_sma_len)-1).mean()
        self.dataSet = fast.merge(slow, right_index=True, left_index=True, suffixes=['Fast','Slow']).rename(columns={'0Fast':'Fast','0Slow':'Slow'})


    def GetSignals(self):
        return self.findCross(self.dataSet)

    def findCross(self, data):
        trading_positions_raw = data['Fast'] - data['Slow']
        trading_positions = trading_positions_raw.apply(np.sign)
        signals = trading_positions.diff()[trading_positions.diff() != 0].index.values
        signal = {'Type': [], 'Date': [], 'Value': []}
        for elem in signals:
            if data['Fast'].loc[elem] != None and data['Slow'].loc[elem] != None and data['Fast'].loc[elem] < data['Slow'].loc[elem]:
                signal['Type'].append('Short')
                signal['Date'].append(pd.to_datetime(elem, format='%Y-%m-%d %H:%M'))
                signal['Value'].append(data['Slow'][elem])
            if data['Fast'].loc[elem] != None and data['Slow'].loc[elem] != None and data['Fast'].loc[elem] > data['Slow'].loc[elem]:
                signal['Type'].append('Long')
                signal['Date'].append(pd.to_datetime(elem, format='%Y-%m-%d %H:%M'))
                signal['Value'].append(data['Fast'][elem])
        return pd.DataFrame.from_dict(signal)

    def Plot(self, axe):
        #axe.axhline(y=0.5, color='r', linewidth=0.5)
        #axe.axhline(y=-0.5, color='g', linewidth=0.5)
        axe.plot(self.dataSet, linewidth=0.5)

    def PlotSignals(self, ax):
        signals = self.GetSignals()
        colors = np.where(signals["Type"] == 'Long', 'g', 'r')
        ax.scatter(signals['Date'], signals['Value'],  c=colors, s=6)




