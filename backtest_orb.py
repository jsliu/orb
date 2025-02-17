# %%
import backtrader as bt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ib_async import IB, Stock
import asyncio  # Import asyncio for asynchronous operations
import nest_asyncio  # Import nest_asyncio to allow nested event loops
import matplotlib.dates as mdates  # Import for date formatting
import logging  # Import logging for debug information

# Ensure plots are displayed inline in Jupyter
# %matplotlib inline

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.CRITICAL)  # Set logging level to CRITICAL to disable logging

class ResampledDataObserver(bt.Observer):
    lines = ('datetime', 'open', 'high', 'low', 'close', 'volume')

    plotinfo = dict(plot=True, subplot=True)

    def next(self):
        dt = self.data1.datetime.datetime(0)
        open_ = self.data1.open[0]
        high = self.data1.high[0]
        low = self.data1.low[0]
        close = self.data1.close[0]
        volume = self.data1.volume[0]
        print(f"Resampled Data - Date: {dt}, Open: {open_}, High: {high}, Low: {low}, Close: {close}, Volume: {volume}")

# Define a custom analyzer to record portfolio values
class PortfolioValue(bt.Analyzer):
    def __init__(self):
        self.values = []

    def next(self):
        self.values.append((self.strategy.datetime.datetime(), self.strategy.broker.getvalue()))

# Define the ORBStrategy class
class ORBStrategy(bt.Strategy):
    params = (
        ('opening_range_minutes', 5),  # Opening range in minutes
        ('atr_multiplier', 1.5),  # ATR multiplier for stop orders
        ('atr_period_days', 14),  # ATR period for daily data
    )

    def __init__(self):
        self.first_bar = None
        self.first_bar_high = None
        self.first_bar_low = None
        self.buy_signal = False
        self.sell_signal = False

        # Add ATR indicator based on resampled daily data
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.params.atr_period_days)

    def next(self):
        # Log the current data values
        logging.debug(f"Date: {self.data.datetime.datetime(0)}, Open: {self.data.open[0]}, High: {self.data.high[0]}, Low: {self.data.low[0]}, Close: {self.data.close[0]}, Volume: {self.data.volume[0]}")

        # Check if the current bar is the first bar of the day
        if self.data.datetime.date(0) != self.data1.datetime.date(-1):
            self.first_bar = True
            self.buy_signal = False
            self.sell_signal = False

        # If it's the first 5-minute bar, record the high and low prices
        if self.first_bar:
            self.first_bar = False
            self.first_bar_high = self.data.high[0]
            self.first_bar_low = self.data.low[0]
            if self.data.close[0] > self.data.open[0]:
                self.buy_signal = True  
            elif self.data.close[0] < self.data.open[0]:
                self.sell_signal = True

        # Monitor prices throughout the day
        if self.buy_signal and self.data.high[0] > self.first_bar_high:
            self.buy(size=100)  # Buy 100 shares
            self.buy_signal = False
            self.stop_loss = self.data.close[0] - self.atr[0] * self.params.atr_multiplier
        
        if self.sell_signal and self.data.low[0] < self.first_bar_low:
            self.sell(size=100)  # Sell 100 shares
            self.sell_signal = False
            self.stop_loss = self.data.close[0] + self.atr[0] * self.params.atr_multiplier

        # Implement stop-loss orders
        if self.position:
            if self.position.size > 0 and self.data.close[0] < self.stop_loss:
                self.close()  # Close the position
            elif self.position.size < 0 and self.data.close[0] > self.stop_loss:
                self.close()  # Close the position

# Asynchronous function to connect to Interactive Brokers
async def connect_ib():
    ib = IB()
    while True:
        try:
            ib.connect('127.0.0.1', 7497, clientId=2)
            print('Connected to Interactive Brokers')
            return ib
        except Exception as e:
            print(f"Connection failed: {e}. Retrying in 5 seconds")
            await asyncio.sleep(5)

# Asynchronous function to fetch historical data
async def fetch_historical_data(ticker, duration, bar_size):
    ib = await connect_ib()
    contract = Stock(ticker, 'SMART', 'USD')
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,
    )
    ib.disconnect()
    
    # Convert bars to DataFrame and ensure correct format
    df = pd.DataFrame([bar.__dict__ for bar in bars])
    df['datetime'] = pd.to_datetime(df['date'])  # Ensure the date column is in datetime format
    
    # Convert to the correct timezone (e.g., US/Eastern)
    df['datetime'] = df['datetime'].dt.tz_convert('US/Eastern')
    
    df.set_index('datetime', inplace=True)  # Set the datetime column as the index
    df = df[['open', 'high', 'low', 'close', 'volume']]  # Ensure the DataFrame has the required columns
    return df

# Function to plot performance using Seaborn
def plot_performance(portfolio_values):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    dates, values = zip(*portfolio_values)
    sns.lineplot(x=dates, y=values)
    plt.title('Portfolio Performance')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    plt.show()

# Function to run the backtest
def run_backtest(ticker):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ORBStrategy)  # Add the ORB strategy
    cerebro.addanalyzer(PortfolioValue, _name='portfolio_value')  # Add the custom analyzer
    # cerebro.addobserver(ResampledDataObserver)  # Add the custom observer

    # Fetch historical data for 1-minute bars
    df_1min = asyncio.run(fetch_historical_data(ticker, '30 D', '5 mins'))
    
    # Align the data to start at the correct time
    df_min = df_1min[df_1min.index.time >= pd.to_datetime('09:30:00').time()]
    
    data_min = bt.feeds.PandasData(dataname=df_1min)  # Convert DataFrame to Backtrader data feed
    
    cerebro.adddata(data_min)  # Add 1-minute data feed to Cerebro

    # Resample the 1-minute data to daily data
    cerebro.resampledata(data_min, timeframe=bt.TimeFrame.Days)

    cerebro.broker.setcash(100000.0)  # Set initial cash
    cerebro.broker.setcommission(commission=0.001)  # Set commission

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()  # Run the backtest
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Collect performance data
    portfolio_values = results[0].analyzers.portfolio_value.values

    # Plot performance using Seaborn
    plot_performance(portfolio_values)
    # cerebro.plot(iplot=True, volume=False)  # Ensure plot is displayed inline

# Main entry point
if __name__ == '__main__':
    run_backtest('NVDA')  # Run backtest for NVDA ticker

# %%
