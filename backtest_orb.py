# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates as mdates  # Add this import
from ib_async import IB, Stock
import asyncio
import nest_asyncio
import logging
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

# Ensure plots are displayed inline in Jupyter
# %matplotlib inline

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.CRITICAL)  # Set logging level to CRITICAL to disable logging

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
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})  # Capitalize column names
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Ensure the DataFrame has the required columns
    return df

# ADV function
def ADV(data, period=14):
    return data[data>0].rolling(window=period).mean()

# RVol function
def RVol(data, period=14):
    return data[data>0]/data[data>0].rolling(window=period).std()

# Average True Range (ATR) function
def ATR(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    # need to shift the result by 1 to align with the data
    return atr

# Define the ORBStrategy class
class ORBStrategy(Strategy):
    opening_period = 5
    atr_multiplier = 0.1
    roll_period_days = 14
    min_adv = 500_000
    min_atr = 0.3
    min_rvol = 1

    def init(self):
        self.first_bar = None
        self.first_bar_open = None
        self.first_bar_close = None
        self.first_bar_high = None
        self.first_bar_low = None
        self.buy_signal = False
        self.sell_signal = False
        self.previous_day = None

        # Use resample_apply to calculate ATR and ADV
        self.atr = resample_apply('D', ATR, self.data.df, period=self.roll_period_days)
        self.adv = resample_apply('D', ADV, self.data.df.Volume, period=self.roll_period_days)
        self.rvol = resample_apply('D', RVol, self.data.df.Volume, period=self.roll_period_days)

    def next(self):
        # Log the current data values
        logging.debug(f"Date: {self.data.index[-1]}, Open: {self.data.Open[-1]}, High: {self.data.High[-1]}, Low: {self.data.Low[-1]}, Close: {self.data.Close[-1]}, Volume: {self.data.Volume[-1]}")
        
        current_date = self.data.index[-1].date()

        # Check if the current bar is the first bar of the day
        if self.previous_day != current_date:
            self.previous_day = current_date
            self.first_range_bar = []
            self.first_bar = True
            self.buy_signal = False
            self.sell_signal = False

        if len(self.first_range_bar) < self.opening_period:
            self.first_range_bar.append({
                'open': self.data.Open[-1],
                'high': self.data.High[-1],
                'low': self.data.Low[-1],
                'close': self.data.Close[-1],
                })

        if self.first_bar and len(self.first_range_bar) == self.opening_period:
            self.first_bar_open = self.first_range_bar[0]['open']
            self.first_bar_close = self.first_range_bar[-1]['close']
            self.first_bar_high = max([x['high'] for x in self.first_range_bar])
            self.first_bar_low = min([x['low'] for x in self.first_range_bar])

            self.first_bar = False
            
            if self.atr[-1] > self.min_atr and self.rvol[-1] > self.min_rvol and self.adv[-1] > self.min_adv:
                if self.first_bar_close > self.first_bar_open:
                    self.buy_signal = True
                elif self.first_bar_close < self.first_bar_open:
                    self.sell_signal = True        

        # If it's the first bar, record the high and low prices
        if self.buy_signal and self.data.Open[-1] > self.first_bar_high:
            self.buy(size=10)  # Buy 100 shares
            self.buy_signal = False
            self.stop_loss = self.data.Close[-1] - self.atr[-1] * self.atr_multiplier
            
        if self.sell_signal and self.data.Open[-1] < self.first_bar_low:
            self.sell(size=10)  # Sell 100 shares
            self.sell_signal = False
            self.stop_loss = self.data.Close[-1] + self.atr[-1] * self.atr_multiplier

        # Implement stop-loss orders
        if self.position:
            if self.position.is_long and self.data.Close[-1] < self.stop_loss:
                self.position.close()  # Close the position
            elif self.position.is_short and self.data.Close[-1] > self.stop_loss:
                self.position.close()  # Close the position

        # Close positions at the end of the trading day
        bar_size_minutes = (self.data.index[1] - self.data.index[0]).seconds // 60
        # Close positions 2 bar size minutes before the market closes, because it closes after the last bar
        market_close_time = (pd.to_datetime('16:00:00') - pd.Timedelta(minutes=2 * bar_size_minutes)).time()
        if self.data.index[-1].time() >= market_close_time:
            if self.position:
                self.position.close()


# Function to plot performance using Seaborn
def plot_performance(portfolio_values):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    dates, values = zip(*portfolio_values)
    # Resample to daily frequency
    df = pd.DataFrame({'Date': dates, 'Value': values})
    df.set_index('Date', inplace=True)
    df = df.resample('D').last().dropna()
    
    sns.lineplot(x=df.index, y=df['Value'])
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()
    plt.show()

# Function to run the backtest
def run_backtest(ticker, bar_size='1 min', cash=100000, commission=.001, opening_period=5):
    # Fetch historical data for the specified bar size
    df_minute = asyncio.run(fetch_historical_data(ticker, '30 D', bar_size=bar_size))
    
    # Align the data to start at the correct time
    # df_minute = df_minute[df_minute.index.time >= pd.to_datetime('09:30:00').time()]
    
    # Run the backtest
    bt = Backtest(df_minute, ORBStrategy, cash=cash, commission=commission)
    ORBStrategy.opening_period = opening_period
    stats = bt.run()
    
    # Plot the results
    bt.plot()

    # Collect performance data
    portfolio_values = list(zip(stats['_equity_curve'].index, stats['_equity_curve']['Equity']))

    # Plot performance using Seaborn
    # plot_performance(portfolio_values)
    return stats

# Main entry point
if __name__ == '__main__':
    stats = run_backtest('MMM', bar_size='1 min', cash=20000, opening_period=5)
    print(stats)

# %%
