# %%
import pandas as pd
import numpy as np
import vectorbt as vbt
import asyncio
from ib_async import IB, Stock
import nest_asyncio
import logging

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
    # df.columns = df.columns.str.capitalize()  # Capitalize column names
    df = df[['open', 'high', 'low', 'close', 'volume']]  # Ensure the DataFrame has the required columns
    return df

# Function to get the first opening period data of each day
def get_first_bar(df, opening_period):
    first_opening_period_data = df.groupby(df.index.date).head(opening_period)
    # Define entry and exit signals
    first_bar  = first_opening_period_data.groupby(first_opening_period_data.index.date).agg({
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min'
    })
    return first_bar

# Function to find the entry point when the price is above the high of the first bar or below the low of the first bar
def find_entry_point(df_minute, first_bar, min_atr, min_rvol, min_adv, roll_period_days=14):
    # Agrregate data to daily
    df_daily = df_minute.groupby(df_minute.index.date).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # df_daily.index = pd.to_datetime(df_daily.index).tz_localize('UTC').tz_convert('US/Eastern')  # Add timezone to index

    # Calculate indicators
    atr = vbt.ATR.run(df_daily['high'], df_daily['low'], df_daily['close'], window=roll_period_days).atr.shift()
    adv = df_daily['volume'].rolling(window=roll_period_days).mean().shift()
    rvol = df_daily['volume'].shift() / adv
    
    long_enteries = pd.Series(False, index=df_minute.index, name='long')
    short_enteries = pd.Series(False, index=df_minute.index, name='short')
    dates = np.unique(df_minute.index.date)

    for d in dates:
        daily_data = df_minute.loc[df_minute.index.date == d]
        first_bar_data = first_bar.loc[d]
        first_price_above_high = daily_data['open'][daily_data['open'] > first_bar_data['high']]
        first_price_below_low = daily_data['open'][daily_data['open'] < first_bar_data['low']]
        buy_signal = first_bar_data['close'] > first_bar_data['open']
        sell_signal = first_bar_data['close'] < first_bar_data['open']  
        
        if atr.loc[d] > min_atr and adv.loc[d] > min_adv and rvol.loc[d] > min_rvol:
            if buy_signal and len(first_price_above_high) > 0:
                long_enteries.loc[first_price_above_high.index[0]] = True
            if sell_signal and len(first_price_below_low) > 0:
                short_enteries.loc[first_price_below_low.index[0]] = True
    
    atr.index = pd.to_datetime(atr.index).tz_localize('UTC').tz_convert('US/Eastern')  # Add timezone to index
    stop_loss = (atr * 0.1).reindex(df_minute.index, method='ffill')
    return long_enteries, short_enteries, stop_loss 

# Function to get the time of the last bar of each day
def get_exit_signal(df):
    last_bar_time = df.groupby(df.index.date).apply(lambda x: x.index[-1])
    last_bar_signal = df.index.isin(last_bar_time)
    return last_bar_signal

# Function to run the backtest using vectorbt
def run_backtest(ticker, 
                 bar_size='1 min', 
                 min_atr = 0.5,
                 min_rvol = 1.5,   
                 min_adv = 1_000_000,
                 cash=100000, 
                 commission=0.001, 
                 opening_period=5):
    # Fetch historical data for the specified bar size
    df_minute = asyncio.run(fetch_historical_data(ticker, '30 D', bar_size=bar_size))

    # Calculate position signals
    first_bar = get_first_bar(df_minute, opening_period)
    long, short, stop_loss = find_entry_point(df_minute, first_bar=first_bar, min_atr=min_atr, min_rvol=min_rvol, min_adv=min_adv)
    exit = get_exit_signal(df_minute)
    
    # Run the backtest
    pf = vbt.Portfolio.from_signals(
        df_minute['close'],
        entries=long,
        exits=exit,
        short_entries=short,
        short_exits=exit,
        init_cash=cash,
        fees=commission,
        sl_stop=stop_loss,
        freq='T'  # Set frequency to minute
    )
    
    # Ensure the frequency is set for Sharpe ratio calculation
    stats = pf.stats()

    # Plot the results
    pf.plot().show()
    
    return stats

# Main entry point
if __name__ == '__main__':
    stats = run_backtest('MMM', bar_size='1 min', 
                         min_atr=0.2,
                         min_rvol=0.5,   
                         min_adv=100_000,
                         cash=20000, 
                         opening_period=5)
    print(stats)

# %%
