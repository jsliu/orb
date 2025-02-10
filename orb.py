import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
from ib_async import IB, Index, Stock, MarketOrder

nest_asyncio.apply()
            
# Function to check breakout
def check_breakout(current_price, opening_range):
    if current_price > opening_range.high:
        return 'buy'
    elif current_price < opening_range.low:
        return 'sell'

    return 'hold'

# Function to place orders with ART-based stop-loss
async def place_order(ib, action, contract, quantity, atr):
    order = MarketOrder(action, quantity)
    trade = await ib.place_order(contract, order)
    # Set stop-loss order based on ATR
    if action == 'buy':
        stop_loss_price = trade.orderStatus.avgFillPrice - 1.5 * atr
        stop_order = MarketOrder('sell', quantity, stopPrice=stop_loss_price)
        await ib.place_order(contract, stop_order)
    else:
        stop_loss_price = trade.orderStatus.avgFillPrice + 1.5 * atr
        stop_order = MarketOrder('buy', quantity, stopPrice=stop_loss_price)
        await ib.place_order(contract, stop_order)
    return trade

# Real-time bar update event handler
async def on_bar_update(ib, contract, opening_range, atr, bars):
    current_price = bars[-1].close
    action = check_breakout(current_price, opening_range)
    if action != 'hold':
        await place_order(action, contract, 1, atr)  # Adjust quantity as needed 

async def connect_ib():
    ib = IB()
    while True:
        try:
            ib.connect('127.0.0.1', 7497, clientId=1)
            print('Connected to Interactive Brokers')
            return ib
        except Exception as e:
            print(f"Connection failed: {e}. Retrying in 5 seconds")
            await asyncio.sleep(5)

async def main():

    # Connect to Interactive Brokers
    ib = await connect_ib()

    # Define the index contract
    # contract = Index('SPX', 'CBOE', 'USD')
    # contract = Stock('NVDA', 'SMART', 'USD')
    
    sp500_ticks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

    # Define criteria for the Opening Range Breakout (ORB) strategy
    min_price = 5
    min_adv = 1_000_000
    min_atr = 0.5
    min_rvol = 1.5

    selected_stocks = []

    for ticker in sp500_ticks:
        contract = Stock(ticker, 'SMART', 'USD')

        try:
            # Fetch historical data for the opening range
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='14 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
            )

            # Convert bars to DataFrame
            df = pd.DataFrame([bars.__dict__ for bars in bars])

            # if df.empty:
                # continue    
        
            # Calculate Average True Range (ATR)
            df['high_low'] = df.high - df.low
            df['high_close'] = (df.high - df.close.shift()).abs()
            df['low_close'] = (df.low - df.close.shift()).abs()
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df.true_range.rolling(14).mean()
            atr = df.atr.iloc[-1]

            # Calculate Average Daily Volume (ADV)
            average_volume = df.volume.mean()

            # Fetch current day's volume
            current_day_bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
            )
            current_volume = current_day_bars[-1].volume

            # Calculate relative volume (RVOL)
            relative_volume = current_volume / average_volume
        
            # Check if the stock meets the criteria
            if (df['open'].iloc[-1] > min_price and
                average_volume > min_adv and
                atr > min_atr and
                relative_volume > min_rvol):

                print(f'Stock {ticker} meets the criteria for ORB strategy')

                # Fetch historical data for the opening range
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime='',
                    durationStr='1 D',
                    barSizeSetting='5 mins',
                    whatToShow='TRADES',
                    useRTH=True,
                )
                df = pd.DataFrame([bars.__dict__ for bars in bars])
                # Fisrt 5 min bar
                opening_range = df.iloc[:1]

                # Request real-time bars and set up event handler
                bars = ib.reqRealTimeBars(contract, 5, 'TRADES', False, [])
                bars.updateEvent += lambda bars, hasNewBar: asyncio.create_task(on_bar_update(ib, contract, opening_range, atr, bars))
                selected_stocks.append(contract)

                # Throttle requests to avoid rate-limiting
                await asyncio.sleep(1)
        except Exception as e:
            print(f'Error fetching data for {ticker}: {e}')
    if selected_stocks:
        ib.run()

# Run the main function
asyncio.run(main())