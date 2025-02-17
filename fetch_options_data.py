import pandas as pd
from datetime import timedelta
from ib_async import IB, Stock, Option, util
import asyncio
import nest_asyncio
import random

nest_asyncio.apply()

async def fetch_options_data(ticker_symbol, start_date, end_date, target_dte, target_delta):
    ib = IB()
    client_id = random.randint(1, 10000)  # Use a random client ID
    ib.connect('127.0.0.1', 7497, clientId=client_id)

    stock = Stock(ticker_symbol, 'SMART', 'USD')
    options_data = []

    for date in pd.date_range(start=start_date, end=end_date, freq='B'):
        target_date = date + timedelta(days=target_dte)
        expirations = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
        
        if not expirations or not expirations[0].expirations:
            print(f"No expiration data available for {date}. Skipping.")
            continue

        closest_expiration = min(expirations[0].expirations, key=lambda x: abs(pd.to_datetime(x) - target_date))

        option_chain = await ib.reqContractDetails(Option(stock.symbol, closest_expiration, 0, 'P'))
        puts = [opt.contract for opt in option_chain]

        puts_data = await ib.reqMktData(puts, '', False, False)
        await asyncio.sleep(1)
        puts_df = util.df(puts_data)
        puts_df['delta'] = puts_df['bidGreeks'].apply(lambda x: x.delta if x else 0)
        target_put = puts_df.iloc[(puts_df['delta'] - target_delta).abs().argsort()[:1]]

        options_data.append(target_put)

    ib.disconnect()
    return pd.concat(options_data)

if __name__ == "__main__":
    ticker_symbol = 'NVDA'
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    target_dte = 30
    target_delta = 0.30

    options_data = asyncio.run(fetch_options_data(ticker_symbol, start_date, end_date, target_dte, target_delta))
    options_data.to_csv('/Users/zhenliu/Documents/project/orb/options_data.csv', index=False)
