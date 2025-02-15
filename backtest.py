# %%
# backtest option overwriting strategy
import datetime
import asyncio
from ib_async import IB, Stock, util
import backtrader as bt
import yfinance as yf

class OptionOverwritingStrategy(bt.Strategy):
    params = (
        ('start_cash', 100000),
        ('position_size', 100),
        ('sma_period', 20),
    )

    def __init__(self):
        self.order = None
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)

    def next(self):
        if self.order:
            return

        if self.data.datetime.date(0).weekday() == 0:  # Every Monday
            if self.data.close[0] > self.sma[0]:
                # Write call options if price is above SMA
                self.order = self.sell(size=self.params.position_size)
            else:
                # Write put options if price is below SMA
                self.order = self.buy(size=self.params.position_size)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None

async def fetch_ib_data(symbol='NVDA'):
    ib = IB()
    await ib.connectAsync('127.0.0.1', 7497, clientId=1)

    # stock = Stock('TQQQ', 'SMART', 'USD')
    stock = Stock(symbol, 'SMART', 'USD')
    await ib.qualifyContractsAsync(stock)

    bars = await ib.reqHistoricalDataAsync(
        stock, endDateTime='2022-12-31', durationStr='1 Y',
        barSizeSetting='1 day', whatToShow='MIDPOINT', useRTH=True
    )

    await ib.disconnectAsync()

    return bars

def fetch_yahoo_data(symbol, fromdate, todate):
    try:
        # Fetch historical data from Yahoo Finance
        data = yf.download(symbol, start=fromdate, end=todate)
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        # Convert to backtrader feed
        data.columns = ['_'.join(col).strip().lower() for col in data.columns.values]
        data_feed = bt.feeds.PandasData(dataname=data)
        return data_feed
    except Exception as e:
        raise ValueError(f"Error fetching data for symbol {symbol}: {e}")

if __name__ == '__main__':
    use_ib = False  # Set to False to use Yahoo Finance data
    symbol = 'NVDA'
    cerebro = bt.Cerebro()
    cerebro.addstrategy(OptionOverwritingStrategy)

    if use_ib:
        loop = asyncio.get_event_loop()
        bars = loop.run_until_complete(fetch_ib_data(symbol))

        data = bt.feeds.PandasData(dataname=util.df(bars))
    else:
        fromdate = datetime.datetime(2022, 1, 1)
        todate = datetime.datetime(2022, 12, 31)
        data = fetch_yahoo_data(symbol, fromdate, todate)

        cerebro.adddata(data)

    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# %%
