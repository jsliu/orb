import backtrader as bt
import pandas as pd

class PutsOverwritingStrategy(bt.Strategy):
    params = (
        ('target_dte', 30),
        ('target_delta', 0.30),
    )

    def __init__(self):
        self.cash = self.broker.get_cash()
        self.positions = []

    def next(self):
        # Implement the logic to sell puts based on the target DTE and delta
        # Use self.data to access the historical options data
        pass

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(25000)

    options_data = pd.read_csv('/Users/zhenliu/Documents/project/orb/options_data.csv')
    data = bt.feeds.PandasData(dataname=options_data)
    cerebro.adddata(data)

    cerebro.addstrategy(PutsOverwritingStrategy)
    cerebro.run()
    cerebro.plot()
