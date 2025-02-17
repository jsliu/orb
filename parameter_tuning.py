# %%
import pandas as pd
import numpy as np
import asyncio
from backtesting import Backtest
from backtest_orb import fetch_historical_data, ORBStrategy

# Function to run the backtest with given parameters
def run_backtest_with_params(ticker, cash, opening_range_minutes, atr_multiplier, atr_period_days):
    class TunedORBStrategy(ORBStrategy):
        def init(self):
            super().init()
            self.opening_range_minutes = opening_range_minutes
            self.atr_multiplier = atr_multiplier
            self.atr_period_days = atr_period_days

    df_minute = asyncio.run(fetch_historical_data(ticker, '30 D', '1 min'))
    bt = Backtest(df_minute, TunedORBStrategy, cash=cash, commission=.001)
    stats = bt.run()
    return stats

# Function to perform parameter tuning
def tune_parameters(ticker, cash, param_grid):
    best_stats = None
    best_params = None
    best_performance = -np.inf

    for opening_range_minutes in param_grid['opening_range_minutes']:
        for atr_multiplier in param_grid['atr_multiplier']:
            for atr_period_days in param_grid['atr_period_days']:
                stats = run_backtest_with_params(ticker, cash, opening_range_minutes, atr_multiplier, atr_period_days)
                performance = stats['Return [%]']  # Use return percentage as the performance metric

                if performance > best_performance:
                    best_performance = performance
                    best_stats = stats
                    best_params = {
                        'opening_range_minutes': opening_range_minutes,
                        'atr_multiplier': atr_multiplier,
                        'atr_period_days': atr_period_days
                    }

    return best_params, best_stats

# Main entry point
if __name__ == '__main__':
    param_grid = {
        'opening_range_minutes': [5, 10, 15],
        'atr_multiplier': [0.1, 0.2, 0.3],
        'atr_period_days': [14, 21, 28]
    }
    best_params, best_stats = tune_parameters('AAPL', 20000, param_grid)
    print("Best Parameters:", best_params)
    print("Best Stats:", best_stats)

# %%
