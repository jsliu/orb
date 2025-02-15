# %%
import yfinance as yf

def check_stock_availability(symbol):
    stock = yf.Ticker(symbol)
    try:
        # Fetch stock info to check if the symbol is valid
        info = stock.info
        if 'regularMarketPrice' in info:
            print(f"Stock {symbol} is available on Yahoo Finance.")
        else:
            print(f"Stock {symbol} is not available on Yahoo Finance.")
    except Exception as e:
        print(f"Error checking stock {symbol}: {e}")

if __name__ == '__main__':
    symbol = 'TQQQ'  # Replace with the stock symbol you want to check
    check_stock_availability(symbol)

# %%
