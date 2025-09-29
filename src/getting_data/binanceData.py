from binance.client import Client
import pandas as pd

client = Client()

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1HOUR
start_str = "5 Nov, 2021"
end_str = "2 Sep, 2025"

klines = client.get_historical_klines(symbol, interval, start_str, end_str)

df = pd.DataFrame(
    klines,
    columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ],
)

# Keep only timestamp & close
df = df[["timestamp", "close"]]

# Convert timestamp → datetime, close → float
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df["close"] = df["close"].astype(float)

df.to_csv("lst_price/BTCUSDT_closing.csv", index=False)

print(df.head())
