import time
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator

def fetch_binance_klines(symbol='BTCUSDT', interval='15m',
                         start_ts=None, end_ts=None, limit=1000):
    """
    Fetch one batch of up to `limit` candles from Binance.
    Timestamps in milliseconds.
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
        'startTime': start_ts,
        'endTime': end_ts,
    }
    resp = requests.get(url, params={k: v for k, v in params.items() if v is not None})
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','num_trades',
        'taker_base_vol','taker_quote_vol','ignore'
    ])
    df['open_time']  = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

# ─── 1) Download full 2020–2025 history ─────────────────────────────────────────
start = pd.to_datetime("2020-01-01T00:00:00Z")
end   = pd.to_datetime("2025-07-28T00:00:00Z")
all_data    = []
batch_start = int(start.timestamp() * 1000)
batch_count = 0

print(f"Starting download from {start.isoformat()} to {end.isoformat()}")
while True:
    batch = fetch_binance_klines(
        start_ts=batch_start,
        end_ts=int(end.timestamp() * 1000),
        limit=1000
    )
    if batch.empty:
        print("No more data to fetch, exiting loop.")
        break

    batch_count += 1
    first = batch['open_time'].iloc[0]
    last  = batch['close_time'].iloc[-1]
    print(f" Batch {batch_count}: fetched {len(batch)} candles from {first} to {last}")
    all_data.append(batch)

    # advance to one ms after the last candle's close_time
    batch_start = int((last + pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    time.sleep(0.2)

df = pd.concat(all_data, ignore_index=True)
print(f"Total candles fetched: {len(df)}")

# ─── 2) Compute RSI ────────────────────────────────────────────────────────────
print("Computing 14-period RSI")
df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

# ─── 3) Generate positions & extract trades ────────────────────────────────────
print("Generating signals and extracting trades")
df['position'] = 0
entry_price   = None

for i in range(1, len(df)):
    price    = df.at[i, 'close']
    prev_pos = df.at[i-1, 'position']
    r        = df.at[i, 'rsi']

    # flat → entry
    if prev_pos == 0:
        if r < 30:
            df.at[i, 'position'] = 1
            entry_price = price
        elif r > 70:
            df.at[i, 'position'] = -1
            entry_price = price

    # long → exit on RSI crossing above 50
    elif prev_pos == 1:
        if (df.at[i-1, 'rsi'] < 50 <= r):
            df.at[i, 'position'] = 0
        else:
            df.at[i, 'position'] = 1

    # short → exit on RSI crossing below 50
    elif prev_pos == -1:
        if (df.at[i-1, 'rsi'] > 50 >= r):
            df.at[i, 'position'] = 0
        else:
            df.at[i, 'position'] = -1

# extract trades
trades = []
pos       = 0
entry_idx = None

for idx, row in df.iterrows():
    if pos == 0 and row['position'] != 0:
        pos       = row['position']
        entry_idx = idx
    elif pos != 0 and row['position'] == 0:
        ep  = df.at[entry_idx, 'close']
        xp  = row['close']
        pnl = (xp - ep) / ep * (1 if pos == 1 else -1)
        trades.append({
            'entry_time' : df.at[entry_idx, 'open_time'],
            'exit_time'  : row['open_time'],
            'side'       : 'Long' if pos == 1 else 'Short',
            'entry_price': ep,
            'exit_price' : xp,
            'pnl_pct'    : pnl * 100
        })
        pos = 0

trades_df = pd.DataFrame(trades)
print(f"Total trades extracted: {len(trades_df)}")

# ─── 4) Calculate monthly win rates ─────────────────────────────────────────────
print("Calculating monthly win rate and trade counts")
trades_df['year']  = trades_df['entry_time'].dt.year
trades_df['month'] = trades_df['entry_time'].dt.month

monthly_summary = trades_df.groupby(['year','month']).agg(
    num_trades = ('pnl_pct', 'size'),
    win_rate   = ('pnl_pct', lambda x: (x > 0).mean())
).reset_index().sort_values(['year','month'])

print("Monthly summary (year, month, trades, win_rate):")
print(monthly_summary.to_string(index=False))

# ─── 4b) Calculate yearly win rates ──────────────────────────────────────────────
yearly_summary = trades_df.groupby('year').agg(
    num_trades = ('pnl_pct', 'size'),
    win_rate   = ('pnl_pct', lambda x: (x > 0).mean())
).reset_index().sort_values('year')

print("\nYearly summary (year, trades, win_rate):")
print(yearly_summary.to_string(index=False))

# ─── 4c) Overall win rate ───────────────────────────────────────────────────────
overall_win_rate = (trades_df['pnl_pct'] > 0).mean()
print(f"\nOverall win rate: {overall_win_rate * 100:.2f}%")

# ─── 5) Save results ────────────────────────────────────────────────────────────
trades_df.to_csv('backtest_results.csv', index=False)
print("Saved detailed trades to 'backtest_results.csv'")
monthly_summary.to_csv('backtest_summary.csv', index=False)
print("Saved monthly summary to 'backtest_summary.csv'")
