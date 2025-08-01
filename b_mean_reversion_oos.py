import os
import time
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator

# ─── Utility to fetch Binance klines (5m interval by default) ────────────────────
def fetch_binance_klines(symbol='BTCUSDT', interval='5m', start_ts=None, end_ts=None, limit=1000):
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
    # convert timestamps and types
    df['open_time']  = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

# ─── 0) Cache download to avoid re-fetching ─────────────────────────────────────
cache_file = 'b_data.pkl'
if os.path.exists(cache_file):
    print(f"Loading cached data from '{cache_file}'...")
    b_data = pd.read_pickle(cache_file)
else:
    # ─── 1) Download full 2020–2025 history (5m candles) ─────────────────────────
    start = pd.to_datetime("2020-01-01T00:00:00Z")
    end   = pd.to_datetime("2025-07-28T00:00:00Z")
    all_data    = []
    batch_start = int(start.timestamp() * 1000)
    batch_count = 0

    print(f"Starting download (5m candles) from {start.isoformat()} to {end.isoformat()}")
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

    # concatenate into one DataFrame and cache
    df = pd.concat(all_data, ignore_index=True)
    print(f"Total candles fetched: {len(df)}")
    b_data = df.copy()
    b_data.to_pickle(cache_file)
    print(f"Saved cached data to '{cache_file}'.")

# ─── 2) Compute 70-period RSI ──────────────────────────────────────────────────
print("Computing 14-period RSI")
b_data['rsi'] = RSIIndicator(b_data['close'], window=14).rsi()

# ─── 3) Generate positions & extract trades ────────────────────────────────────
print("Generating signals and extracting trades")
b_data['position'] = 0
entry_idx = None

for i in range(1, len(b_data)):
    r = b_data.at[i, 'rsi']
    prev_pos = b_data.at[i-1, 'position']

    # flat → entry
    if prev_pos == 0:
        if r < 30:
            b_data.at[i, 'position'] = 1
            entry_idx = i
        elif r > 70:
            b_data.at[i, 'position'] = -1
            entry_idx = i

    # long → exit on RSI crossing above 50
    elif prev_pos == 1:
        if b_data.at[i-1, 'rsi'] < 50 <= r:
            b_data.at[i, 'position'] = 0
        else:
            b_data.at[i, 'position'] = 1

    # short → exit on RSI crossing below 50
    elif prev_pos == -1:
        if b_data.at[i-1, 'rsi'] > 50 >= r:
            b_data.at[i, 'position'] = 0
        else:
            b_data.at[i, 'position'] = -1

# extract trades
trades = []
pos = 0
entry_idx = None

for idx, row in b_data.iterrows():
    if pos == 0 and row['position'] != 0:
        pos = row['position']
        entry_idx = idx
    elif pos != 0 and row['position'] == 0:
        entry_price = b_data.at[entry_idx, 'close']
        exit_price  = row['close']
        pnl = (exit_price - entry_price) / entry_price * (1 if pos == 1 else -1)
        trades.append({
            'entry_time' : b_data.at[entry_idx, 'close_time'],
            'exit_time'  : b_data.at[idx, 'close_time'],
            'side'       : 'Long' if pos == 1 else 'Short',
            'entry_price': entry_price,
            'exit_price' : exit_price,
            'pnl_pct'    : pnl * 100
        })
        pos = 0

trades_df = pd.DataFrame(trades)
print(f"Total trades extracted: {len(trades_df)}")

# ─── 4) Calculate monthly & yearly win rates ───────────────────────────────────
print("Calculating monthly win rate and trade counts")
trades_df['year']  = trades_df['entry_time'].dt.year
trades_df['month'] = trades_df['entry_time'].dt.month

monthly_summary = trades_df.groupby(['year','month']).agg(
    num_trades = ('pnl_pct', 'size'),
    win_rate   = ('pnl_pct', lambda x: (x > 0).mean())
).reset_index().sort_values(['year','month'])
print(monthly_summary.to_string(index=False))

yearly_summary = trades_df.groupby('year').agg(
    num_trades = ('pnl_pct', 'size'),
    win_rate   = ('pnl_pct', lambda x: (x > 0).mean())
).reset_index().sort_values('year')
print(yearly_summary.to_string(index=False))

overall_win_rate = (trades_df['pnl_pct'] > 0).mean()
print(f"Overall win rate: {overall_win_rate * 100:.2f}%")

# ─── 5) Save results ────────────────────────────────────────────────────────────
trades_df.to_csv('backtest_results.csv', index=False)
print("Saved detailed trades to 'backtest_results.csv'")
monthly_summary.to_csv('backtest_summary.csv', index=False)
print("Saved monthly summary to 'backtest_summary.csv'")