# utils.py

#!/usr/bin/env python3

import os
import glob
import logging
import numpy as np
import pandas as pd
import yfinance as yf

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_df_list(
    data_dir: str             = "data",
    tickers   : list[str]    | None = None,
    start     : str          | None = None,
    end       : str          | None = None,
    reload    : bool                  = False
) -> list[pd.DataFrame]:
    logging.info("Entering load_df_list")
    os.makedirs(data_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    logging.info(f"  Found {len(paths)} CSV(s) in '{data_dir}'")

    if paths and not reload:
        dfs = []
        for fp in paths:
            logging.info(f"  Loading local CSV: {fp}")
            df = pd.read_csv(fp, parse_dates=[0], index_col=0)
            df = df.apply(pd.to_numeric, errors="coerce")
            n0 = len(df); df.dropna(inplace=True); n1 = len(df)
            if n1 < n0:
                logging.warning(f"    Dropped {n0-n1} NaN rows")
            logging.info(f"    → shape={df.shape}")
            dfs.append(df)
        logging.info("Exiting load_df_list with local data")
        return dfs

    if not (tickers and start and end):
        raise ValueError("No CSVs found (or reload=True) and missing tickers/start/end")

    dfs = []
    for t in tickers:
        logging.info(f"Downloading '{t}' from {start} to {end}")
        df = yf.download(t, start=start, end=end, progress=False)
        if df.empty:
            logging.warning(f"  No data for {t}; skipping")
            continue
        fp = os.path.join(data_dir, f"{t}.csv")
        df.to_csv(fp)
        logging.info(f"  Saved raw CSV to {fp}")
        df = pd.read_csv(fp, parse_dates=[0], index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.dropna(inplace=True)
        logging.info(f"    → shape={df.shape}")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No dataframes loaded or downloaded")
    logging.info("Exiting load_df_list with downloaded data")
    return dfs


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("  Entering compute_all_features")
    df = df.copy()

    # 1) MA5 & Trend (Eqs 10–12)
    df['MA5']  = df['Close'].rolling(5).mean()
    df['ΔMA5'] = df['MA5'].diff()
    df['Trend'] = np.where(df['ΔMA5']>0, 1,
                   np.where(df['ΔMA5']<0, -1, 0))

    # 2) CRP (Eq 13)
    df['CRP'] = 0
    df.loc[(df['Trend']==1)  & (df['Close'] < df['Close'].shift(1)), 'CRP'] =  1
    df.loc[(df['Trend']==-1) & (df['Close'] > df['Close'].shift(1)), 'CRP'] = -1

    # 3) RP (Eq 14) – map +1→1 (URP), -1→2 (DRP)
    df['RP'] = 0
    for i in range(len(df)-10):
        c = df['CRP'].iat[i]
        if c != 0:
            window = df['MA5'].iloc[i+1:i+11]
            if (window.max() - window.min()) / df['MA5'].iat[i] > 0.03:
                df.iat[i, df.columns.get_loc('RP')] = 1 if c==1 else 2

    # 4) Candlestick indicators (Eqs 15–19)
    df['Candle']     = np.sign(df['Close']-df['Open'])
    df['Body']       = (df['Close']-df['Open']).abs()/df['Close'].shift(1)
    df['topTail']    = (df['High']-df[['Open','Close']].max(axis=1))/df['Close'].shift(1)
    df['bottomTail'] = (df[['Open','Close']].min(axis=1)-df['Low'])/df['Close'].shift(1)
    df['Whole']      = (df['High']-df['Low'])/df['Close'].shift(1)

    # 5) Volume & pctMV20, VR20, PL20
    df['pctMV20'] = df['Volume']/df['Volume'].rolling(20).mean().shift(1)
    upv   = df['Volume'].where(df['Close']>df['Close'].shift(1),   0)
    downv = df['Volume'].where(df['Close']<df['Close'].shift(1),   0)
    eqv   = df['Volume'].where(df['Close']==df['Close'].shift(1),  0)
    df['VR20'] = (upv.rolling(20).sum() + eqv.rolling(20).sum()/2) \
               / (downv.rolling(20).sum() + eqv.rolling(20).sum()/2) * 100
    df['PL20'] = df['Candle'].rolling(20).apply(lambda x: (x>0).sum()/20*100)

    # 6) CCI14 & CCIS14
    tp      = (df['High'] + df['Low'] + df['Close'])/3
    df['CCI14']  = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std())
    df['CCIS14'] = df['CCI14'].rolling(14).mean()

    # 7) RSI20
    d  = df['Close'].diff()
    up = d.where(d>0, 0); dn = -d.where(d<0, 0)
    df['RSI20'] = 100 - 100/(1 + up.rolling(20).mean()/dn.rolling(20).mean())

    # 8) Stochastic (n=5)
    low5  = df['Low'].rolling(5).min(); high5 = df['High'].rolling(5).max()
    df['StoK5'] = (df['Close'] - low5)/(high5 - low5)*100
    df['StoD5'] = df['StoK5'].rolling(3).mean()
    df['StoR5'] = (high5 - df['Close'])/(high5 - low5)*100

    # 9) MACDR, ROCMA5, ROC5
    ema5  = df['Close'].ewm(span=5,  adjust=False).mean()
    ema10 = df['Close'].ewm(span=10, adjust=False).mean()
    df['MACDR']  = ema5 - ema10
    df['ROCMA5'] = df['MA5'].pct_change(5)*100
    df['ROC5']   = df['Close'].pct_change(5)*100

    # 10) A/B Ratios (n=26)
    n26 = 26
    A   = (df['High'] - df['Open']).rolling(n26).sum()
    B   = (df['Open'] - df['Low']).rolling(n26).sum()
    C   = (df['High'] - df['Close']).rolling(n26).sum()
    D   = (df['Close'] - df['Low']).rolling(n26).sum()
    df['ARatio26']  = A/B*100
    df['BRatio26']  = C/D*100
    df['ABRatio26'] = df['ARatio26']/df['BRatio26']

    # 11) pctChange
    df['pctChange'] = df['Close'].pct_change()*100

    df.dropna(inplace=True)
    logging.info(f"    → compute_all_features output shape={df.shape}")
    return df


def create_sliding_window_data(
    df_list    : list[pd.DataFrame],
    window_size: int,
    target_col : str
) -> tuple[np.ndarray, np.ndarray]:
    logging.info("Entering create_sliding_window_data")
    X, y = [], []
    for idx, df in enumerate(df_list):
        logging.info(f"  DF#{idx} shape={df.shape}")
        feats = df.drop(columns=[target_col]).values
        tgts  = df[target_col].values
        for i in range(len(df) - window_size):
            X.append(feats[i:i+window_size])
            y.append(tgts[i+window_size])
    X_arr = np.array(X); y_arr = np.array(y)
    logging.info(f"  → windows: X.shape={X_arr.shape}, y.shape={y_arr.shape}")
    return X_arr, y_arr


FEATURE_SETS = {
    'S0':  ['CRP','Trend','Candle','pctMV20','VR20','PL20','CCI14','CCIS14',
            'RSI20','StoK5','StoD5','StoR5','MACDR','ROCMA5','ROC5','pctChange'],
    'S1':  ['CRP','Trend','Candle','pctMV20','VR20','PL20','pctChange'],
    'S2':  ['CRP','Trend','Candle','CCI14','CCIS14','RSI20','pctChange'],
    'S3':  ['CRP','Trend','Candle','StoK5','StoD5','StoR5','pctChange'],
    'S4':  ['CRP','Trend','Candle','MACDR','ROCMA5','ROC5','pctChange'],
    'S5':  ['CRP','Trend','Candle','ARatio26','BRatio26','ABRatio26','pctChange'],
    'S6':  ['CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S7':  ['CRP','Trend','Candle','pctMV20','VR20','PL20','Body','topTail',
            'bottomTail','Whole'],
    'S8':  ['CRP','Trend','Candle','CCI14','CCIS14','RSI20','Body','topTail',
            'bottomTail','Whole'],
    'S9':  ['CRP','Trend','Candle','StoK5','StoD5','StoR5','Body','topTail',
            'bottomTail','Whole'],
    'S10':['CRP','Trend','Candle','MACDR','ROCMA5','ROC5','Body','topTail',
            'bottomTail','Whole'],
    'S11':['CRP','Trend','Candle','ARatio26','BRatio26','ABRatio26','Body','topTail',
            'bottomTail','Whole'],
    'S12':['pctMV20','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S13':['VR20','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S14':['PL20','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S15':['CCI14','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S16':['CCIS14','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S17':['RSI20','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S18':['StoK5','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S19':['StoD5','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S20':['StoR5','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S21':['MACDR','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S22':['ROCMA5','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S23':['ROC5','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S24':['ARatio26','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S25':['BRatio26','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S26':['ABRatio26','CRP','Trend','Candle','Body','topTail','bottomTail','Whole'],
    'S27': ['CRP','Trend','Candle','pctMV20','VR20','PL20','CCI14','CCIS14','RSI20',
            'StoK5','StoD5','StoR5','MACDR','ROCMA5','ROC5','pctChange','ARatio26',
            'BRatio26','ABRatio26','Body','topTail','bottomTail','Whole'],
}
