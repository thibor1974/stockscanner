"""oversold_screener.py

Simple stock oversold screener using RSI, Bollinger Bands and Stochastic.

Criteria (by default): consider a ticker oversold when it meets at least two
of the following:
 - RSI(14) < 30
 - Stochastic %K(14) < 20
 - Close < lower Bollinger Band (20, 2)

Usage examples:
    python oversold_screener.py --tickers AAPL MSFT TSLA
    python oversold_screener.py --tickers-file tickers.txt --save results.csv

The script accepts a list of tickers or a file with tickers (one per line).
"""

from typing import List, Tuple
import argparse
import sys

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_history(ticker: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """Fetch historical OHLCV data for `ticker`.

    Returns a DataFrame indexed by date.
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Uses exponential moving average smoothing for gains/losses.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (middle_ma, upper_band, lower_band)"""
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower


def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Return %K and %D stochastic oscillator series."""
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = (df['Close'] - low_min) / (high_max - low_min) * 100
    d = k.rolling(window=d_period).mean()
    return k, d


def is_oversold(df: pd.DataFrame, rsi_thresh: float = 30.0, stoch_thresh: float = 20.0) -> Tuple[bool, dict]:
    """Decide if the latest row of `df` is oversold.

    Returns (result, details) where details has the last RSI, %K and Bollinger info.
    """
    close = df['Close']
    rsi = compute_rsi(close)
    ma, upper, lower = bollinger_bands(close)
    k, d = stochastic_oscillator(df)

    last = df.index[-1]
    rsi_last = float(rsi.loc[last]) if not pd.isna(rsi.loc[last]) else np.nan
    k_last = float(k.loc[last]) if not pd.isna(k.loc[last]) else np.nan
    lower_last = float(lower.loc[last]) if not pd.isna(lower.loc[last]) else np.nan
    close_last = float(close.loc[last])

    checks = {
        'rsi_below_thresh': rsi_last < rsi_thresh if not np.isnan(rsi_last) else False,
        'stochastic_below_thresh': k_last < stoch_thresh if not np.isnan(k_last) else False,
        'below_lower_bband': close_last < lower_last if not np.isnan(lower_last) else False,
    }

    # Consider oversold when at least two of the three indicators agree.
    score = sum(1 for v in checks.values() if v)
    result = score >= 2

    details = {
        'last_close': close_last,
        'rsi': rsi_last,
        'stochastic_k': k_last,
        'lower_bband': lower_last,
        'checks': checks,
        'score': score,
    }
    return result, details


def screen_tickers(tickers: List[str], period: str = '1y') -> pd.DataFrame:
    """Screen a list of tickers and return a DataFrame of oversold candidates."""
    rows = []
    for t in tickers:
        try:
            df = fetch_history(t, period=period)
            if df.empty or len(df) < 30:
                rows.append({'Ticker': t, 'Oversold': False, 'Notes': 'Insufficient data'})
                continue

            oversold, details = is_oversold(df)
            notes = ''
            if oversold:
                notes = 'Meets oversold criteria'
            else:
                notes = 'Not oversold'

            rows.append({'Ticker': t, 'Oversold': oversold, 'Notes': notes, 'Details': details})
        except Exception as e:
            rows.append({'Ticker': t, 'Oversold': False, 'Notes': f'Error: {e}'})

    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description='Simple oversold stock screener')
    p.add_argument('--tickers', nargs='*', help='List of tickers to screen')
    p.add_argument('--tickers-file', help='Path to a file with tickers (one per line)')
    p.add_argument('--period', default='1y', help='History period for yfinance (default: 1y)')
    p.add_argument('--save', help='Save results to CSV file')
    return p.parse_args()


def main():
    args = parse_args()
    tickers = []
    if args.tickers:
        tickers = args.tickers
    elif args.tickers_file:
        try:
            with open(args.tickers_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f'Failed to read tickers file: {e}', file=sys.stderr)
            sys.exit(1)
    else:
        # Fallback sample list; replace with your universe
        tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD']

    results = screen_tickers(tickers, period=args.period)

    # Print a concise summary
    print(results[['Ticker', 'Oversold', 'Notes']].to_string(index=False))

    if args.save:
        results.to_csv(args.save, index=False)
        print(f'Saved results to {args.save}')


if __name__ == '__main__':
    main()
