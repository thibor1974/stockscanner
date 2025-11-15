"""stage-analysis.py

Lightweight utility to fetch historical stock data and identify the current
Wyckoff-like stage based on price vs moving averages.

The script computes the 50-day and 200-day moving averages and assigns a
stage integer to each row:
 - 0: Undefined / insufficient data
 - 1: Close < 200_MA and 50_MA > 200_MA
 - 2: Close > 200_MA and 50_MA > 200_MA
 - 3: Close > 200_MA and 50_MA < 200_MA
 - 4: Close < 200_MA and 50_MA < 200_MA

Functions are small and documented so they can be imported and used from
other scripts or run interactively.
"""

from typing import Tuple, Dict, Optional

import yfinance as yf
import pandas as pd
import numpy as np


def get_stock_data(ticker: str) -> pd.DataFrame:
    """Download historical data for a ticker using yfinance.

    Parameters
    - ticker: Stock ticker symbol (e.g., 'AAPL').

    Returns
    - pd.DataFrame: Historical OHLCV data indexed by date.
    """
    stock = yf.Ticker(ticker)
    # Fetch 10 years of daily history; adjust `period` as needed.
    df = stock.history(period="10y")
    return df


def calculate_moving_average(df: pd.DataFrame, window: int) -> pd.Series:
    """Return the rolling moving average for the `Close` column.

    Parameters
    - df: DataFrame containing a 'Close' column.
    - window: Window size in periods (days for daily data).

    Returns
    - pd.Series: Rolling mean of `Close`.
    """
    return df['Close'].rolling(window=window).mean()


def identify_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate the DataFrame with moving averages and a `Stage` column.

    Adds the following columns to `df`:
    - '200_MA': 200-period moving average of Close
    - '50_MA': 50-period moving average of Close
    - 'Volume_Avg': 50-period moving average of Volume
    - 'Stage': integer stage as described in the module docstring

    Parameters
    - df: DataFrame with at least 'Close' and 'Volume' columns.

    Returns
    - pd.DataFrame: The same DataFrame with added columns.
    """
    df['200_MA'] = calculate_moving_average(df, 200)
    df['50_MA'] = calculate_moving_average(df, 50)
    df['Volume_Avg'] = df['Volume'].rolling(window=50).mean()

    # Define boolean masks for the stages
    stage_2 = (df['Close'] > df['200_MA']) & (df['50_MA'] > df['200_MA'])
    stage_4 = (df['Close'] < df['200_MA']) & (df['50_MA'] < df['200_MA'])
    stage_1 = (df['Close'] < df['200_MA']) & (df['50_MA'] > df['200_MA'])
    stage_3 = (df['Close'] > df['200_MA']) & (df['50_MA'] < df['200_MA'])

    conditions = [stage_2, stage_4, stage_1, stage_3]
    stages = [2, 4, 1, 3]

    # Default stage 0 indicates undefined / insufficient data
    df['Stage'] = np.select(conditions, stages, default=0)

    return df


def analyze_stock(ticker: str) -> Tuple[Dict[str, Optional[int]], pd.DataFrame]:
    """Analyze a ticker and return current stage info plus annotated DataFrame.

    The returned `stage_info` contains:
    - 'Ticker': ticker symbol
    - 'Current Stage': integer stage at the latest date
    - 'Duration in Current Stage (days)': number of days since last stage change
      (None when stage is 0 / undefined)

    Parameters
    - ticker: Stock ticker symbol.

    Returns
    - stage_info: dict with current stage and duration
    - df: annotated DataFrame as returned by `identify_stage`
    """
    df = get_stock_data(ticker)
    df = identify_stage(df)

    # Current stage is the stage on the most recent row
    current_stage = df['Stage'].iloc[-1]

    if current_stage == 0:
        return {
            'Ticker': ticker,
            'Current Stage': current_stage,
            'Duration in Current Stage (days)': None
        }, df

    # Mark rows where the stage value changed compared to the previous row
    df['Stage_Change'] = df['Stage'].diff().fillna(0) != 0
    stage_change_indices = df[df['Stage_Change']].index

    # If there was no stage change in the historical data, consider the entire
    # available period as the duration (number of rows/days).
    if stage_change_indices.empty:
        stage_duration = len(df)
    else:
        # Last index where a stage change occurred
        last_stage_change_index = stage_change_indices[-1]
        # Duration in calendar days since that change
        stage_duration = (df.index[-1] - last_stage_change_index).days

    stage_info = {
        'Ticker': ticker,
        'Current Stage': current_stage,
        'Duration in Current Stage (days)': stage_duration
    }

    return stage_info, df


if __name__ == '__main__':
    # Simple CLI usage: prompt for ticker and print stage info.
    # Example: run `python stage-analysis.py` and enter `AAPL`.
    ticker = input("Enter stock ticker: ")
    stage_info, stock_data = analyze_stock(ticker)

    print(f"Ticker: {stage_info['Ticker']}")
    print(f"Current Stage: {stage_info['Current Stage']}")
    print(f"Duration in Current Stage (days): {stage_info['Duration in Current Stage (days)']}")
