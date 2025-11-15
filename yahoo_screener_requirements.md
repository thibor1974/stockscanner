# Yahoo Finance Screener Requirements and Examples

## Requirements

### Python Version

-   Python 3.8 or newer

### Install Required Packages

``` bash
pip install yahooquery pandas tqdm
```

## Script Features

The extended screener script allows filtering by:

-   Country (ISO code)
-   Market Capitalization
-   Sector
-   Industry
-   Index membership (S&P 500, NASDAQ 100, DAX, CAC40, etc.)
-   Average volume
-   Price range
-   P/E ratio range

## Example Configurations

### Example 1: French Tech Stocks ≥ €1B Market Cap

``` python
CONFIG = {
    "country": "FR",
    "sector": "Technology",
    "min_market_cap": 1e9,
}
```

### Example 2: US Mid‑Cap Tech Stocks in the NASDAQ100

``` python
CONFIG = {
    "country": "US",
    "index_filter": "NASDAQ100",
    "min_market_cap": 2e9,
    "max_market_cap": 1e10,
    "sector": "Technology",
}
```

### Example 3: Belgian Stocks with High Liquidity

``` python
CONFIG = {
    "country": "BE",
    "min_volume": 100000,
}
```

## How to Run

1.  Save the extended script as `yahoo_screener.py`
2.  Adjust the `CONFIG` section at the top of the script
3.  Run:

``` bash
python yahoo_screener.py
```

4.  Output will be saved to:

```{=html}
<!-- -->
```
    filtered_results.csv
