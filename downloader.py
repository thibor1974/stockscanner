from yahooquery import Ticker, search
import pandas as pd
from tqdm import tqdm
import time

# -------------------------
# CONFIGURATION BLOCK
# -------------------------

CONFIG = {
    "country": "US",         # e.g., US, FR, DE, BE, CA, JP, GB
    "min_market_cap": 1e9,   # minimum market cap ($1B)
    "max_market_cap": None,  # or a number, e.g. 5e10
    "sector": None,          # e.g. "Technology", "Financial Services"
    "industry": None,        # e.g. "Software—Infrastructure"
    "index_filter": None,    # examples: "SP500", "NASDAQ100", "DOW"
    "min_volume": 200000,    # average daily volume
    "min_price": None,
    "max_price": None,
    "min_pe": None,
    "max_pe": None,
}

# -------------------------
# INDEX DEFINITIONS
# -------------------------

INDEXES = {
    "SP500": "^GSPC",
    "NASDAQ100": "^NDX",
    "DOW": "^DJI",
    "CAC40": "^FCHI",
    "DAX": "^GDAXI",
    "BEL20": "^BFX",
    "FTSE100": "^FTSE",
}


# -------------------------
# FUNCTIONS
# -------------------------

def get_index_components(index_symbol):
    """Fetch all components of a major index."""
    try:
        t = Ticker(index_symbol)
        comps = t.components
        if index_symbol in comps:
            return comps[index_symbol]["symbol"]
        return []
    except:
        return []


def get_tickers_by_country(country_code):
    """Retrieve all tickers for a given country."""
    tickers = []
    seen = set()
    queries = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    for q in tqdm(queries, desc="Scanning Yahoo Finance"):
        result = search(q)
        if not result or "quotes" not in result:
            continue

        for item in result["quotes"]:
            if (
                item.get("quoteType") == "EQUITY"
                and item.get("country") == country_code
            ):
                sym = item.get("symbol")
                if sym and sym not in seen:
                    seen.add(sym)
                    tickers.append(sym)

        time.sleep(0.2)

    return tickers


def fetch_fundamentals(tickers):
    """Fetch fundamental metrics for a group of tickers."""
    rows = []

    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        tq = Ticker(batch)

        summary = tq.summary_detail
        price = tq.price
        key_stats = tq.key_stats

        for sym in batch:
            s = summary.get(sym, {})
            p = price.get(sym, {})
            k = key_stats.get(sym, {})

            rows.append({
                "symbol": sym,
                "marketCap": s.get("marketCap"),
                "sector": p.get("sector"),
                "industry": p.get("industry"),
                "avgVolume": s.get("averageVolume"),
                "price": p.get("regularMarketPrice"),
                "pe": k.get("trailingPE"),
            })

        time.sleep(0.3)

    return pd.DataFrame(rows)


def apply_filters(df):
    """Filter tickers using the CONFIG rules."""
    cfg = CONFIG

    # Ensure 'marketCap' is numeric and handle missing values
    if "marketCap" in df.columns:
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
        df = df.dropna(subset=["marketCap"])

    if cfg["min_market_cap"]:
        df = df[df["marketCap"] >= cfg["min_market_cap"]]

    if cfg["max_market_cap"]:
        df = df[df["marketCap"] <= cfg["max_market_cap"]]

    if cfg["sector"]:
        df = df[df["sector"] == cfg["sector"]]

    if cfg["industry"]:
        df = df[df["industry"] == cfg["industry"]]

    if cfg["min_volume"]:
        df = df[df["avgVolume"] >= cfg["min_volume"]]

    if cfg["min_price"]:
        df = df[df["price"] >= cfg["min_price"]]

    if cfg["max_price"]:
        df = df[df["price"] <= cfg["max_price"]]

    if cfg["min_pe"]:
        df = df[df["pe"] >= cfg["min_pe"]]

    if cfg["max_pe"]:
        df = df[df["pe"] <= cfg["max_pe"]]

    # Index membership filter
    if cfg["index_filter"]:
        index_sym = INDEXES.get(cfg["index_filter"])
        index_members = get_index_components(index_sym)
        df = df[df["symbol"].isin(index_members)]

    return df.sort_values("marketCap", ascending=False)


# -------------------------
# MAIN SCRIPT
# -------------------------

if __name__ == "__main__":
    print(f"Fetching tickers for {CONFIG['country']}")

    tickers = get_tickers_by_country(CONFIG["country"])

    print("Fetching fundamentals...")
    df = fetch_fundamentals(tickers)

    print("Applying filters...")
    filtered = apply_filters(df)

    print(filtered)

    filtered.to_csv("filtered_results.csv", index=False)
    print("Saved to filtered_results.csv")