from playwright.sync_api import sync_playwright
import pandas as pd
import json
from datetime import datetime


def fetch_contract_via_barchart_json(underlying: str, year: int, month: int) -> pd.DataFrame:
    """
    Navigates to the TradingCharts page for a given contract, listens
    for the Barchart JSON XHR, and returns it as a DataFrame.

    underlying: e.g. "CC"
    year:       four-digit year, e.g. 2025
    month:      numeric month 1..12
    """
    target_symbol = f"{underlying}{['', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'][month]}{str(year)[2:]}"
    # e.g. CC → month=7 → 'N' → 'CCN25'

    data_payload = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Listen to all JSON responses
        def on_response(resp):
            nonlocal data_payload
            url = resp.url
            if "getHistory.json" not in url:
                return
            try:
                body = resp.text()
                js = json.loads(body)
                # look for results matching our symbol
                results = js.get("results", [])
                if results and results[0].get("symbol") == target_symbol:
                    data_payload = results
            except:
                pass

        page.on("response", on_response)

        # Go load the page (lets the JS fire the XHR)
        page.goto(
            f"https://futures.tradingcharts.com/historical/{underlying}/{year}/{month}/linewchart.html",
            wait_until="networkidle"
        )
        browser.close()

    if not data_payload:
        raise RuntimeError(f"Did not catch getHistory.json for symbol {target_symbol}")

    # Build DataFrame
    df = pd.DataFrame(data_payload)
    # tradingDay → Date
    df["Date"] = pd.to_datetime(df["tradingDay"])
    # keep only OHLC + volume + Date
    return df[["Date", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    # Example: July 2025 Cocoa (CC → month=7 maps to 'N' → CCN25)
    df = fetch_contract_via_barchart_json("CC", 2025, 3)
    print("First 5 rows:\n", df.head(), "\n")
    print("Last 5 rows:\n", df.tail())
    # Save if you like
    df.to_csv("CCN25_Jul2025.csv", index=False)