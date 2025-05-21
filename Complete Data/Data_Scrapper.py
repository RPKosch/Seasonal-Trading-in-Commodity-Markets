import os
from playwright.sync_api import sync_playwright
import pandas as pd
import json
from datetime import datetime


MONTH_CODES = ['', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']

def fetch_contract_via_barchart_json(underlying: str, year: int, month: int) -> pd.DataFrame:
    """
    Fetch OHLCV history for a given futures contract from Barchart JSON.

    underlying: e.g. "CC"
    year:       four-digit year, e.g. 2025
    month:      numeric month 1..12

    Returns a DataFrame with Date, open, high, low, close, volume.
    """
    # Build target symbol, e.g. CC + month code + two-digit year
    month_code = MONTH_CODES[month]
    target_symbol = f"{underlying}{month_code}{str(year)[2:]}"
    data_payload = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.set_default_navigation_timeout(45000)
        page.set_default_timeout(45000)

        def on_response(resp):
            nonlocal data_payload
            if "getHistory.json" not in resp.url:
                return
            try:
                js = resp.json()
                results = js.get("results", [])
                if results and results[0].get("symbol") == target_symbol:
                    data_payload = results
            except Exception:
                pass

        MONTH_DIR = {i: str(i) for i in range(1, 10)}
        MONTH_DIR.update({10: "A", 11: "B", 12: "C"})

        # …then, inside your loop or function:
        month_new = MONTH_DIR[month]

        page.on("response", on_response)
        page.goto(
            f"https://futures.tradingcharts.com/historical/{underlying}/{year}/{month_new}/linewchart.html",
            wait_until="networkidle"
        )
        browser.close()

    if not data_payload:
        raise RuntimeError(f"Did not catch getHistory.json for symbol {target_symbol}")

    df = pd.DataFrame(data_payload)
    df["Date"] = pd.to_datetime(df["tradingDay"])
    return df[["Date", "open", "high", "low", "close", "volume"]]


def main(
        underlying: str = "PA",
        start_year: int = 1999,
        end_year: int = 2025,
        months_to_check: list[int] | None = None
):
    """
    Fetch and save historical CSVs for `underlying` from start_year to end_year.

    months_to_check: optional list of month numbers (1–12).
                     If None, all months are fetched.
    """
    # If no filter is provided, check all 1–12
    if months_to_check is None:
        months_to_check = list(range(1, 13))

    base_dir = f"{underlying}_Historic_Data"
    os.makedirs(base_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        for month in months_to_check:
            try:
                df = fetch_contract_via_barchart_json(underlying, year, month)
                if df.empty:
                    print(f"No data for {underlying} {year}-{month:02d}")
                    continue

                file_name = f"{underlying}_{year:04d}-{month:02d}.csv"
                file_path = os.path.join(base_dir, file_name)
                df.to_csv(file_path, index=False)
                print(f"Saved {file_path}")

            except Exception as e:
                print(f"Error fetching {underlying} {year}-{month:02d}: {e}")

if __name__ == "__main__":
    #main(months_to_check=[1, 4, 7, 10])
    main()