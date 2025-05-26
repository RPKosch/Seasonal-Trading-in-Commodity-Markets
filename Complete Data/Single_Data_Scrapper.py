import os
from playwright.sync_api import sync_playwright
import pandas as pd
import json

# Month codes for symbol + path mapping
MONTH_CODES = ['', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
# Directory part for URL path (1-9, A, B, C)
MONTH_DIR  = {i: str(i) for i in range(1, 10)}
MONTH_DIR.update({10: "A", 11: "B", 12: "C"})

def fetch_contract_via_barchart_json(underlying: str, year: int, month: int) -> pd.DataFrame:
    month_code   = MONTH_CODES[month]
    target_symbol = f"{underlying}{month_code}{str(year)[2:]}"
    data_payload = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()

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

        page.on("response", on_response)
        page.goto(
            f"https://futures.tradingcharts.com/historical/{underlying}/{year}/{MONTH_DIR[month]}/linewchart.html",
            wait_until="networkidle"
        )
        browser.close()

    if not data_payload:
        raise RuntimeError(f"Did not catch getHistory.json for symbol {target_symbol}")

    df = pd.DataFrame(data_payload)
    df["Date"] = pd.to_datetime(df["tradingDay"])
    return df[["Date", "open", "high", "low", "close", "volume"]]

def main(
    underlying: str,
    output_folder: str,
    year_month_map: dict[int, list[int]]
):
    """
    underlying:      e.g. "CC"
    output_folder:   path where CSVs will be saved
    year_month_map:  {1999: [3,5,12], 2000: [1,2], ...}
    """
    os.makedirs(output_folder, exist_ok=True)

    for year, months in sorted(year_month_map.items()):
        for month in sorted(months):
            try:
                df = fetch_contract_via_barchart_json(underlying, year, month)
                if df.empty:
                    print(f"No data for {underlying} {year}-{month:02d}")
                    continue

                fname = f"{underlying}_{year:04d}-{month:02d}.csv"
                path  = os.path.join(output_folder, fname)
                df.to_csv(path, index=False)
                print(f"Saved {path}")

            except Exception as e:
                print(f"Error fetching {underlying} {year}-{month:02d}: {e}")

if __name__ == "__main__":
    # === configure here ===
    UNDERLYING     = "SI"
    BASE_DIR       = r"C:\Users\ralph\PycharmProjects\Seasonal-Trading-in-Commodity-Markets\Complete Data"
    OUTPUT_FOLDER  = os.path.join(BASE_DIR, f"{UNDERLYING}_Historic_Data")
    YEAR_MONTH_MAP = {
        2025: [1,2,3],
        2020: [5, 7, 2]
    }
    # ========================

    main(UNDERLYING, OUTPUT_FOLDER, YEAR_MONTH_MAP)

    #https://futures.tradingcharts.com/historical/PL/1999/4/linechart.html
    #https://futures.tradingcharts.com/historical/CP/1999/9/linechart.html
