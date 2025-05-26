import os
import pandas as pd
from playwright.sync_api import sync_playwright

# Directory segment in the URL (1–9 stay numeric, 10→A, 11→B, 12→C)
MONTH_DIR   = {i: str(i) for i in range(1, 10)}
MONTH_DIR.update({10: "A", 11: "B", 12: "C"})

def fetch_history_sync(url: str) -> pd.DataFrame:
    """
    Navigate to a linechart page, intercept the getHistory.json POST,
    and return a DataFrame of the results (with parsed timestamp).
    """
    all_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()

        def on_response(resp):
            if "getHistory.json" in resp.url and resp.request.method == "POST":
                try:
                    payload = resp.json()
                    results = payload.get("results", [])
                    if results:
                        all_data.extend(results)
                except:
                    pass

        page.on("response", on_response)
        page.goto(url, wait_until="networkidle")
        browser.close()

    if not all_data:
        raise RuntimeError(f"No data intercepted for {url}")

    df = pd.DataFrame(all_data)
    # parse ISO‐8601 timestamps or epoch ms
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def main(
    underlying: str,
    output_folder: str,
    year_month_map: dict[int, list[int]]
):
    """
    underlying:     e.g. "SV"
    output_folder:  where to save the CSVs
    year_month_map: e.g. {2025:[1,2,3], 2020:[5,7,2]}
    """
    os.makedirs(output_folder, exist_ok=True)

    for year, months in sorted(year_month_map.items()):
        for month in sorted(months):
            mdir = MONTH_DIR[month]
            url = (
                f"https://futures.tradingcharts.com/historical/"
                f"{underlying}/{year}/{mdir}/linechart.html"
            )
            try:
                print(f"→ Fetching {underlying} {year}-{month:02d} …")
                full = fetch_history_sync(url)
                if full.empty:
                    print(f"   ⚠️ No data for {underlying} {year}-{month:02d}")
                    continue

                # select & rename to exactly Date, open, high, low, close, volume
                df = full[["tradingDay","open","high","low","close","volume"]].copy()
                df.rename(columns={"tradingDay":"Date"}, inplace=True)

                fname = f"{underlying}_{year:04d}-{month:02d}.csv"
                path  = os.path.join(output_folder, fname)
                df.to_csv(path, index=False)
                print(f"   ✅ Saved {path}")

            except Exception as e:
                print(f"   ❌ Error for {underlying} {year}-{month:02d}: {e}")

if __name__ == "__main__":
    # === configure here ===
    UNDERLYING     = "SV"
    BASE_DIR       = r"C:\Users\ralph\PycharmProjects\Seasonal-Trading-in-Commodity-Markets\Complete Data"
    OUTPUT_FOLDER  = os.path.join(BASE_DIR, f"{UNDERLYING}_Historic_Data")
    YEAR_MONTH_MAP = {
        2015: [3],
    }
    # ======================

    main(UNDERLYING, OUTPUT_FOLDER, YEAR_MONTH_MAP)
