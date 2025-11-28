import os
from pathlib import Path

import pandas as pd
import json
from playwright.sync_api import sync_playwright
from collections import Counter
from collections import defaultdict

# Directory segment in the URL (1–9 stay numeric, 10→A, 11→B, 12→C)
MONTH_DIR   = {i: str(i) for i in range(1, 10)}
MONTH_DIR.update({10: "A", 11: "B", 12: "C"})

def fetch_history_sync(url: str) -> pd.DataFrame:
    """
    Navigate to a linechart page, intercept all getHistory.json POSTs,
    then return the series for which an exact duplicate exists.
    If none are duplicated, fall back to the longest series.
    """
    all_series = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()

        def on_response(resp):
            if "getHistory.json" in resp.url and resp.request.method == "POST":
                try:
                    payload = resp.json()
                    results = payload.get("results", [])
                    if results:
                        all_series.append(results)
                except:
                    pass

        page.on("response", on_response)
        page.goto(url, wait_until="networkidle")
        browser.close()

    if not all_series:
        raise RuntimeError(f"No data intercepted for {url}")

    # --- fingerprint each series and count ---
    fps = [json.dumps(series, sort_keys=True) for series in all_series]
    counts = Counter(fps)

    # find any fingerprint that occurs >1
    dup_fps = [fp for fp, cnt in counts.items() if cnt > 1]
    if dup_fps:
        # use the first duplicate fingerprint
        chosen_fp = dup_fps[0]
        # find all indices in all_series that match
        dup_indices = [i for i, fp in enumerate(fps, start=1) if fp == chosen_fp]
        #   print(f"Found duplicate series at indices {dup_indices}, using the first one.")
        main_series = all_series[dup_indices[0] - 1]  # -1 because we started at 1
    else:
        #   print("No exact duplicates found; falling back to the longest series.")
        return pd.DataFrame()


    # build the DataFrame to return
    df = pd.DataFrame(main_series)
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

    missing = defaultdict(list)

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
                missing[year].append(month)

    # After all loops, print the summary of missing months
    if missing:
        print("\nMissing months by year:")
        for yr in sorted(missing):
            # sort months and remove duplicates if any
            months = sorted(set(missing[yr]))
            months_str = ", ".join(str(m) for m in months)
            print(f"    {yr}: [{months_str}],")
    else:
        print("\nAll requested months fetched successfully!")

if __name__ == "__main__":
    BASE_DIR = Path().resolve() / "All_Daily_Contract_Data"

    UNDERLYING     = "ZL"
    OUTPUT_FOLDER  = os.path.join(BASE_DIR, f"{UNDERLYING}_Historic_Data")
    YEAR_MONTH_MAP = {
        2020: [3],
    }

    main(UNDERLYING, OUTPUT_FOLDER, YEAR_MONTH_MAP)

    UNDERLYING     = "ZM"
    OUTPUT_FOLDER  = os.path.join(BASE_DIR, f"{UNDERLYING}_Historic_Data")
    YEAR_MONTH_MAP = {
        2003: [3],
        2021: [1],
    }

    main(UNDERLYING, OUTPUT_FOLDER, YEAR_MONTH_MAP)

    UNDERLYING     = "ZO"
    OUTPUT_FOLDER  = os.path.join(BASE_DIR, f"{UNDERLYING}_Historic_Data")
    YEAR_MONTH_MAP = {
        2005: [2],
    }

    main(UNDERLYING, OUTPUT_FOLDER, YEAR_MONTH_MAP)
