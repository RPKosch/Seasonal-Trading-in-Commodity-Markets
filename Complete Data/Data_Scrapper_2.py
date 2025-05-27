import os
import asyncio
import pandas as pd
import json
from playwright.sync_api import sync_playwright
from collections import Counter
from collections import defaultdict

# map 1–12 → URL segment for the historical path
MONTH_DIR = {i: str(i) for i in range(1, 10)}
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


import os
import pandas as pd
from collections import defaultdict

def main(
    underlying: str = "CF",
    start_year: int = 1999,
    end_year: int   = 2000,
    months_to_check: list[int] | None = None
):
    if months_to_check is None:
        months_to_check = list(range(1, 13))

    out_dir = f"{underlying}_Historic_Data"
    os.makedirs(out_dir, exist_ok=True)

    # This will collect all the missing months per year
    missing = defaultdict(list)

    for year in range(start_year, end_year + 1):
        for month in months_to_check:
            mdir = MONTH_DIR[month]
            url = (
                f"https://futures.tradingcharts.com/historical/"
                f"{underlying}/{year}/{mdir}/linechart.html"
            )
            try:
                print(f"Fetching {underlying} {year}-{month:02d} …")
                full = fetch_history_sync(url)

                if full.empty:
                    print(f"  ⚠️ No data for {underlying} {year}-{month:02d}")
                    continue

                # select and rename to exactly Date,open,high,low,close,volume
                df = full[["tradingDay","open","high","low","close","volume"]].copy()
                df.rename(columns={"tradingDay":"Date"}, inplace=True)

                fname = f"{underlying}_{year:04d}-{month:02d}.csv"
                path  = os.path.join(out_dir, fname)
                df.to_csv(path, index=False)
                print(f"  ✅ Saved {path}")

            except Exception as e:
                print(f"  ❌ Error for {underlying} {year}-{month:02d}: {e}")
                missing[year].append(month)

    # After all loops, print the summary of missing months
    if missing:
        print("\nMissing months by year:")
        for yr in sorted(missing):
            # sort months and remove duplicates if any
            months = sorted(set(missing[yr]))
            months_str = ", ".join(str(m) for m in months)
            print(f"    {yr}: [{months_str}]")
    else:
        print("\nAll requested months fetched successfully!")

if __name__ == "__main__":
    main(
        underlying="CF",
        start_year=2014,
        end_year=2025,
        months_to_check=None,
    )

