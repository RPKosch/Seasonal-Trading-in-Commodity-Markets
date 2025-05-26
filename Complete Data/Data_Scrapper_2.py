import os
import asyncio
import pandas as pd
from playwright.async_api import async_playwright

# map 1–12 → URL segment for the historical path
MONTH_DIR = {i: str(i) for i in range(1, 10)}
MONTH_DIR.update({10: "A", 11: "B", 12: "C"})

async def fetch_history_via_playwright(url: str) -> pd.DataFrame:
    """
    Navigates to `url`, intercepts the POST to getHistory.json,
    and returns a DataFrame of the results (with a parsed timestamp).
    """
    all_data = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()

        # Capture the historical JSON response
        async def on_response(response):
            if "getHistory.json" in response.url and response.request.method == "POST":
                payload = await response.json()
                results = payload.get("results", [])
                if results:
                    all_data.extend(results)

        page.on("response", on_response)

        await page.goto(url)
        # wait until no network activity to be sure the POST fired
        await page.wait_for_load_state("networkidle")
        await browser.close()

    if not all_data:
        raise RuntimeError(f"No data intercepted for {url}")

    df = pd.DataFrame(all_data)
    # Let pandas infer format (handles ISO strings and/or ms)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

async def main(
    underlying: str = "CF",
    start_year: int = 1999,
    end_year: int   = 2025,
    months_to_check: list[int] | None = None
):
    if months_to_check is None:
        months_to_check = list(range(1, 13))

    out_dir = f"{underlying}_Historic_Data"
    os.makedirs(out_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        for month in months_to_check:
            mdir = MONTH_DIR[month]
            url = (
                f"https://futures.tradingcharts.com/historical/"
                f"{underlying}/{year}/{mdir}/linechart.html"
            )
            try:
                print(f"Fetching {underlying} {year}-{month:02d} …")
                full = await fetch_history_via_playwright(url)
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


if __name__ == "__main__":
    # to run: pip install playwright pandas
    # then: playwright install
    asyncio.run(main())
