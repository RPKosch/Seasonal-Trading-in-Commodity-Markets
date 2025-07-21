# File: Data Analysis/Compute_Monthly_Returns.py

import math
import re
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import date, datetime, timedelta

# -----------------------------------------------------------------------------
# === CONFIGURATION ===
# -----------------------------------------------------------------------------
USE_LOG_RETURNS = True

TICKERS         = ["CC", "CF", "CO", "CP", "CT", "ZW", "GD", "HE", "HO",
                   "LE", "NG", "PA", "PL", "ZS", "SU", "SV", "ZC"]
START_YEAR      = 2001
START_MONTH     = 1
END_YEAR        = 2025
END_MONTH       = 4

# -----------------------------------------------------------------------------
def month_iterator(start_year, start_month, end_year, end_month):
    """Yield (year, month) descending from end back to start."""
    current = date(end_year, end_month, 1)
    last    = date(start_year, start_month, 1)
    while current >= last:
        yield current.year, current.month
        current -= relativedelta(months=1)

def find_contract(tkr: str, y: int, m: int, root: Path):
    """
    Scan TICKER_Historic_Data directory for the front‐month contract:
    pick the file with smallest (year*12+month) lag ≥ 2 that still
    trades through month-end+15 days and has at least one trade in-month.
    Returns (ticker, filtered_dataframe) or (None, None).
    """
    m0   = datetime(y, m, 1)
    mend = m0 + relativedelta(months=1) - timedelta(days=1)
    pat  = re.compile(rf"^{tkr}[_-](\d{{4}})-(\d{{2}})\.csv$")
    cands = []
    folder = root / f"{tkr}_Historic_Data"
    for p in folder.iterdir():
        mm = pat.match(p.name)
        if not mm:
            continue
        fy, fm = map(int, mm.groups())
        lag = (fy - y) * 12 + (fm - m)
        if lag < 2:
            continue
        df = pd.read_csv(p, parse_dates=["Date"])
        if df.Date.max() < mend + timedelta(days=15):
            continue
        mdf = df[(df.Date >= m0) & (df.Date <= mend)]
        if mdf.empty:
            continue
        cands.append((lag, mdf.sort_values("Date")))
    if not cands:
        return None, None
    # pick the closest front‐month
    _, best = min(cands, key=lambda x: x[0])
    return tkr, best

def compute_return_for_month(data_root: Path, ticker: str, year: int, month: int):
    """Compute simple or log return over the calendar month from front‐month contract."""
    tkr, mdf = find_contract(ticker, year, month, data_root)
    if mdf is None:
        return None

    start_dt = pd.Timestamp(year, month, 1)
    end_dt   = start_dt + relativedelta(months=1) - pd.Timedelta(days=1)
    in_month = mdf[(mdf["Date"] >= start_dt) & (mdf["Date"] <= end_dt)]
    if in_month.empty:
        return None

    first_open = in_month.iloc[0]["open"]
    last_close = in_month.iloc[-1]["close"]

    if USE_LOG_RETURNS:
        ret = math.log(last_close / first_open)
    else:
        ret = last_close / first_open - 1

    return {
        "ticker":      ticker,
        "year":        year,
        "month":       month,
        "return":      ret,
        "contract":    f"{ticker}_{year:04d}-{month:02d}",
        "start_date":  in_month.iloc[0]["Date"].strftime("%Y-%m-%d"),
        "start_value": first_open,
        "end_date":    in_month.iloc[-1]["Date"].strftime("%Y-%m-%d"),
        "end_value":   last_close,
    }

def main():
    project_root = Path().resolve().parent.parent
    data_root    = project_root / "Complete Data"
    folder_name  = "All_Monthly_Log_Return_Data" if USE_LOG_RETURNS else "All_Monthly_Return_Data"
    output_dir   = data_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n=== Processing {ticker} ===")
        records = []
        hist_root = data_root  # find_contract uses data_root / f"{ticker}_Historic_Data"

        for yr, mo in month_iterator(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
            res = compute_return_for_month(hist_root, ticker, yr, mo)
            if res:
                records.append(res)
            else:
                print(f"  {yr}-{mo:02d}: skipped")

        if records:
            out_df = pd.DataFrame(records).sort_values(["ticker","year","month"])
            out_path = output_dir / f"{ticker}_Monthly_Revenues.csv"
            out_df.to_csv(out_path, index=False)
            typ = "log returns" if USE_LOG_RETURNS else "simple returns"
            print(f"  ▶ Saved {len(out_df)} rows of {typ} to {out_path}")
        else:
            print(f"  ⚠️ No data for {ticker}, nothing saved.")

if __name__ == "__main__":
    main()
