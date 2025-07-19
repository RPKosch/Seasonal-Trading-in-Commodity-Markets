# File: Data Analysis/Compute_Monthly_Returns.py

import math
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import date, datetime, timedelta

# -----------------------------------------------------------------------------
# === CONFIGURATION ===
# -----------------------------------------------------------------------------
# If True, compute log returns and write to "All_Monthly_Log_Return_Data"
# otherwise compute simple returns and write to "All_Monthly_Return_Data"
USE_LOG_RETURNS = True

TICKERS         = ["CC", "CF", "CO", "CP", "CT", "ZW", "GD", "HE", "HO",
                   "LE", "NG", "PA", "PL", "ZS", "SU", "SV", "ZC"]
START_YEAR      = 2001
START_MONTH     = 1
END_YEAR        = 2025
END_MONTH       = 4
MIN_DAYS_AFTER  = 3   # require contract to trade at least this many days past month end

# -----------------------------------------------------------------------------
def month_iterator(start_year, start_month, end_year, end_month):
    """Yield (year, month) descending from end back to start."""
    current = date(end_year, end_month, 1)
    last    = date(start_year, start_month, 1)
    while current >= last:
        yield current.year, current.month
        current -= relativedelta(months=1)

def find_contract_file(data_root: Path, ticker: str, year: int, month: int) -> Path | None:
    """Pick first nearby contract that covers the month and trades MIN_DAYS_AFTER days past."""
    month_start = date(year, month, 1)
    month_end   = month_start + relativedelta(months=1) - timedelta(days=1)
    for offset in (2, 3, 4, 5, 6):
        cdt   = month_start + relativedelta(months=offset)
        fname = f"{ticker}_{cdt.year:04d}-{cdt.month:02d}.csv"
        p     = data_root / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["Date"])
        dates = df["Date"].dt.date
        if dates.min() > month_start:
            continue
        if dates.max() < (month_end + timedelta(days=MIN_DAYS_AFTER)):
            continue
        return p
    return None

def compute_return_for_month(data_root: Path, ticker: str, year: int, month: int) -> dict | None:
    """Compute either simple or log return for the chosen contract over the month."""
    contract_file = find_contract_file(data_root, ticker, year, month)
    if contract_file is None:
        return None

    df = pd.read_csv(contract_file, parse_dates=["Date"])
    start_dt = pd.Timestamp(year, month, 1)
    end_dt   = start_dt + relativedelta(months=1) - pd.Timedelta(days=1)
    mdf = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    if mdf.empty:
        return None

    first_open = mdf.iloc[0]["open"]
    last_close = mdf.iloc[-1]["close"]

    if USE_LOG_RETURNS:
        ret = math.log(last_close / first_open)
    else:
        ret = last_close / first_open - 1

    return {
        "ticker":      ticker,
        "year":        year,
        "month":       month,
        "return":      ret,
        "contract":    contract_file.stem,
        "start_date":  mdf.iloc[0]["Date"].strftime("%Y-%m-%d"),
        "start_value": first_open,
        "end_date":    mdf.iloc[-1]["Date"].strftime("%Y-%m-%d"),
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
        hist_root = data_root / f"{ticker}_Historic_Data"
        records   = []

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
