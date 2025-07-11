# File: Data Analysis/Compute_Monthly_Returns.py

import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import date

# === CONFIGURATION ===
TICKERS     = ["CC", "CF", "CO", "CP", "CT", "CW", "GD", "HE", "HO", "LE", "NG", "PA", "PL", "RB", "SB", "SU", "SV", "ZC"]
START_YEAR, START_MONTH = 1999, 1
END_YEAR,   END_MONTH   = 2025, 4

def month_iterator(start_year, start_month, end_year, end_month):
    """Yields (year, month) from (end_year,end_month) back to (start_year,start_month)."""
    current = date(end_year, end_month, 1)
    last    = date(start_year, start_month, 1)
    while current >= last:
        yield current.year, current.month
        current -= relativedelta(months=1)

def find_contract_file(data_root: Path, ticker: str, year: int, month: int) -> Path | None:
    """
    For a target-month (year,month), try ticker files for month+1, then month+2.
    Returns Path or None.
    """
    for offset in (1, 2):
        cdt = date(year, month, 1) + relativedelta(months=offset)
        fname = f"{ticker}_{cdt.year:04d}-{cdt.month:02d}.csv"
        p = data_root / fname
        if p.exists():
            return p
    return None

def compute_return_for_month(data_root: Path, ticker: str, year: int, month: int) -> dict | None:
    """
    Returns a dict with:
      year, month, return,
      ticker, contract, start_date, start_value,
      end_date, end_value
    or None if cannot compute.
    """
    # try both rollover possibilities
    for offset in (1, 2, 3, 4):
        # build candidate contract filename
        cdt = date(year, month, 1) + relativedelta(months=offset)
        fname = f"{ticker}_{cdt.year:04d}-{cdt.month:02d}.csv"
        contract_file = data_root / fname
        if not contract_file.exists():
            continue

        # load & filter
        df = pd.read_csv(contract_file, parse_dates=["Date"])
        start_dt = pd.Timestamp(year, month, 1)
        end_dt   = start_dt + relativedelta(months=1) - pd.Timedelta(days=1)
        mdf = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
        if mdf.empty:
            # this contract doesn’t actually cover our month—try next rollover
            continue

        # we found a valid contract with data
        mdf = mdf.sort_values("Date")
        first_row = mdf.iloc[0]
        last_row  = mdf.iloc[-1]

        first_open = first_row["open"]
        last_close = last_row["close"]
        ret = last_close / first_open - 1

        return {
            "ticker":      ticker,
            "year":        year,
            "month":       month,
            "return":      ret,
            "contract":    contract_file.stem,
            "start_date":  first_row["Date"].strftime("%Y-%m-%d"),
            "start_value": first_open,
            "end_date":    last_row["Date"].strftime("%Y-%m-%d"),
            "end_value":   last_close,
        }

    # no offset worked
    return None


def main():
    project_root = Path().resolve().parent.parent
    output_dir   = project_root / "Complete Data" / "All_Monthly_Return_Data"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n=== Processing {ticker} ===")
        data_root = project_root / "Complete Data" / f"{ticker}_Historic_Data"
        records = []

        for yr, mo in month_iterator(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
            res = compute_return_for_month(data_root, ticker, yr, mo)
            if res:
                records.append(res)
                print(f"  {yr}-{mo:02d}: OK (ret={res['return']:.4f}, contract={res['contract']})")
            else:
                print(f"  {yr}-{mo:02d}: skipped")

        if records:
            out_df = pd.DataFrame(records)
            out_df = out_df.sort_values(["ticker", "year", "month"])
            out_path = output_dir / f"{ticker}_Monthly_Revenues.csv"
            out_df.to_csv(out_path, index=False)
            print(f"  ▶ Saved {len(out_df)} rows to {out_path}")
        else:
            print(f"  ⚠️ No data for {ticker}, nothing saved.")

if __name__ == "__main__":
    main()
