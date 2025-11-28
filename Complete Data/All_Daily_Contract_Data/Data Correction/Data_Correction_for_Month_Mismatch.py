# File: Complete Data/Data_Correction_Month_OffByOne.py

import pandas as pd
from pathlib import Path
from collections import defaultdict

# === CONFIGURATION ===
TICKERS     = ["CC", "CF", "CO", "CP", "CT", "ZW", "GD", "HE", "HO", "LE",
               "NG", "PA", "PL", "ZS", "SU", "SV", "ZC"]

def find_offbyone_months(base_dir: Path):
    """
    Walk each ticker’s Historic_Data folder, check each CSV’s last-line date
    against its filename’s year-month. Accept if
      actual_month == filename_month or actual_month == (filename_month-1),
    skipping any files for July 2025 onward.
    Returns: { ticker: { year: [bad_month1, ...], ... }, ... }
    """
    mismatches = {}

    for ticker in TICKERS:
        hist_dir = base_dir / f"{ticker}_Historic_Data"
        if not hist_dir.exists():
            continue

        per_year = defaultdict(list)

        for csv_path in hist_dir.glob(f"{ticker}_????-??.csv"):
            # parse expected year and month
            _, ym = csv_path.stem.split("_")
            exp_year, exp_month = map(int, ym.split("-"))

            # skip July 2025 and beyond
            if exp_year == 2025 and exp_month >= 7:
                continue

            # read CSV
            try:
                df = pd.read_csv(csv_path, parse_dates=["Date"])
            except Exception:
                continue

            if df.empty:
                per_year[exp_year].append(exp_month)
                continue

            last_date = df["Date"].iloc[-1]
            act_year, act_month = last_date.year, last_date.month

            # compute previous month (with wrap)
            if exp_month == 1:
                prev_month, prev_year = 12, exp_year - 1
            else:
                prev_month, prev_year = exp_month - 1, exp_year

            # flag if not matching either exp_month or prev_month
            if not (
                (act_year, act_month) == (exp_year, exp_month)
                or (act_year, act_month) == (prev_year, prev_month)
            ):
                per_year[exp_year].append(exp_month)

        if per_year:
            for yr in per_year:
                per_year[yr] = sorted(set(per_year[yr]))
            mismatches[ticker] = dict(per_year)

    return mismatches

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    base_dir     = project_root / "Complete Data"
    bad = find_offbyone_months(base_dir)

    if not bad:
        print("All files have last-date matching filename-month or the prior month (excl. post-Jul 2025).")
    else:
        for ticker, years in bad.items():
            print(f"\nTicker {ticker}:")
            print("    Off‑by‑one mismatches by year:")
            for yr in sorted(years):
                months = years[yr]
                months_str = ", ".join(str(m) for m in months)
                print(f"        {yr}: [{months_str}],")
