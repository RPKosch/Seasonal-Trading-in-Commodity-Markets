import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ----------------------------
# User‑adjustable parameters
# ----------------------------
TICKER =                    "HO"    # Ticker to analyze
START_YEAR, START_MONTH =   2001, 1    # Analysis start Date (inclusive)
END_YEAR, END_MONTH =       2001, 8    # Analysis end Date (inclusive)
START_VALUE =               1000.0  # Starting portfolio value in CHF

def month_range_by_ym(sy, sm, ey, em):
    """Yield (year, month) from (sy,sm) to (ey,em) inclusive."""
    current = datetime(sy, sm, 1)
    end_month = datetime(ey, em, 1)
    while current <= end_month:
        yield current.year, current.month
        current += relativedelta(months=1)

def find_contract_file(data_root: Path, ticker: str, year: int, month: int) -> Path | None:
    month_end = datetime(year, month, 1) + relativedelta(months=1) - timedelta(days=1)
    for offset in (2, 3, 4, 5, 6):
        cdt = datetime(year, month, 1) + relativedelta(months=offset)
        fname = f"{ticker}_{cdt.year:04d}-{cdt.month:02d}.csv"
        p = data_root / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["Date"])
        if df["Date"].max() >= month_end:
            return p
    return None

def track_portfolio(ticker: str,
                    sy: int, sm: int,
                    ey: int, em: int,
                    start_value: float = START_VALUE):
    project_root = Path().resolve().parent.parent
    data_root    = project_root / "Complete Data" / f"{ticker}_Historic_Data"
    value        = start_value
    records      = []
    prev_contract = None
    prev_close    = None

    for year, month in month_range_by_ym(sy, sm, ey, em):
        cf = find_contract_file(data_root, ticker, year, month)
        if cf is None:
            continue
        df = pd.read_csv(cf, parse_dates=["Date"]).sort_values("Date")
        start_dt = datetime(year, month, 1)
        end_dt   = start_dt + relativedelta(months=1) - pd.Timedelta(days=1)
        mdf = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
        if mdf.empty:
            continue

        contract_name = cf.name
        first_day     = True

        for _, row in mdf.iterrows():
            d     = row["Date"]
            close = row["close"]
            if first_day:
                if contract_name == prev_contract and prev_close is not None:
                    ratio = close / prev_close
                else:
                    ratio = close / row["open"]
                first_day = False
            else:
                ratio = close / prev_close

            value *= ratio
            records.append((d, value))
            prev_close    = close
            prev_contract = contract_name

    perf = pd.DataFrame(records, columns=["Date","Value"]) \
             .drop_duplicates("Date") \
             .set_index("Date") \
             .sort_index()
    return perf

# === RUNNING ===
perf = track_portfolio(TICKER,
                       START_YEAR, START_MONTH,
                       END_YEAR, END_MONTH)

# Print all daily values
pd.set_option('display.max_rows', None)
print("Daily Portfolio Value")
print(perf.to_string())

# Plot performance
plt.figure(figsize=(10,4))
plt.plot(perf.index, perf["Value"], marker="o")
plt.title(f"{TICKER} Rollover Portfolio Performance")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (CHF)")
plt.grid(True)
plt.tight_layout()
plt.show()
