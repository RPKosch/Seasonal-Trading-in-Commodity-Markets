import re
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates

# ----------------------------
# Parameters (month-based)
# ----------------------------
START_YEAR, START_MONTH             = 2001, 1
INITIAL_END_YEAR, INITIAL_END_MONTH = 2010, 12
FINAL_END_YEAR, FINAL_END_MONTH     = 2024, 12

START_VALUE = 1000.0
ENTRY_COST  = 0.0025
EXIT_COST   = 0.0025

# how many years of history to use for RLSSA?
# set to None to use full history, or an integer like 5 for last 5 years
LOOKBACK_YEARS = 10

# Plot window
PLOT_START_YEAR, PLOT_START_MONTH   = 2011, 1
PLOT_END_YEAR,   PLOT_END_MONTH     = 2024, 12

# ----------------------------
# Convert to datetime endpoints
# ----------------------------
start_date  = datetime(START_YEAR, START_MONTH, 1)
initial_end = datetime(INITIAL_END_YEAR, INITIAL_END_MONTH, 1) + pd.offsets.MonthEnd(0)
final_end   = datetime(FINAL_END_YEAR, FINAL_END_MONTH,   1) + pd.offsets.MonthEnd(0)
plot_start  = datetime(PLOT_START_YEAR, PLOT_START_MONTH, 1)
plot_end    = datetime(PLOT_END_YEAR,   PLOT_END_MONTH,   1) + pd.offsets.MonthEnd(0)

# ----------------------------
# Robust low‑rank helper
# ----------------------------
def robust_low_rank(X, q, max_iter=25, eps=1e-7):
    L_, K = X.shape
    U0, s0, V0t = la.svd(X, full_matrices=False)
    U = U0[:, :q] * np.sqrt(s0[:q])
    V = (V0t[:q, :].T) * np.sqrt(s0[:q])
    for _ in range(max_iter):
        R = X - U @ V.T
        W = 1.0 / (np.abs(R) + eps)
        Xw = np.sqrt(W) * X
        Uw, sw, Vwt = la.svd(Xw, full_matrices=False)
        U = Uw[:, :q] * np.sqrt(sw[:q])
        V = (Vwt[:q, :].T) * np.sqrt(sw[:q])
    return U, V

# ----------------------------
# RLSSA‐score function
# ----------------------------
def rlssa_score(series: pd.Series, L: int, q: int):
    x = series.values.astype(float)
    N = len(x)
    if N < L:
        return np.nan
    K = N - L + 1
    X = np.column_stack([x[i:i+L] for i in range(K)])
    U, V = robust_low_rank(X, q)
    X_rob = U @ V.T
    rec, counts = np.zeros(N), np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i+j]    += X_rob[i, j]
            counts[i+j] += 1
    rec /= counts
    return rec[-L:].mean()

# ----------------------------
# Contract‑finder (unchanged)
# ----------------------------
def find_contract(ticker: str, year: int, month: int):
    root    = Path().resolve().parent.parent / "Complete Data" / f"{ticker}_Historic_Data"
    m0      = datetime(year, month, 1)
    mend    = m0 + relativedelta(months=1) - timedelta(days=1)
    pattern = re.compile(rf"^{ticker}[_-](\d{{4}})-(\d{{2}})\.csv$")
    candidates = []
    for p in root.iterdir():
        m = pattern.match(p.name)
        if not m:
            continue
        fy, fm = int(m.group(1)), int(m.group(2))
        diff = (fy - year)*12 + (fm - month)
        if diff < 2:
            continue
        df = pd.read_csv(p, parse_dates=["Date"])
        if df["Date"].max() < mend + timedelta(days=15):
            continue
        mdf = df[(df["Date"] >= m0) & (df["Date"] <= mend)]
        if mdf.empty:
            continue
        candidates.append((diff, p.name, mdf.sort_values("Date")))
    if not candidates:
        return None, None
    _, fname, mdf = min(candidates, key=lambda x: x[0])
    return fname, mdf

# ----------------------------
# Load monthly returns
# ----------------------------
monthly_dir = Path().resolve().parent.parent / "Complete Data" / "All_Monthly_Return_Data"
files       = list(monthly_dir.glob("*_Monthly_Revenues.csv"))
print(f"Found {len(files)} monthly‑revenue files.")

# RLSSA parameters
RLSSA_WINDOW = 12
RLSSA_RANK   = 2

# ----------------------------
# Build selection history using RLSSA
# ----------------------------
history = []
current = initial_end
while current <= final_end:
    ny, nm = (current + pd.DateOffset(months=1)).year, (current + pd.DateOffset(months=1)).month

    # compute each ticker's RLSSA score
    scores = []
    for f in files:
        ticker = f.stem.replace("_Monthly_Revenues","")
        df     = pd.read_csv(f)
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        df['date']   = pd.to_datetime(df[['year','month']].assign(day=1))
        df = df[(df['date'] >= start_date) & (df['date'] <= current)]
        if LOOKBACK_YEARS is not None:
            cutoff = current - relativedelta(years=LOOKBACK_YEARS) + relativedelta(days=1)
            df     = df[df['date'] >= cutoff]
        if len(df) < RLSSA_WINDOW:
            continue

        series = df.set_index('date')['return'].sort_index()
        score  = rlssa_score(series, RLSSA_WINDOW, RLSSA_RANK)
        if pd.notna(score):
            scores.append((abs(score), score, ticker))

    if scores:
        _, best_score, best_tkr = max(scores, key=lambda x: x[0])
        sig = best_tkr if best_score > 0 else f"-{best_tkr}"
        fname, mdf = find_contract(best_tkr, ny, nm)
    else:
        sig, fname, mdf = "NoContract", None, None

    history.append({
        'analysis_end':   current.strftime("%Y-%m-%d"),
        'forecast_month': f"{ny}-{nm:02d}",
        'signal':         sig,
        'contract_file':  fname,
        'daily_df':       mdf
    })
    current += pd.DateOffset(months=1)

hist_df = pd.DataFrame(history)

# ----------------------------
# Display selection table
# ----------------------------
display_df = hist_df[[
    'analysis_end','forecast_month','signal','contract_file'
]]
print(display_df.to_string(index=False))


# ----------------------------
# Daily‑compounded portfolio simulation
# ----------------------------
vc_nc = vc_wc = START_VALUE
dates, vals_nc, vals_wc = [], [], []

for _, row in hist_df.iterrows():
    fm = row['forecast_month']
    year, month = map(int, fm.split('-'))
    if not (plot_start <= datetime(year,month,1) <= plot_end):
        continue

    sig = row['signal']
    mdf = row['daily_df']

    # If no contract that month, hold flat
    if mdf is None:
        month_start = datetime(year, month, 1)
        month_end   = month_start + pd.offsets.MonthEnd(0)
        dates.append(month_start)
        vals_nc.append(vc_nc); vals_wc.append(vc_wc)
        dates.append(month_end)
        vals_nc.append(vc_nc); vals_wc.append(vc_wc)
        continue

    # Entry cost
    vc_wc *= (1 - ENTRY_COST)
    prev_close = None

    for r in mdf.itertuples():
        date  = r.Date
        open_ = r.open
        close = r.close
        ratio = (close/open_) if prev_close is None else (close/prev_close)
        prev_close = close

        sign = -1 if sig.startswith('-') else 1
        vc_nc *= (1 + sign*(ratio-1))
        vc_wc *= (1 + sign*(ratio-1))

        dates.append(date)
        vals_nc.append(vc_nc)
        vals_wc.append(vc_wc)

    # Exit cost
    vc_wc *= (1 - EXIT_COST)

perf = pd.DataFrame({'Date':dates,'NoCosts':vals_nc,'WithCosts':vals_wc})
perf.set_index('Date', inplace=True)

# ----------------------------
# Plot performance
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(perf.index, perf['NoCosts'], label='No Costs')
plt.plot(perf.index, perf['WithCosts'], label='With Costs')
plt.xlabel('Date'); plt.ylabel('Portfolio Value (CHF)')
plt.title('Seasonal Strategy via RLSSA Daily‑Compounded')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(); plt.grid(True)
plt.xlim(plot_start, plot_end)
plt.tight_layout()
plt.show()


# ----------------------------
# Helper to inspect daily returns
# ----------------------------
def show_daily_returns_and_monthly_return(year: int, month: int):
    fm  = f"{year}-{month:02d}"
    row = hist_df.loc[hist_df['forecast_month']==fm].squeeze()
    if row.empty or row.daily_df is None:
        print(f"No data for forecast month {fm}.")
        return None

    sig      = row.signal
    fname    = row.contract_file
    mdf      = row.daily_df.copy().sort_values("Date")

    print(f"Forecast {fm} → signal = {sig}, contract = {fname}")
    prev, rets = None, []
    for r in mdf.itertuples():
        ret = (r.close/r.open - 1) if prev is None else (r.close/prev - 1)
        rets.append(ret); prev = r.close

    mdf['daily_return'] = rets
    print(mdf[["Date","open","close","daily_return","volume"]].to_string(index=False))
    return mdf

# === Example ===
show_daily_returns_and_monthly_return(2020, 3)
show_daily_returns_and_monthly_return(2020, 4)
