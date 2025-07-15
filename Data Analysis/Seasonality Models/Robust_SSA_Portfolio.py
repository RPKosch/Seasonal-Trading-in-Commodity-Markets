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

START_VALUE      = 1000.0
ENTRY_COST       = 0.0025
EXIT_COST        = 0.0025
LOOKBACK_YEARS   = 10      # or None for full history
NUM_SELECT       = 1
STRICT_SELECTION = True
MODE             = "Short"  # "Long", "Short", or "LongShort"

PLOT_START_YEAR, PLOT_START_MONTH = 2011, 1
PLOT_END_YEAR,   PLOT_END_MONTH   = 2024, 12

# RLSSA parameters
RLSSA_WINDOW = 12
RLSSA_RANK   = 2

# ----------------------------
# Convert to datetime endpoints
# ----------------------------
start_date  = datetime(START_YEAR, START_MONTH, 1)
initial_end = datetime(INITIAL_END_YEAR, INITIAL_END_MONTH, 1) + pd.offsets.MonthEnd(0)
final_end   = datetime(FINAL_END_YEAR, FINAL_END_MONTH, 1) + pd.offsets.MonthEnd(0)
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
        if not m: continue
        fy, fm = int(m.group(1)), int(m.group(2))
        diff = (fy - year)*12 + (fm - month)
        if diff < 2: continue
        df = pd.read_csv(p, parse_dates=["Date"])
        if df.Date.max() < mend + timedelta(days=15): continue
        mdf = df[(df.Date>=m0)&(df.Date<=mend)]
        if mdf.empty: continue
        candidates.append((diff, p.name, mdf.sort_values("Date")))
    if not candidates:
        return None, None
    _, fname, mdf = min(candidates, key=lambda x: x[0])
    return fname, mdf

# ----------------------------
# Load monthly returns into dict
# ----------------------------
monthly_dir = Path().resolve().parent.parent / "Complete Data" / "All_Monthly_Return_Data"
files       = list(monthly_dir.glob("*_Monthly_Revenues.csv"))
returns     = {}
for f in files:
    t = f.stem.replace("_Monthly_Revenues","")
    df = pd.read_csv(f)
    df['return'] = pd.to_numeric(df['return'], errors='coerce')
    returns[t] = df.set_index(['year','month'])['return'].to_dict()

print(f"Found {len(files)} monthly‑revenue files.")

# ----------------------------
# Build selection history using RLSSA
# ----------------------------
history = []
current = initial_end
while current <= final_end:
    ny, nm = (current + pd.DateOffset(months=1)).year, (current + pd.DateOffset(months=1)).month

    # compute each ticker's RLSSA score
    candidates = []
    for f in files:
        ticker = f.stem.replace("_Monthly_Revenues","")
        df     = pd.read_csv(f)
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        df['date']   = pd.to_datetime(df[['year','month']].assign(day=1))
        df = df[(df.date>=start_date)&(df.date<=current)]
        if LOOKBACK_YEARS is not None:
            lb0 = current - relativedelta(years=LOOKBACK_YEARS) + relativedelta(months=1)
            df = df[df.date>=lb0]
        if len(df) < RLSSA_WINDOW:
            continue

        series = df.set_index('date')['return'].sort_index()
        score  = rlssa_score(series, RLSSA_WINDOW, RLSSA_RANK)
        if pd.notna(score):
            candidates.append((abs(score), score, ticker))

    # split longs/shorts by sign of score
    longs  = [(b,s,t) for b,s,t in candidates if s>0]
    shorts = [(b,s,t) for b,s,t in candidates if s<0]

    # select according to MODE
    selected = []
    if MODE=="Long":
        longs.sort(key=lambda x:x[0],reverse=True)
        if not (STRICT_SELECTION and len(longs)<NUM_SELECT):
            selected = [t for _,_,t in longs[:NUM_SELECT]]
    elif MODE=="Short":
        shorts.sort(key=lambda x:x[0],reverse=True)
        if not (STRICT_SELECTION and len(shorts)<NUM_SELECT):
            selected = [f"-{t}" for _,_,t in shorts[:NUM_SELECT]]
    else:  # LongShort
        half = NUM_SELECT//2
        longs.sort(key=lambda x:x[0],reverse=True)
        shorts.sort(key=lambda x:x[0],reverse=True)
        if not (STRICT_SELECTION and (len(longs)<half or len(shorts)<half)):
            selected = [t for _,_,t in longs[:half]] + [f"-{t}" for _,_,t in shorts[:half]]

    # fetch contract files & daily dfs
    contract_files = []
    daily_dfs      = {}
    for sig in selected:
        tkr = sig.lstrip('-')
        fname, mdf = find_contract(tkr, ny, nm)
        contract_files.append(fname)
        daily_dfs[sig] = mdf

    # per‐ticker monthly contributions
    contribs   = []
    combined_r = 0.0
    n = len(selected) or 1
    for sig in selected:
        tkr  = sig.lstrip('-')
        sign = -1 if sig.startswith('-') else 1
        r    = returns[tkr].get((ny,nm), float('nan'))
        w    = sign * r / n
        combined_r += w
        contribs.append(f"{sig}:{r:.2%}→{w:.2%}")

    history.append({
        'analysis_end':   current.strftime("%Y-%m-%d"),
        'forecast_month': f"{ny}-{nm:02d}",
        'signals':        selected,
        'contract_files': contract_files,
        'contribs':       contribs,
        'combined_ret':   combined_r,
        'daily_dfs':      daily_dfs
    })
    current += pd.DateOffset(months=1)

hist_df = pd.DataFrame(history)

# ----------------------------
# Display selection table
# ----------------------------
display_df = hist_df[[
    'analysis_end','forecast_month','signals',
    'contract_files','contribs','combined_ret'
]]
print(display_df.to_string(index=False))

# ----------------------------
# Daily‑compounded performance
# ----------------------------
vc_nc = vc_wc = START_VALUE
dates, vals_nc, vals_wc = [], [], []

for _, row in hist_df.iterrows():
    year, month = map(int, row['forecast_month'].split('-'))
    if not (plot_start <= datetime(year,month,1) <= plot_end):
        continue

    daily_dfs = row['daily_dfs']
    n = len(daily_dfs)

    # if none, hold flat
    if n == 0:
        m0 = datetime(year,month,1); mend = m0+pd.offsets.MonthEnd(0)
        dates += [m0, mend]; vals_nc += [vc_nc,vc_nc]; vals_wc += [vc_wc,vc_wc]
        continue

    # entry cost
    vc_wc *= (1-ENTRY_COST)

    # concat daily
    df_list = []
    for sig,mdf in row['daily_dfs'].items():
        tmp = mdf.copy(); tmp['signal']=sig
        df_list.append(tmp.set_index('Date'))
    df_all = pd.concat(df_list).sort_index()

    prev_closes={sig:None for sig in daily_dfs}
    for date,group in df_all.groupby(level=0):
        ret_sum=0.0
        for _,r2 in group.iterrows():
            sig   = r2['signal']
            prev  = prev_closes[sig]
            r_val = (r2.close/r2.open -1) if prev is None else (r2.close/prev -1)
            prev_closes[sig]=r2.close
            weight=1.0/n
            ret_sum += weight*(r_val if not sig.startswith('-') else -r_val)
        vc_nc *= (1+ret_sum)
        vc_wc *= (1+ret_sum)
        dates.append(date); vals_nc.append(vc_nc); vals_wc.append(vc_wc)

    vc_wc *= (1-EXIT_COST)

perf = pd.DataFrame({'Date':dates,'NoCosts':vals_nc,'WithCosts':vals_wc}).set_index('Date')

# ----------------------------
# Plot performance (total‐period returns)
# ----------------------------
initial_val=perf['NoCosts'].iloc[0]
final_nc   =perf['NoCosts'].iloc[-1]
final_wc   =perf['WithCosts'].iloc[-1]
tot_nc     =(final_nc/initial_val-1)*100
tot_wc     =(final_wc/initial_val-1)*100

title_str = f"RLSSA_{MODE}_Portfolio_{NUM_SELECT}_Assets_&_Lookback_{LOOKBACK_YEARS}Y.png"
output_dir = Path("plots/RLSSA_Plots")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10,6))
plt.plot(perf.index, perf['NoCosts'],
         label=f'No Costs (Total: {tot_nc:.2f}%)')
plt.plot(perf.index, perf['WithCosts'],
         label=f'With Costs (Total: {tot_wc:.2f}%)')
plt.xlabel('Date'); plt.ylabel('Portfolio Value (CHF)')
plt.title(f'RLSSA {MODE} Portfolio with {NUM_SELECT} Assets & Lookback of {LOOKBACK_YEARS} Years')
ax=plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(); plt.grid(True)
plt.xlim(plot_start,plot_end);
plt.tight_layout();
#plt.show()

# Save to disk
save_path = output_dir / title_str
plt.savefig(save_path, dpi=600)



# ----------------------------
# Helper to inspect daily returns
# ----------------------------
def show_daily_returns_and_monthly_return(year:int,month:int):
    fm  = f"{year}-{month:02d}"
    row = hist_df.loc[hist_df['forecast_month']==fm].squeeze()
    if row.empty or not row['daily_dfs']:
        print(f"No data for forecast month {fm}."); return
    print(f"Forecast {fm} → signals={row['signals']}")
    for sig,mdf in row['daily_dfs'].items():
        prev,rets=None,[]
        for r in mdf.itertuples():
            ret=(r.close/r.open -1) if prev is None else (r.close/prev -1)
            rets.append(ret); prev=r.close
        tmp=mdf.copy(); tmp['daily_return']=rets
        idx=row['signals'].index(sig)
        cf =row['contract_files'][idx]
        print(f"\n--- {sig} ({cf}) ---")
        print(tmp[['Date','open','close','daily_return','volume']].to_string(index=False))

# === Examples ===
show_daily_returns_and_monthly_return(2020,3)
show_daily_returns_and_monthly_return(2020,4)
