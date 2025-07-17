import re
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# -----------------------------------------------------------------------------
# 1) PARAMETERS
# -----------------------------------------------------------------------------
START_DATE      = datetime(2001, 1, 1)
LOOKBACK_YEARS  = 10   # only used to build SSA history
CALIB_YEARS     = 5
FINAL_END       = datetime(2024, 12, 31)

L               = 12    # SSA window length
Q               = 2     # number of robust components
SIG_LEVEL       = 0.05
NUM_SELECT      = 1
STRICT_SEL      = True
MODE            = "Short"   # "Long", "Short", or "LongShort"

ENTRY_COST      = EXIT_COST = 0.0025
START_VALUE     = 1000.0
PLOT_START, PLOT_END = datetime(2016, 1, 1), datetime(2024, 12, 31)

DEBUG_DATE      = datetime(2016, 8, 1)

# Data paths
ROOT_DIR    = Path().resolve().parent.parent / "Complete Data"
MONTHLY_DIR = ROOT_DIR / "All_Monthly_Return_Data"

# -----------------------------------------------------------------------------
# 2) BUILD SSA HISTORY
# -----------------------------------------------------------------------------
def load_returns():
    out = {}
    for f in sorted(MONTHLY_DIR.glob("*_Monthly_Revenues.csv")):
        tkr = f.stem.replace("_Monthly_Revenues", "")
        df  = pd.read_csv(f)
        df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        out[tkr] = df.set_index('date')['return'].sort_index()
    return out

# -----------------------------------------------------------------------------
# 2) ROBUST SSA HELPERS
# -----------------------------------------------------------------------------
def robust_low_rank(X, q, max_iter=25, eps=1e-7):
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

def rlssa_score(series: np.ndarray, L: int, q: int) -> float:
    N = len(series)
    if N < L:
        return np.nan
    K = N - L + 1
    X = np.column_stack([series[i:i+L] for i in range(K)])
    U, V = robust_low_rank(X, q)
    Xr = U @ V.T
    rec = np.zeros(N)
    cnt = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i+j] += Xr[i, j]
            cnt[i+j] += 1
    rec /= cnt
    return rec[-L:].mean()

def build_ssa_history(returns):
    start = (START_DATE + relativedelta(years=LOOKBACK_YEARS)).replace(day=1)
    end   = FINAL_END.replace(day=1)
    dates = pd.date_range(start, end, freq='MS')
    ssa_df = pd.DataFrame(index=dates, columns=returns.keys(), dtype=float)
    for dt in dates:
        lb0 = dt - relativedelta(years=LOOKBACK_YEARS)
        lb1 = dt - relativedelta(months=1)
        for tkr, series in returns.items():
            ssa_df.at[dt, tkr] = rlssa_score(series.loc[lb0:lb1].values, L, Q)
    return ssa_df

# -----------------------------------------------------------------------------
# 3) CONTRACT FINDER
# -----------------------------------------------------------------------------
def find_contract(tkr, y, m):
    root = ROOT_DIR / f"{tkr}_Historic_Data"
    m0   = datetime(y, m, 1)
    mend = m0 + relativedelta(months=1) - timedelta(days=1)
    pat  = re.compile(rf"^{tkr}[_-](\d{{4}})-(\d{{2}})\.csv$")
    cands = []
    for p in root.iterdir():
        mm = pat.match(p.name)
        if not mm: continue
        fy, fm = map(int, mm.groups())
        if (fy-y)*12 + (fm-m) < 2: continue
        df = pd.read_csv(p, parse_dates=['Date'])
        if df.Date.max() < mend + timedelta(days=15): continue
        mdf = df[(df.Date>=m0)&(df.Date<=mend)]
        if mdf.empty: continue
        diff = (fy-y)*12 + (fm-m)
        cands.append((diff, mdf.sort_values('Date')))
    if not cands:
        return None, None
    _, mdf = min(cands, key=lambda x: x[0])
    return tkr, mdf

# -----------------------------------------------------------------------------
# 4) PREP & SSA TABLE
# -----------------------------------------------------------------------------
returns   = load_returns()
ssa_score = build_ssa_history(returns)
print(ssa_score)
tickers   = list(returns)

# first possible forecast date
initial_lb_end       = (START_DATE + relativedelta(years=LOOKBACK_YEARS)
                        - pd.offsets.MonthEnd(1))
first_trade_forecast = (initial_lb_end
                        + pd.offsets.MonthBegin(1)
                        + relativedelta(years=CALIB_YEARS))

# -----------------------------------------------------------------------------
# 5) MAIN LOOP (pick by raw SSA)
# -----------------------------------------------------------------------------

records = []
cur     = initial_lb_end + pd.offsets.MonthBegin(0)

while cur <= FINAL_END:
    if cur < first_trade_forecast:
        cur += relativedelta(months=1)
        continue

    # 1) current robust‑SSA vector
    current = ssa_score.loc[cur]

    # 2) build raw list of (ticker, score)
    raw = [(t, float(current[t])) for t in tickers if not np.isnan(current[t])]

    # debug
    if cur == DEBUG_DATE:
        print(f"[DEBUG] Raw SSA scores @ {DEBUG_DATE.date()}:\n",
              pd.DataFrame(raw, columns=['Tkr','SSA']).set_index('Tkr'))

    # 3) sort for longs & shorts
    longs  = sorted(raw, key=lambda x: x[1], reverse=True)   # highest scores
    shorts = sorted(raw, key=lambda x: x[1])                # most negative

    # 4) pick signals
    picks = []
    if MODE == "Long":
        picks = [t for t,_ in longs[:NUM_SELECT]]
    elif MODE == "Short":
        picks = [f"-{t}" for t,_ in shorts[:NUM_SELECT]]
    else:  # LongShort
        half = NUM_SELECT // 2
        picks = [t for t,_ in longs[:half]] + [f"-{t}" for t,_ in shorts[:half]]

    # 5) fetch contracts & record
    daily, contrib = {}, []
    comb = 0.0; n = max(1, len(picks))
    for sig in picks:
        tkr, mdf = find_contract(sig.lstrip('-'), cur.year, cur.month)
        daily[sig] = mdf
        r = returns[tkr].get(datetime(cur.year, cur.month, 1), np.nan)
        w = (-r if sig.startswith('-') else r) / n
        comb += w
        contrib.append(f"{sig}:{r:.2%}→{w:.2%}")

    records.append({
        'forecast':  cur,
        'signals':   picks,
        'contribs':  contrib,
        'combined':  comb,
        'daily_dfs': daily
    })

    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# 6) TABULATE & PLOT
# -----------------------------------------------------------------------------
hist_df = pd.DataFrame([{
    'forecast': rec['forecast'],
    'signals':  rec['signals'],
    'contribs': rec['contribs'],
    'combined_ret': rec['combined']
} for rec in records])

print(hist_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 6) PERFORMANCE & PLOT (no costs when not trading)
# -----------------------------------------------------------------------------
vc_nc = vc_wc = START_VALUE
dates, nc, wc = [], [], []

for rec in records:
    fc = rec['forecast']
    if not (PLOT_START <= fc <= PLOT_END): continue

    daily = rec['daily_dfs']
    if not daily:
        d = (fc - relativedelta(months=1)) + pd.offsets.MonthEnd(0)
        dates += [d,d]; nc += [vc_nc,vc_nc]; wc += [vc_wc,vc_wc]
    else:
        vc_wc *= (1 - ENTRY_COST)
        all_df = pd.concat([mdf.assign(signal=s) for s,mdf in daily.items()])\
                   .sort_values('Date')
        prevs  = {s:None for s in daily}
        for d, grp in all_df.groupby('Date'):
            rs = 0.0
            for r in grp.itertuples():
                sig = r.signal
                prev = prevs[sig]
                ret  = (r.close/r.open-1) if prev is None else (r.close/prev-1)
                if sig.startswith('-'): ret = -ret
                rs += ret/len(daily)
                prevs[sig] = r.close
            vc_nc *= (1+rs); vc_wc *= (1+rs)
            dates.append(d); nc.append(vc_nc); wc.append(vc_wc)
        vc_wc *= (1 - EXIT_COST)

perf = pd.DataFrame({'Date':dates,'NoCosts':nc,'WithCosts':wc})\
           .set_index('Date')

initial_val = perf['NoCosts'].iloc[0]
final_nc    = perf['NoCosts'].iloc[-1]
final_wc    = perf['WithCosts'].iloc[-1]
tot_nc      = (final_nc/initial_val - 1)*100
tot_wc      = (final_wc/initial_val - 1)*100

plt.figure(figsize=(10,6))
plt.plot(perf.index, perf['NoCosts'],
         label=f'No Costs (Total: {tot_nc:.2f}%)')
plt.plot(perf.index, perf['WithCosts'],
         label=f'With Costs (Total: {tot_wc:.2f}%)')
plt.xlabel('Date'); plt.ylabel('Portfolio Value (CHF)')
plt.title(f'SSA {MODE} Portfolio (p≤{SIG_LEVEL})')
plt.legend(); plt.grid(True)
plt.xlim(PLOT_START, PLOT_END)
plt.tight_layout()
plt.show()

