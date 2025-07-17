import re
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

# -----------------------------------------------------------------------------
# 1) PARAMETERS
# -----------------------------------------------------------------------------
START_DATE      = datetime(2001, 1, 1)
LOOKBACK_YEARS  = 10   # used for building history only
CALIB_YEARS     = 5
FINAL_END       = datetime(2024, 12, 31)

L               = 12    # SSA window length
Q               = 2     # number of robust components

SIG_LEVEL       = 0.05
NUM_SELECT      = 2
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

# -----------------------------------------------------------------------------
# 3) LOAD RETURNS & BUILD SSA HISTORY
# -----------------------------------------------------------------------------
def load_returns():
    out = {}
    for f in sorted(MONTHLY_DIR.glob("*_Monthly_Revenues.csv")):
        tkr = f.stem.replace("_Monthly_Revenues", "")
        df  = pd.read_csv(f)
        df['date']   = pd.to_datetime(df[['year','month']].assign(day=1))
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        out[tkr] = df.set_index('date')['return'].sort_index()
    return out

def build_ssa_history(returns):
    start = (START_DATE + relativedelta(years=LOOKBACK_YEARS)).replace(day=1)
    end   = FINAL_END.replace(day=1)
    dates = pd.date_range(start, end, freq='MS')
    ssa_df = pd.DataFrame(index=dates, columns=returns.keys(), dtype=float)
    for dt in dates:
        lb0 = dt - relativedelta(years=LOOKBACK_YEARS)
        lb1 = dt - relativedelta(months=1)
        for tkr, series in returns.items():
            window = series.loc[lb0:lb1].values
            ssa_df.at[dt, tkr] = rlssa_score(window, L, Q)
    return ssa_df

# -----------------------------------------------------------------------------
# 4) CONTRACT FINDER
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
# 5) PREP & SSA TABLE
# -----------------------------------------------------------------------------
returns   = load_returns()
ssa_score = build_ssa_history(returns)
tickers   = list(returns)

initial_lb_end       = (START_DATE + relativedelta(years=LOOKBACK_YEARS)
                        - pd.offsets.MonthEnd(1))
first_trade_forecast = (initial_lb_end
                        + pd.offsets.MonthBegin(1)
                        + relativedelta(years=CALIB_YEARS))

print(f"[INFO] First forecast on {first_trade_forecast.date()}")

# -----------------------------------------------------------------------------
# 6) MAIN LOOP
# -----------------------------------------------------------------------------
records = []
cur     = initial_lb_end + pd.offsets.MonthBegin(0)

while cur <= FINAL_END:
    if cur < first_trade_forecast:
        cur += relativedelta(months=1)
        continue

    # current robust-SSA vector
    current = ssa_score.loc[cur]

    # calibration slice
    cal0   = cur - relativedelta(years=CALIB_YEARS)
    cal1   = cur - relativedelta(months=1)
    calib  = ssa_score.loc[cal0:cal1]

    means  = calib.mean()
    stds   = calib.std(ddof=1)

    # debug calibration block
    if cur == DEBUG_DATE:
        print(f"\n[DEBUG] Calibration block @ {DEBUG_DATE.date()}:\n", calib, means, stds)

    # build raw list with z & p
    raw = []
    for t in tickers:
        sc = current[t]
        if np.isnan(sc) or stds[t] <= 0:
            continue
        z = (sc - means[t]) / stds[t]
        p = 2*(1 - norm.cdf(abs(z)))
        raw.append((t, sc, z, p))

    # debug raw z/p table
    if cur == DEBUG_DATE:
        df_dbg = pd.DataFrame(raw, columns=['Tkr','SSA','Z','P']).set_index('Tkr')
        print(f"[DEBUG] SSA/Z/P @ {DEBUG_DATE.date()}:\n", df_dbg)

    # select signals
    longs  = sorted([r for r in raw if r[2]>0 and r[3]<=SIG_LEVEL], key=lambda x: x[3])
    shorts = sorted([r for r in raw if r[2]<0 and r[3]<=SIG_LEVEL], key=lambda x: x[3])
    picks  = []
    if MODE=="Long" and (not STRICT_SEL or len(longs)>=NUM_SELECT):
        picks = [t for t,_,_,_ in longs[:NUM_SELECT]]
    elif MODE=="Short" and (not STRICT_SEL or len(shorts)>=NUM_SELECT):
        picks = [f"-{t}" for t,_,_,_ in shorts[:NUM_SELECT]]
    elif MODE=="LongShort":
        half = NUM_SELECT//2
        if not STRICT_SEL or (len(longs)>=half and len(shorts)>=half):
            picks = [t for t,_,_,_ in longs[:half]] + \
                    [f"-{t}" for t,_,_,_ in shorts[:half]]

    # fetch contracts & record
    daily, contrib = {}, []
    comb = 0.0; n = max(1, len(picks))
    for sig in picks:
        tkr, mdf = find_contract(sig.lstrip('-'), cur.year, cur.month)
        daily[sig] = mdf
        r = returns[tkr].get(datetime(cur.year, cur.month, 1), np.nan)
        w = (-r if sig.startswith('-') else r)/n
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
# 7) TABULATE & PLOT
# -----------------------------------------------------------------------------
hist_df = pd.DataFrame([{
    'forecast':    rec['forecast'],
    'signals':     rec['signals'],
    'contribs':    rec['contribs'],
    'combined_ret':rec['combined']
} for rec in records])

print(hist_df.to_string(index=False))

# -----------------------------------------------------------------------------
# PERFORMANCE & PLOT (no costs when not trading)
# -----------------------------------------------------------------------------
vc_nc = vc_wc = START_VALUE
dates, nc, wc = [], [], []

for rec in records:
    fc = rec['forecast']
    if not (PLOT_START <= fc <= PLOT_END):
        continue

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
                if sig.startswith('-'):
                    ret = -ret
                rs += ret/len(daily)
                prevs[sig] = r.close
            vc_nc *= (1+rs)
            vc_wc *= (1+rs)
            dates.append(d)
            nc.append(vc_nc)
            wc.append(vc_wc)
        vc_wc *= (1 - EXIT_COST)

perf = pd.DataFrame({'Date':dates,'NoCosts':nc,'WithCosts':wc}).set_index('Date')

initial_val = perf['NoCosts'].iloc[0]
final_nc    = perf['NoCosts'].iloc[-1]
final_wc    = perf['WithCosts'].iloc[-1]
tot_nc      = (final_nc/initial_val - 1)*100
tot_wc      = (final_wc/initial_val - 1)*100

title_str = f"RLSSA_{MODE}_Portfolio_{NUM_SELECT}_Assets_&_Lookback_{LOOKBACK_YEARS}Y_SL_{SIG_LEVEL}.png"
output_dir = Path("plots/RLSSA_Plots")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10,6))
plt.plot(perf.index, perf['NoCosts'],  label=f'No Costs (Total: {tot_nc:.2f}%)')
plt.plot(perf.index, perf['WithCosts'],label=f'With Costs (Total: {tot_wc:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (CHF)')
plt.title(f'RLSSA {MODE} Portfolio with {NUM_SELECT} Assets & Lookback of {LOOKBACK_YEARS} Years & SL {SIG_LEVEL}')
plt.legend()
plt.grid(True)
plt.xlim(PLOT_START, PLOT_END)
plt.tight_layout()
#plt.show()

save_path = output_dir / title_str
plt.savefig(save_path, dpi=300)