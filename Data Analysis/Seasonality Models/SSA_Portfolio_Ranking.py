import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

# -----------------------------------------------------------------------------
# 1) PARAMETERS & WINDOWS
# -----------------------------------------------------------------------------
START_DATE      = datetime(2001, 1, 1)
LOOKBACK_YEARS  = 10   # only used to build SSA history
CALIB_YEARS     = 5
FINAL_END       = datetime(2024, 12, 31)

SSA_WINDOW      = 12
SSA_COMPS       = 2
SIG_LEVEL       = 0.05

PLOT_START, PLOT_END = datetime(2016, 1, 1), datetime(2024, 12, 31)

START_VALUE     = 1000.0

# -----------------------------------------------------------------------------
# 2) DERIVE DATE RANGES
# -----------------------------------------------------------------------------
# lookback period → SSA history
SSA_START = (START_DATE + relativedelta(years=LOOKBACK_YEARS)).replace(day=1)
SSA_END   = FINAL_END.replace(day=1)

# first possible trade date after calibration
INITIAL_SSA_END = (START_DATE + relativedelta(years=LOOKBACK_YEARS)
                   - pd.offsets.MonthEnd(1))
FIRST_TRADE     = (INITIAL_SSA_END + pd.offsets.MonthBegin(1)
                   + relativedelta(years=CALIB_YEARS))

print(f"SSA history: {SSA_START.date()} → {SSA_END.date()}")
print(f"First trade : {FIRST_TRADE.date()}")

# -----------------------------------------------------------------------------
# 3) HELPERS
# -----------------------------------------------------------------------------
def load_monthly_returns(root_dir: Path) -> dict[str,pd.Series]:
    out = {}
    for p in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        tkr = p.stem.replace("_Monthly_Revenues","")
        df = pd.read_csv(p)
        df['date']   = pd.to_datetime(df[['year','month']].assign(day=1))
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        out[tkr] = df.set_index('date')['return'].sort_index()
    return out

def compute_ssa(x):
    N = len(x)
    if N < SSA_WINDOW: return np.nan
    K = N - SSA_WINDOW + 1
    X = np.column_stack([x[i:i+SSA_WINDOW] for i in range(K)])
    S = X @ X.T
    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(vals)[::-1]
    U = vecs[:,idx[:SSA_COMPS]] * np.sqrt(vals[idx[:SSA_COMPS]])
    V = (X.T @ vecs[:,idx[:SSA_COMPS]]) / np.sqrt(vals[idx[:SSA_COMPS]])
    Xr = U @ V.T
    rec = np.zeros(N); cnt = np.zeros(N)
    for i in range(SSA_WINDOW):
        for j in range(K):
            rec[i+j] += Xr[i,j]
            cnt[i+j] += 1
    return (rec/cnt)[-SSA_WINDOW:].mean()

def build_ssa_history(returns):
    dates = pd.date_range(SSA_START, SSA_END, freq='MS')
    df = pd.DataFrame(index=dates, columns=returns.keys(), dtype=float)
    for dt in dates:
        lb0 = dt - relativedelta(years=LOOKBACK_YEARS)
        lb1 = dt - relativedelta(months=1)
        for tkr, ser in returns.items():
            df.at[dt,tkr] = compute_ssa(ser.loc[lb0:lb1].values)
    return df

def find_contract(tkr, y, m):
    root = ROOT_DIR / f"{tkr}_Historic_Data"
    m0   = datetime(y,m,1)
    mend = m0 + relativedelta(months=1) - timedelta(days=1)
    pat = re.compile(rf"^{tkr}[_-](\d{{4}})-(\d{{2}})\.csv$")
    cands=[]
    for p in root.iterdir():
        mm = pat.match(p.name)
        if not mm: continue
        fy,fm = map(int,mm.groups())
        if (fy-y)*12+(fm-m)<2: continue
        tmp = pd.read_csv(p,parse_dates=['Date'])
        if tmp.Date.max() < mend + timedelta(days=15): continue
        mdf = tmp[(tmp.Date>=m0)&(tmp.Date<=mend)]
        if mdf.empty: continue
        cands.append(((fy-y)*12+(fm-m),mdf.sort_values('Date')))
    return None if not cands else min(cands,key=lambda x:x[0])[1]

# -----------------------------------------------------------------------------
# 4) LOAD & PREP
# -----------------------------------------------------------------------------
ROOT_DIR = Path().resolve().parent.parent / "Complete Data"
log_rets = load_monthly_returns(ROOT_DIR/"All_Monthly_Log_Return_Data")
simple_rets = load_monthly_returns(ROOT_DIR/"All_Monthly_Return_Data")
ssa_score = build_ssa_history(log_rets)
print(ssa_score)
tickers = list(log_rets)
NUM_T = len(tickers)

# -----------------------------------------------------------------------------
# 5) RANK LONG & SHORT EACH MONTH
# -----------------------------------------------------------------------------
long_rank, short_rank = {}, {}
cur = FIRST_TRADE
while cur <= SSA_END:
    block = ssa_score.loc[cur - relativedelta(years=CALIB_YEARS):
                          cur - relativedelta(months=1)]
    mu, sd = block.mean(), block.std(ddof=1)
    raw = []
    for t in tickers:
        sc = ssa_score.at[cur,t]
        if pd.isna(sc) or sd[t]==0: continue
        z = (sc-mu[t])/sd[t]
        p = 2*(1 - norm.cdf(abs(z)))
        raw.append((t,sc,z,p))
    df = pd.DataFrame(raw,columns=['tkr','ssa','z','p']).set_index('tkr')

    # select & rank signals
    longs  = sorted(
        [r for r in raw if r[1] > 0 and r[3] <= SIG_LEVEL],
        key=lambda x: x[1],
        reverse=True
    )
    shorts = sorted(
        [r for r in raw if r[1] < 0 and r[3] <= SIG_LEVEL],
        key=lambda x: x[1]
    )

    # now split out the rest and sort them the same way
    restL = sorted(
        [r for r in raw if r not in longs],
        key=lambda x: x[1],
        reverse=True
    )
    restS = sorted(
        [r for r in raw if r not in shorts],
        key=lambda x: x[1]
    )

    print(df)

    # build full ordered lists of tickers
    ordL = [t for t,_,_,_ in longs] + [t for t,_,_,_ in restL]
    ordS = [t for t,_,_,_ in shorts] + [t for t,_,_,_ in restS]
    print(ordS)


    long_rank[cur]  = ordL
    short_rank[cur] = ordS

    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# 6) COLLECT RETURNS & COMPOUND
# -----------------------------------------------------------------------------
from decimal import Decimal, getcontext
getcontext().prec = 28  # or higher if you like

def compute_cum(rankings, direction='long'):
    holdings = {r: {} for r in range(1, NUM_T+1)}
    rets     = {r: [] for r in range(1, NUM_T+1)}

    for dt, order in rankings.items():
        if not (PLOT_START <= dt <= PLOT_END):
            continue
        for r, t in enumerate(order, start=1):
            holdings[r][dt] = t
            raw = Decimal(str(simple_rets[t].get(dt, 0.0)))
            if direction == 'short':
                # exact short return: (1/(1+long)) - 1
                raw = Decimal(1) / (Decimal(1) + raw) - Decimal(1)
            rets[r].append(raw)

    rows = []
    START_D = Decimal(str(START_VALUE))
    for r in range(1, NUM_T+1):
        vc = START_D
        for x in rets[r]:
            vc *= (Decimal(1) + x)
        total = vc / START_D - Decimal(1)
        rows.append({'rank': float(r), 'cum_ret': total})
    return pd.DataFrame(rows).set_index('rank'), holdings


# then call:
long_cum, long_hold   = compute_cum(long_rank,  'long')
short_cum, short_hold = compute_cum(short_rank, 'short')

pd.set_option('display.precision', 30)

print("\nLong cumulative by rank:\n", long_cum)
print("\nShort cumulative by rank:\n", short_cum)

# -----------------------------------------------------------------------------
# Calculate cumulative returns for all ranks
# -----------------------------------------------------------------------------
long_cum_df, _   = compute_cum(long_rankings,  direction='long')
short_cum_df, _  = compute_cum(short_rankings, direction='short')

# -----------------------------------------------------------------------------
# Print the top‑17 long portfolios
# -----------------------------------------------------------------------------
print("\nCumulative returns for LONG portfolios (ranks 1–17):")
for rank in range(1, 18):
    cum = long_cum_df.loc[rank, 'cum_ret']
    print(f" Rank {rank:2d}: {cum:.2%}")

# -----------------------------------------------------------------------------
# Print the top‑17 short portfolios
# -----------------------------------------------------------------------------
print("\nCumulative returns for SHORT portfolios (ranks 1–17):")
for rank in range(1, 18):
    cum = short_cum_df.loc[rank, 'cum_ret']
    print(f" Rank {rank:2d}: {cum:.2%}")

