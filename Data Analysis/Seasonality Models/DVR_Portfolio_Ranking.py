import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from decimal import Decimal, getcontext

# -----------------------------------------------------------------------------
# 1) PARAMETERS & WINDOWS
# -----------------------------------------------------------------------------
START_YEAR, START_MONTH = 2001, 1
FINAL_END       = datetime(2024, 12, 31)
LOOKBACK_YEARS  = 10
SIG_LEVEL       = 1
START_VALUE     = 1000.0
PLOT_START, PLOT_END = datetime(2011, 1, 1), datetime(2024, 12, 31)


# -----------------------------------------------------------------------------
# 2) DATE RANGES
# -----------------------------------------------------------------------------
START_DATE   = datetime(START_YEAR, START_MONTH, 1)
LOOKBACK_END = (START_DATE + relativedelta(years=LOOKBACK_YEARS)
                - pd.offsets.MonthEnd(1))
RANK_START   = LOOKBACK_END + pd.offsets.MonthBegin(1)

print(f"Lookback: {START_DATE.date()} → {LOOKBACK_END.date()}")
print(f"Ranking  : {RANK_START.date()} → {FINAL_END.date()}")

# -----------------------------------------------------------------------------
# 3) HELPERS
# -----------------------------------------------------------------------------
def load_returns(root_dir: Path) -> dict[str, pd.Series]:
    out = {}
    for f in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        ticker = f.stem.replace("_Monthly_Revenues","")
        df = (pd.read_csv(f)
              .assign(
                  date=lambda d: pd.to_datetime(d[['year','month']].assign(day=1)),
                  rtn =lambda d: pd.to_numeric(d['return'], errors='coerce')
              )
              .set_index('date')['rtn']
              .sort_index())
        out[ticker] = df
    return out

def find_contract(tkr: str, yr: int, mo: int):
    root = Path().resolve().parent.parent/"Complete Data"/f"{tkr}_Historic_Data"
    m0   = datetime(yr, mo, 1)
    mend = m0 + relativedelta(months=1) - timedelta(days=1)
    pat  = re.compile(rf"^{tkr}[_-](\d{{4}})-(\d{{2}})\.csv$")
    cands=[]
    for p in root.iterdir():
        m = pat.match(p.name)
        if not m: continue
        fy,fm = map(int,m.groups())
        if (fy-yr)*12+(fm-mo)<2: continue
        df = pd.read_csv(p, parse_dates=['Date'])
        if df.Date.max()<mend+timedelta(days=15): continue
        mdf = df[(df.Date>=m0)&(df.Date<=mend)]
        if not mdf.empty:
            cands.append(((fy-yr)*12+(fm-mo), mdf.sort_values('Date')))
    return None if not cands else min(cands, key=lambda x:x[0])[1]

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

# -----------------------------------------------------------------------------
# 4) LOAD DATA
# -----------------------------------------------------------------------------
base        = Path().resolve().parent.parent/"Complete Data"
log_rets    = load_returns(base/"All_Monthly_Log_Return_Data")
simple_rets = load_returns(base/"All_Monthly_Return_Data")
tickers     = list(log_rets)
NUM_T       = len(tickers)

# -----------------------------------------------------------------------------
# 5) ROLLING REGRESSION & RANKING
# -----------------------------------------------------------------------------
long_rankings, short_rankings = {}, {}
# and we’ll store models for the best tickers each month:
models_long, models_short = {}, {}

cur = RANK_START
while cur <= FINAL_END:
    m     = cur.month
    stats = []

    hb0 = cur - relativedelta(years=LOOKBACK_YEARS)
    hb1 = cur - relativedelta(months=1)

    # run regression for each ticker
    for t in tickers:
        s = log_rets[t].loc[hb0:hb1].dropna()
        if len(s)<12: continue

        df = (s.to_frame('rtn')
              .assign(
                month=lambda d: d.index.month,
                D=lambda d: (d.month==m).astype(float)
              ))
        X   = sm.add_constant(df['D'])
        res = sm.OLS(df['rtn'], X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
        stats.append({
            'ticker': t,
            'beta': res.params['D'],
            'pval': res.pvalues['D'],
            'avg':  df.loc[df.month==m,'rtn'].mean(),
            'model': res
        })

    if not stats:
        long_rankings[cur]=tickers.copy()
        short_rankings[cur]=tickers.copy()
        cur+=relativedelta(months=1)
        continue

    dfm = pd.DataFrame(stats).set_index('ticker')

    # Long ranking
    sigL = dfm[(dfm.pval<=SIG_LEVEL)&(dfm.beta>0)&(dfm.avg>0)]
    restL= dfm.drop(sigL.index,errors='ignore')
    orderL = list(sigL.sort_values('avg',ascending=False).index) \
           + list(restL.sort_values('avg',ascending=False).index)
    orderL += [t for t in tickers if t not in orderL]
    long_rankings[cur]=orderL

    # Short ranking
    sigS = dfm[(dfm.pval<=SIG_LEVEL)&(dfm.beta<0)&(dfm.avg<0)]
    restS= dfm.drop(sigS.index,errors='ignore')
    orderS = list(sigS.sort_values('avg',ascending=True).index) \
           + list(restS.sort_values('avg',ascending=True).index)
    orderS += [t for t in tickers if t not in orderS]
    short_rankings[cur]=orderS

    # Capture the model object for the best long & short
    bestL = orderL[0]
    bestS = orderS[0]
    models_long[cur]  = dfm.loc[bestL,'model']
    models_short[cur] = dfm.loc[bestS,'model']

    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# 7) PRINT MONTHLY SELECTION + FULL RANKINGS + REGRESSION SUMMARIES
# -----------------------------------------------------------------------------
for dt in sorted(models_long.keys()):
    bl = long_rankings[dt][0]
    bs = short_rankings[dt][0]

    rL = simple_rets[bl].get(dt, np.nan)
    rS = simple_rets[bs].get(dt, np.nan)

    #print(f"\n=== {dt.date()} ===")
    #print(f"Best Long : {bl}  return {rL:.2%}")
    #print("Long Ranking with returns:")
    for rank, tkr in enumerate(long_rankings[dt], start=1):
        rtn = simple_rets[tkr].get(dt, np.nan)
        if rank == 1:
            #print(f"  {dt.date()}. {tkr:4s} → {rtn:.2%}")
            print(f"{rtn:.4%}")
    #print(models_long[dt].summary())

    #print(f"\nBest Short: {bs}  return {rS:.2%}")
    #print("Short Ranking with returns:")
    for rank, tkr in enumerate(short_rankings[dt], start=1):
        rtn = simple_rets[tkr].get(dt, np.nan)
        # invert sign to show the effective P&L direction
        rtn_display = 1/(1+rtn)-1 if not np.isnan(rtn) else np.nan
        #print(f"  {rank:2d}. {tkr:4s} → {rtn_display:.2%}")
        #print(models_short[dt].summary())

# -----------------------------------------------------------------------------
# Calculate cumulative returns for all ranks
# -----------------------------------------------------------------------------
long_cum_df, _   = compute_cum(long_rankings,  direction='long')
short_cum_df, _  = compute_cum(short_rankings, direction='short')

# -----------------------------------------------------------------------------
# Print the top‑17 long portfolios
# -----------------------------------------------------------------------------
print("\nCumulative returns for LONG portfolios (ranks 1–17):")
for rank in range(1, 17):
    cum = long_cum_df.loc[rank, 'cum_ret']
    print(f" Rank {rank:2d}: {cum:.2%}")

# -----------------------------------------------------------------------------
# Print the top‑17 short portfolios
# -----------------------------------------------------------------------------
print("\nCumulative returns for SHORT portfolios (ranks 1–17):")
for rank in range(1, 17):
    cum = short_cum_df.loc[rank, 'cum_ret']
    print(f" Rank {rank:2d}: {cum:.2%}")

