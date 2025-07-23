import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from decimal import Decimal, getcontext

# -----------------------------------------------------------------------------
# 1) PARAMETERS & WINDOWS
# -----------------------------------------------------------------------------
START_YEAR, START_MONTH = 2001, 1
FINAL_END       = datetime(2024, 12, 31)
LOOKBACK_YEARS  = 10
SIG_LEVEL       = 0.05
START_VALUE     = 1000.0
PLOT_START, PLOT_END = datetime(2011, 1, 1), datetime(2024, 12, 31)

DEBUG_DATE      = datetime(2011, 1, 1)

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


# compute cumulative returns and store raw rets
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
    return pd.DataFrame(rows).set_index('rank'), holdings, rets


def sharpe_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    mean = arr.mean()
    std = arr.std(ddof=1)
    return mean/std * np.sqrt(12) if std else np.nan


def sortino_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    mean = arr.mean()
    neg = arr[arr<0]
    dstd = neg.std(ddof=1)
    return mean/ dstd * np.sqrt(12) if dstd else np.nan


def calmar_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    cum = np.cumprod(1+arr)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum/peak - 1)
    mdd = abs(drawdown.min())
    years = len(arr)/12
    cagr = cum[-1]**(1/years) - 1 if years>0 else np.nan
    return cagr / mdd if mdd else np.nan


def gps_harmonic(scores: list[float]) -> float:
    scores = [s for s in scores if s and not np.isnan(s)]
    n = len(scores)
    return n / np.sum([1/s for s in scores]) if n else np.nan

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
cur = RANK_START
while cur <= FINAL_END:
    m     = cur.month
    stats = []

    hb0 = cur - relativedelta(years=LOOKBACK_YEARS)
    hb1 = cur - relativedelta(months=1)

    # run regression for each ticker
    for t in tickers:
        s = log_rets[t].loc[hb0:hb1].dropna()
        if len(s) < 12:
            continue

        df = (s.to_frame('rtn')
              .assign(
                  month=lambda d: d.index.month,
                  D=lambda d: (d.month==m).astype(float)
              ))
        X   = sm.add_constant(df['D'])
        res = sm.OLS(df['rtn'], X).fit(cov_type='HAC', cov_kwds={'maxlags':1})

        stats.append({
            'ticker': t,
            'beta':   res.params['D'],
            'pval':   res.pvalues['D'],
            'avg':    df.loc[df.month==m,'rtn'].mean()
        })

    if not stats:
        long_rankings[cur]=tickers.copy()
        short_rankings[cur]=tickers.copy()
    else:
        dfm = pd.DataFrame(stats).set_index('ticker')
        sigL = dfm[(dfm.pval<=SIG_LEVEL)&(dfm.beta>0)&(dfm.avg>0)]
        restL= dfm.drop(sigL.index,errors='ignore')
        orderL = list(sigL.sort_values('avg',ascending=False).index)+list(restL.sort_values('avg',ascending=False).index)
        orderL += [t for t in tickers if t not in orderL]
        long_rankings[cur]=orderL
        sigS = dfm[(dfm.pval<=SIG_LEVEL)&(dfm.beta<0)&(dfm.avg<0)]
        restS= dfm.drop(sigS.index,errors='ignore')
        orderS = list(sigS.sort_values('avg',ascending=True).index)+list(restS.sort_values('avg',ascending=True).index)
        orderS += [t for t in tickers if t not in orderS]
        short_rankings[cur]=orderS

    # === Debug output if desired ===
    if DEBUG_DATE is not None and cur == DEBUG_DATE:
        print(f"--- DEBUG Return Input {cur.date()} ---")
        print(s)
        dbg = pd.DataFrame(stats).set_index('ticker')[['beta','pval','avg']]
        print(f"--- DEBUG REGRESSION {cur.date()} ---")
        print(dbg)
        print("Long picks order:", long_rankings[cur])
        print("Short picks order:", short_rankings[cur])
        print("------------------------------")

    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# Compute cumulative returns, holdings, and raw returns
# -----------------------------------------------------------------------------
long_cum_df, long_hold, long_rets = compute_cum(long_rankings, direction='long')
short_cum_df, short_hold, short_rets = compute_cum(short_rankings, direction='short')

# -----------------------------------------------------------------------------
# Build metrics tables
# -----------------------------------------------------------------------------
def build_metrics(cum_df, rets_dict):
    rows=[]
    for prev_rank, cum_row in cum_df.iterrows():
        rets = rets_dict[prev_rank]
        sr = sharpe_ratio(rets)
        sor = sortino_ratio(rets)
        cr = calmar_ratio(rets)
        cum = float(cum_row['cum_ret'])
        total = gps_harmonic([cum, sr, sor, cr])
        rows.append({'prev_rank':prev_rank,'cum_ret':cum,'sharpe':sr,'sortino':sor,'calmar':cr,'score':total})
    df = pd.DataFrame(rows).set_index('prev_rank')
    df['new_rank'] = df['score'].rank(ascending=False,method='first')
    df['rank_change'] = df.index - df['new_rank']
    return df.sort_index()

metrics_long = build_metrics(long_cum_df, long_rets)
metrics_short = build_metrics(short_cum_df, short_rets)

# -----------------------------------------------------------------------------
# Print results
# -----------------------------------------------------------------------------
print("\nCumulative returns for LONG portfolios (ranks 1–17):")
for rank in range(1, 18):
    cum = long_cum_df.loc[rank, 'cum_ret']
    print(f" Rank {rank:2d}: {cum:.2%}")

print("\nCumulative returns for SHORT portfolios (ranks 1–17):")
for rank in range(1, 18):
    cum = short_cum_df.loc[rank, 'cum_ret']
    print(f" Rank {rank:2d}: {cum:.2%}")

print("\nMetrics for LONG portfolios (ranks 1–17):")
print(metrics_long.loc[1:17].to_string())
print("\nMetrics for SHORT portfolios (ranks 1–17):")
print(metrics_short.loc[1:17].to_string())