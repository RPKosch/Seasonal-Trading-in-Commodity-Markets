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
START_DATE              = datetime(2001, 1, 1)
FINAL_END               = datetime(2024, 12, 31)
LOOKBACK_YEARS          = 10
TEST_YEARS              = 5       # length of testing period in years
SIM_YEARS        = 5      # 5 yrs for the actual simulation
SIG_LEVEL               = 0.05
START_VALUE             = 1000.0

DEBUG_DATE              = datetime(2011, 1, 1)

# -----------------------------------------------------------------------------
# 2) DATE RANGES
# -----------------------------------------------------------------------------
LOOKBACK_END = (START_DATE + relativedelta(years=LOOKBACK_YEARS) - pd.offsets.MonthEnd(1))

TEST_SIM_START   = START_DATE + relativedelta(years=LOOKBACK_YEARS)
TEST_SIM_END     = TEST_SIM_START + relativedelta(years=SIM_YEARS) - pd.offsets.MonthEnd(1)

FINAL_SIM_START        = START_DATE + relativedelta(years=LOOKBACK_YEARS) + relativedelta(years=SIM_YEARS)
FINAL_SIM_END          = FINAL_END

print(f"Lookback: {START_DATE.date()} → {LOOKBACK_END.date()}")
print(f"Testing:  {TEST_SIM_START.date()} → {TEST_SIM_END.date()}")
print(f"Holdout:  {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")

# -----------------------------------------------------------------------------
# 3) HELPERS
# -----------------------------------------------------------------------------
def load_returns(root_dir: Path) -> dict[str, pd.Series]:
    out = {}
    for f in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        ticker = f.stem.replace("_Monthly_Revenues", "")
        df = (
            pd.read_csv(f)
              .assign(
                  date=lambda d: pd.to_datetime(
                      d[['year','month']].assign(day=1)
                  ),
                  rtn=lambda d: pd.to_numeric(d['return'], errors='coerce')
              )
              .set_index('date')['rtn']
              .sort_index()
        )
        out[ticker] = df
    return out

def compute_cum(rankings, direction='long', start_date=None, end_date=None):
    holdings = {r: {} for r in range(1, NUM_T+1)}
    rets     = {r: [] for r in range(1, NUM_T+1)}

    for dt, order in rankings.items():
        if start_date and dt < start_date: continue
        if end_date   and dt > end_date:   continue
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

# -----------------------------------------------------------------------------
# 3b) RATIO & SCORING HELPERS (with empty-input guards)
# -----------------------------------------------------------------------------
def sharpe_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0: return np.nan
    mean, std = arr.mean(), arr.std(ddof=1)
    return mean/std * np.sqrt(12) if std else np.nan

def sortino_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0: return np.nan
    neg = arr[arr<0]
    if neg.size == 0: return np.nan
    return arr.mean() / neg.std(ddof=1) * np.sqrt(12)

def calmar_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0: return np.nan
    cum      = np.cumprod(1 + arr)
    peak     = np.maximum.accumulate(cum)
    dd       = cum/peak - 1
    mdd      = abs(dd.min())
    years    = len(arr) / 12
    if mdd==0 or years==0: return np.nan
    cagr = cum[-1]**(1/years) - 1
    return cagr / mdd

def gps_harmonic(scores: list[float]) -> float:
    # keep only strictly positive, non‑NaN entries
    vals = [s for s in scores if s > 0 and not np.isnan(s)]
    n = len(vals)
    if  n != 4:
        return -999
    return (n / sum(1.0/s for s in vals)) if n else -999

def build_metrics(cum_df, rets_dict):
    rows=[]
    for prev_rank, cum_row in cum_df.iterrows():
        rets = rets_dict[prev_rank]
        sr, sor, cr = sharpe_ratio(rets), sortino_ratio(rets), calmar_ratio(rets)
        cum = float(cum_row['cum_ret'])
        score = gps_harmonic([cum, sr, sor, cr])
        rows.append({
            'prev_rank': prev_rank,
            'cum_ret':   cum,
            'sharpe':    sr,
            'sortino':   sor,
            'calmar':    cr,
            'score':     score
        })
    df = pd.DataFrame(rows).set_index('prev_rank')
    df['new_rank']   = df['score'].rank(ascending=False, method='first')
    df['rank_change']= df.index - df['new_rank']
    return df.sort_index()

# -----------------------------------------------------------------------------
# 4) LOAD DATA
# -----------------------------------------------------------------------------
base        = Path().resolve().parent.parent / "Complete Data"
log_rets    = load_returns(base / "All_Monthly_Log_Return_Data")
simple_rets = load_returns(base / "All_Monthly_Return_Data")
tickers     = list(log_rets)
NUM_T       = len(tickers)

# -----------------------------------------------------------------------------
# 5) ROLLING REG & DVR RANKINGS
# -----------------------------------------------------------------------------
long_rankings, short_rankings = {}, {}
cur = TEST_SIM_START
while cur <= FINAL_END:
    m = cur.month
    stats = []
    hb0 = cur - relativedelta(years=LOOKBACK_YEARS)
    hb1 = cur - relativedelta(months=1)

    for t in tickers:
        s = log_rets[t].loc[hb0:hb1].dropna()
        if len(s) < 12:
            continue
        df = (s.to_frame('rtn')
              .assign(month=lambda d: d.index.month,
                      D=lambda d: (d.month==m).astype(float)))
        X   = sm.add_constant(df['D'])
        res = sm.OLS(df['rtn'], X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
        stats.append({
            'ticker': t,
            'beta':   res.params['D'],
            'pval':   res.pvalues['D'],
            'avg':    df.loc[df.month==m,'rtn'].mean()
        })

    if not stats:
        long_rankings[cur] = tickers.copy()
        short_rankings[cur]= tickers.copy()
    else:
        dfm = pd.DataFrame(stats).set_index('ticker')
        # LONG DVR
        sigL = dfm[(dfm.pval<=SIG_LEVEL)&(dfm.beta>0)&(dfm.avg>0)]
        restL= dfm.drop(sigL.index, errors='ignore')
        orderL = list(sigL.sort_values('avg',ascending=False).index) + \
                 list(restL.sort_values('avg',ascending=False).index)
        orderL += [t for t in tickers if t not in orderL]
        long_rankings[cur] = orderL
        # SHORT DVR
        sigS = dfm[(dfm.pval<=SIG_LEVEL)&(dfm.beta<0)&(dfm.avg<0)]
        restS= dfm.drop(sigS.index, errors='ignore')
        orderS = list(sigS.sort_values('avg',ascending=True).index) + \
                 list(restS.sort_values('avg',ascending=True).index)
        orderS += [t for t in tickers if t not in orderS]
        short_rankings[cur] = orderS

    # debug if needed
    if DEBUG_DATE and cur == DEBUG_DATE:
        print(f"--- DEBUG at {cur.date()} ---")
        dbg = pd.DataFrame(stats).set_index('ticker')[['beta','pval','avg']]
        print(dbg)
        print("Long order:", long_rankings[cur])
        print("Short order:", short_rankings[cur])
        print("-------------------------------------------------------------------------------------------------")

    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# 6) TEST & HOLDOUT for LONG & SHORT
# -----------------------------------------------------------------------------
# Long test-period & metrics
long_test_cum_df, _, long_test_rets = compute_cum(
    long_rankings, 'long', TEST_SIM_START, TEST_SIM_END
)
metrics_long_test = build_metrics(long_test_cum_df, long_test_rets)

# Short test-period & metrics
short_test_cum_df, _, short_test_rets = compute_cum(
    short_rankings, 'short', TEST_SIM_START, TEST_SIM_END
)
metrics_short_test = build_metrics(short_test_cum_df, short_test_rets)

# Compute final holdout returns directly using original DVR rankings
long_holdout_cum_df, _, _  = compute_cum(
    long_rankings, 'long', FINAL_SIM_START, FINAL_SIM_END
)
short_holdout_cum_df, _, _ = compute_cum(
    short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END
)

# -----------------------------------------------------------------------------
# 7) OUTPUT & SUMMARY
# -----------------------------------------------------------------------------
print(f"Testing Portfolios for Period {TEST_SIM_START} until {TEST_SIM_END}")

print("\nMetrics for LONG portfolios (ranks 1–17):")
print(metrics_long_test.loc[1:17].to_string())
print("\nMetrics for SHORT portfolios (ranks 1–17):")
print(metrics_short_test.loc[1:17].to_string())
print("-------------------------------------------------------------------------------------------------")

print(f"Final Portfolios for Period {FINAL_SIM_START} until {FINAL_END}")
print("\nMetrics for LONG portfolios (ranks 1–17):")
print(long_holdout_cum_df.loc[1:17].to_string())
print("\nMetrics for SHORT portfolios (ranks 1–17):")
print(short_holdout_cum_df.loc[1:17].to_string())

def output_comparison(metrics_calib, orig_cum):
    print(f"\nTesting Period: {TEST_SIM_START.date()} → {TEST_SIM_END.date()}")
    print(f"Final Simulation: {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")
    rows=[]
    for prev_rank, row in metrics_calib.iterrows():
        pr = int(prev_rank)
        nr = (row['new_rank'])
        new_ret = float(orig_cum.loc[pr, 'cum_ret'])
        orig_ret  = float(orig_cum.loc[nr, 'cum_ret'])
        orig_score = float(row['score'])
        diff = new_ret - orig_ret
        rows.append({
            'prev_rank': pr,
            'new_rank' : nr,
            'holdout_ret_now': new_ret,
            'holdout_ret_same_rank_prev' : orig_ret,
            'difference'       : diff,
            'score'            :  orig_score
        })
    return pd.DataFrame(rows).set_index('new_rank').sort_index()

print("Long comparison (orig vs new holdout returns):")
long_comp = output_comparison(
    metrics_long_test,
    long_holdout_cum_df)
print(long_comp.to_string())

print("Short comparison (orig vs new holdout returns):")
short_comp = output_comparison(
    metrics_short_test,
    short_holdout_cum_df)
print(short_comp.to_string())

