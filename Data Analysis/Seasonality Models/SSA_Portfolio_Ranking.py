import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
from decimal import Decimal, getcontext
import scipy.linalg as la

# -----------------------------------------------------------------------------
# 1) PARAMETERS & WINDOWS
# -----------------------------------------------------------------------------
START_DATE       = datetime(2001, 1, 1)
LOOKBACK_YEARS   = 10     # for building SSA history
ZCALIB_YEARS     = 5      # 5 yrs for Z‑score calibration
SIM_YEARS        = 5      # 5 yrs for the actual simulation
FINAL_END        = datetime(2024,12,31)
DEBUG_MONTH      = None   # e.g. datetime(2023,7,1)

SSA_WINDOW       = 12
SSA_COMPS        = 2
SIG_LEVEL        = 0.05
START_VALUE      = 1000.0

# -----------------------------------------------------------------------------
# 2) DERIVE DATE RANGES
# -----------------------------------------------------------------------------
SSA_START        = (START_DATE + relativedelta(years=LOOKBACK_YEARS))
SSA_END          = FINAL_END

# Z‑score calibration: 2011-01 → 2016-12
ZCALIB_START     = START_DATE + relativedelta(years=LOOKBACK_YEARS)
ZCALIB_END       = START_DATE + relativedelta(years=LOOKBACK_YEARS) + relativedelta(years=ZCALIB_YEARS)- pd.offsets.MonthEnd(1)

# Testing/simulation: 2017-01 → 2021-12
TEST_SIM_START        = START_DATE + relativedelta(years=LOOKBACK_YEARS) + relativedelta(years=ZCALIB_YEARS)
TEST_SIM_END          = TEST_SIM_START + relativedelta(years=SIM_YEARS) - pd.offsets.MonthEnd(1)

FINAL_SIM_START        = START_DATE + relativedelta(years=LOOKBACK_YEARS) + relativedelta(years=ZCALIB_YEARS) + relativedelta(years=SIM_YEARS)
FINAL_SIM_END          = FINAL_END

print(f"SSA history   : {SSA_START.date()} → {SSA_END.date()}")
print(f"Z‑calibration : {ZCALIB_START.date()} → {ZCALIB_END.date()}")
print(f"Test Simulation    : {TEST_SIM_START.date()} → {TEST_SIM_END.date()}")
print(f"Final Simulation    : {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")

# -----------------------------------------------------------------------------
# 3) HELPERS
# -----------------------------------------------------------------------------
def load_monthly_returns(root_dir: Path) -> dict[str,pd.Series]:
    out = {}
    for p in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        tkr = p.stem.replace("_Monthly_Revenues", "")
        df = pd.read_csv(p)
        df['date']   = pd.to_datetime(df[['year','month']].assign(day=1))
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        out[tkr] = df.set_index('date')['return'].sort_index()
    return out


def compute_ssa(x: np.ndarray) -> float:
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
    rec = np.zeros(N);
    cnt = np.zeros(N)
    for i in range(SSA_WINDOW):
        for j in range(K):
            rec[i+j] += Xr[i,j]
            cnt[i+j] += 1
    return (rec/cnt)[-SSA_WINDOW:].mean()


def build_ssa_history(returns: dict[str,pd.Series]) -> pd.DataFrame:
    dates = pd.date_range(SSA_START, SSA_END, freq='MS')
    df = pd.DataFrame(index=dates, columns=returns.keys(), dtype=float)
    for dt in dates:
        lb0 = dt - relativedelta(years=LOOKBACK_YEARS)
        lb1 = dt - relativedelta(months=1)
        for tkr, ser in returns.items():
            df.at[dt, tkr] = compute_ssa(ser.loc[lb0:lb1].values)
    return df


def sharpe_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0:
        return np.nan
    mean, std = arr.mean(), arr.std(ddof=1)
    return mean/std * np.sqrt(12) if std else np.nan


def sortino_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0:
        return np.nan
    neg = arr[arr<0]
    if neg.size == 0:
        return np.nan
    return arr.mean() / neg.std(ddof=1) * np.sqrt(12)


def calmar_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0:
        return np.nan
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = cum/peak - 1
    mdd = abs(dd.min())
    years = len(arr)/12
    if mdd == 0 or years == 0:
        return np.nan
    cagr = cum[-1]**(1/years) - 1
    return cagr / mdd

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
            'score':     score,
        })
    df = pd.DataFrame(rows).set_index('prev_rank')
    df['new_rank']   = df['score'].rank(ascending=False, method='first')
    df['rank_change']= df.index - df['new_rank']
    return df.sort_index()

# def gps_harmonic(scores: list[float]) -> float:
#     vals = [s for s in scores if s and not np.isnan(s)]
#     n = len(vals)
#     return n/np.sum([1/s for s in vals]) if n else -999

def gps_harmonic(scores: list[float]) -> float:
    # keep only strictly positive, non‑NaN entries
    vals = [s for s in scores if s > 0 and not np.isnan(s)]
    n = len(vals)
    if  n != 4:
        return -999
    return (n / sum(1.0/s for s in vals)) if n else -999

# -----------------------------------------------------------------------------
# 4) LOAD & PREP
# -----------------------------------------------------------------------------
ROOT_DIR     = Path().resolve().parent.parent / "Complete Data"
log_rets     = load_monthly_returns(ROOT_DIR / "All_Monthly_Log_Return_Data")
simple_rets  = load_monthly_returns(ROOT_DIR / "All_Monthly_Return_Data")
ssa_score    = build_ssa_history(log_rets)
tickers      = list(log_rets)
NUM_T        = len(tickers)

# -----------------------------------------------------------------------------
# 5) RANK LONG & SHORT EACH MONTH
# -----------------------------------------------------------------------------
long_rank, short_rank = {}, {}
cur = TEST_SIM_START
while cur <= SSA_END:
    block = ssa_score.loc[cur - relativedelta(years=ZCALIB_YEARS):cur - relativedelta(months=1)]
    mu, sd = block.mean(), block.std(ddof=1)
    raw = []
    for t in tickers:
        sc = ssa_score.at[cur, t]
        if pd.isna(sc) or sd[t] == 0:
            continue
        z = (sc - mu[t]) / sd[t]
        p = 2 * (1 - norm.cdf(abs(z)))
        raw.append((t, sc, z, p))
    df = pd.DataFrame(raw, columns=['tkr','ssa','z','p']).set_index('tkr')

    longs  = sorted(
        [r for r in raw if r[1]>0 and r[3]<=SIG_LEVEL],
        key=lambda x: x[1], reverse=True
    )
    shorts = sorted(
        [r for r in raw if r[1]<0 and r[3]<=SIG_LEVEL],
        key=lambda x: x[1]
    )
    restL  = sorted(
        [r for r in raw if r not in longs],
        key=lambda x: x[1], reverse=True
    )
    restS  = sorted(
        [r for r in raw if r not in shorts],
        key=lambda x: x[1]
    )

    ordL = [t for t,_,_,_ in longs] + [t for t,_,_,_ in restL] + [t for t in tickers if t not in df.index]
    ordS = [t for t,_,_,_ in shorts] + [t for t,_,_,_ in restS] + [t for t in tickers if t not in df.index]

    if DEBUG_MONTH and cur == DEBUG_MONTH:
        print(f"--- DEBUG {cur.date()} ---")
        print("Calibration block (SSA):", block)
        print(df)
        print("Long order:", ordL)
        print("Short order:", ordS)
        print("----------------------")

    long_rank[cur]  = ordL
    short_rank[cur] = ordS
    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# 6) CALIBRATION & HOLDOUT COMPUTATIONS
# -----------------------------------------------------------------------------
def compute_cum_and_rets(rankings, direction='long', start_date=None, end_date=None):
    holdings = {r:{} for r in range(1,NUM_T+1)}
    rets     = {r:[] for r in range(1,NUM_T+1)}
    for dt, order in rankings.items():
        if start_date and dt<start_date: continue
        if end_date   and dt>end_date:   continue
        for r, t in enumerate(order, start=1):
            holdings[r][dt] = t
            x = Decimal(str(simple_rets[t].get(dt,0.0)))
            if direction=='short':
                x = Decimal(1)/(Decimal(1)+x) - Decimal(1)
            rets[r].append(x)
    rows=[]
    START_D = Decimal(str(START_VALUE))
    for r in range(1,NUM_T+1):
        v = START_D
        for x in rets[r]:
            v *= (Decimal(1)+x)
        rows.append({'rank':float(r),'cum_ret':v/START_D - Decimal(1)})
    return pd.DataFrame(rows).set_index('rank'), holdings, rets

# Calibration/test period
calib_long_cum, _, calib_long_rets = compute_cum_and_rets(
    long_rank, 'long', TEST_SIM_START, TEST_SIM_END
)
metrics_long_calib = build_metrics(calib_long_cum, calib_long_rets)

calib_short_cum, _, calib_short_rets = compute_cum_and_rets(
    short_rank, 'short', TEST_SIM_START, TEST_SIM_END
)
metrics_short_calib = build_metrics(calib_short_cum, calib_short_rets)

# Holdout with original SSA ordering
holdout_long_cum, _, _  = compute_cum_and_rets(
    long_rank, 'long', FINAL_SIM_START, FINAL_SIM_END
)
holdout_short_cum, _, _ = compute_cum_and_rets(
    short_rank,'short', FINAL_SIM_START, FINAL_SIM_END
)

# -----------------------------------------------------------------------------
# 7) OUTPUT & COMPARISON
# -----------------------------------------------------------------------------
print(f"Testing Portfolios for Period {TEST_SIM_START} until {TEST_SIM_END}")

print("\nMetrics for LONG portfolios (ranks 1–17):")
print(metrics_long_calib.loc[1:17].to_string())
print("\nMetrics for SHORT portfolios (ranks 1–17):")
print(metrics_short_calib.loc[1:17].to_string())
print("-------------------------------------------------------------------------------------------------")

print(f"Final Portfolios for Period {FINAL_SIM_START} until {FINAL_SIM_END}")
print("\nMetrics for LONG portfolios (ranks 1–17):")
print(holdout_long_cum.loc[1:17].to_string())
print("\nMetrics for SHORT portfolios (ranks 1–17):")
print(holdout_short_cum.loc[1:17].to_string())

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

print("\nLong SSA comparison:")
print(output_comparison(
    metrics_long_calib,
    holdout_long_cum).to_string())

print("\nShort SSA comparison:")
print(output_comparison(
    metrics_short_calib,
    holdout_short_cum).to_string())
