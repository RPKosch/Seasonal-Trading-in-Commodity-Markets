import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
from decimal import Decimal, getcontext

# -----------------------------------------------------------------------------
# 1) PARAMETERS & WINDOWS
# -----------------------------------------------------------------------------
START_DATE      = datetime(2001, 1, 1)
LOOKBACK_YEARS  = 10   # only used to build SSA history
CALIB_YEARS     = 5
FINAL_END       = datetime(2024, 12, 31)
DEBUG_MONTH     = datetime(2023,7,1)  # e.g., datetime(2023,7,1) to activate detailed prints

SSA_WINDOW      = 12
SSA_COMPS       = 2
SIG_LEVEL       = 0.05

PLOT_START, PLOT_END = datetime(2016, 1, 1), datetime(2024, 12, 31)
START_VALUE     = 1000.0

# -----------------------------------------------------------------------------
# 2) DERIVE DATE RANGES
# -----------------------------------------------------------------------------
SSA_START = (START_DATE + relativedelta(years=LOOKBACK_YEARS)).replace(day=1)
SSA_END   = FINAL_END.replace(day=1)
INITIAL_SSA_END = (START_DATE + relativedelta(years=LOOKBACK_YEARS) - pd.offsets.MonthEnd(1))
FIRST_TRADE     = INITIAL_SSA_END + pd.offsets.MonthBegin(1) + relativedelta(years=CALIB_YEARS)

print(f"SSA history: {SSA_START.date()} → {SSA_END.date()}")
print(f"First trade : {FIRST_TRADE.date()}")

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
    mean, std = arr.mean(), arr.std(ddof=1)
    return (mean/std * np.sqrt(12)) if std else np.nan


def sortino_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    mean = arr.mean()
    neg = arr[arr<0]
    dstd = neg.std(ddof=1)
    return (mean/dstd * np.sqrt(12)) if dstd else np.nan


def calmar_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    drawdown = cum/peak - 1
    mdd = abs(drawdown.min())
    years = len(arr)/12
    cagr = cum[-1]**(1/years) - 1 if years>0 else np.nan
    return (cagr/mdd) if mdd else np.nan


def gps_harmonic(scores: list[float]) -> float:
    vals = [s for s in scores if s and not np.isnan(s)]
    n = len(vals)
    return (n / np.sum([1/s for s in vals])) if n else np.nan

# -----------------------------------------------------------------------------
# 4) LOAD & PREP
# -----------------------------------------------------------------------------
ROOT_DIR = Path().resolve().parent.parent / "Complete Data"
log_rets = load_monthly_returns(ROOT_DIR/"All_Monthly_Log_Return_Data")
simple_rets = load_monthly_returns(ROOT_DIR/"All_Monthly_Return_Data")
ssa_score = build_ssa_history(log_rets)
tickers = list(log_rets)
NUM_T = len(tickers)

# -----------------------------------------------------------------------------
# 5) RANK LONG & SHORT EACH MONTH (with optional DEBUG_MONTH)
# -----------------------------------------------------------------------------
long_rank, short_rank = {}, {}
cur = FIRST_TRADE
while cur <= SSA_END:
    block = ssa_score.loc[cur - relativedelta(years=CALIB_YEARS):cur - relativedelta(months=1)]
    mu, sd = block.mean(), block.std(ddof=1)
    raw = []
    for t in tickers:
        sc = ssa_score.at[cur, t]
        if pd.isna(sc) or sd[t] == 0:
            continue
        z = (sc - mu[t]) / sd[t]
        p = 2*(1 - norm.cdf(abs(z)))
        raw.append((t, sc, z, p))
    df = pd.DataFrame(raw, columns=['tkr','ssa','z','p']).set_index('tkr')

    # select & rank signals
    longs  = sorted([r for r in raw if r[1] > 0 and r[3] <= SIG_LEVEL], key=lambda x: x[1], reverse=True)
    shorts = sorted([r for r in raw if r[1] < 0 and r[3] <= SIG_LEVEL], key=lambda x: x[1])
    restL  = sorted([r for r in raw if r not in longs], key=lambda x: x[1], reverse=True)
    restS  = sorted([r for r in raw if r not in shorts], key=lambda x: x[1])

    ordL = [t for t,_,_,_ in longs] + [t for t,_,_,_ in restL] + [t for t in tickers if t not in df.index]
    ordS = [t for t,_,_,_ in shorts] + [t for t,_,_,_ in restS] + [t for t in tickers if t not in df.index]

    # Debug print if activated for this month
    if DEBUG_MONTH is not None and cur == DEBUG_MONTH:
        print(f"--- DEBUG {cur.date()} ---")
        print("SSA block (calibration window):", block)
        print("Signal DataFrame:", df)
        print("Long picks order:", ordL)
        print("Short picks order:", ordS)
        print("----------------------------")

    long_rank[cur]  = ordL
    short_rank[cur] = ordS
    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# 6) COMPUTE CUMULATIVE RETURNS & METRICS & METRICS
# -----------------------------------------------------------------------------
def compute_cum_and_rets(rankings, direction='long'):
    holdings, rets = ({r:{} for r in range(1,NUM_T+1)}, {r:[] for r in range(1,NUM_T+1)})
    for dt, order in rankings.items():
        if not (PLOT_START <= dt <= PLOT_END): continue
        for r, t in enumerate(order, start=1):
            holdings[r][dt] = t
            x = Decimal(str(simple_rets[t].get(dt,0.0)))
            if direction=='short': x = Decimal(1)/(Decimal(1)+x)-Decimal(1)
            rets[r].append(x)
    rows=[]
    START_D = Decimal(str(START_VALUE))
    for r in range(1,NUM_T+1):
        v = START_D
        for x in rets[r]: v *= (Decimal(1)+x)
        rows.append({'rank':float(r),'cum_ret':v/START_D-Decimal(1)})
    return pd.DataFrame(rows).set_index('rank'), holdings, rets

long_cum_df, long_hold, long_rets = compute_cum_and_rets(long_rank, 'long')
short_cum_df, short_hold, short_rets = compute_cum_and_rets(short_rank, 'short')

# -----------------------------------------------------------------------------
# 7) BUILD & PRINT RESULTS
# -----------------------------------------------------------------------------
def build_metrics(cum_df, rets_dict):
    rows=[]
    for pr, row in cum_df.iterrows():
        rets = rets_dict[pr]
        sr = sharpe_ratio(rets)
        sor = sortino_ratio(rets)
        cr = calmar_ratio(rets)
        cum = float(row['cum_ret'])
        sc = gps_harmonic([cum, sr, sor, cr])
        rows.append({'prev_rank':pr,'cum_ret':cum,'sharpe':sr,'sortino':sor,'calmar':cr,'score':sc})
    df = pd.DataFrame(rows).set_index('prev_rank')
    df['new_rank'] = df['score'].rank(ascending=False,method='first')
    df['rank_change'] = df.index - df['new_rank']
    return df.sort_index()

metrics_long = build_metrics(long_cum_df, long_rets)
metrics_short= build_metrics(short_cum_df, short_rets)

print("\nCumulative returns for LONG portfolios (ranks 1–17):")
for rank in range(1,18):
    print(f" Rank {rank:2d}: {long_cum_df.loc[rank,'cum_ret']:.2%}")
print("\nCumulative returns for SHORT portfolios (ranks 1–17):")
for rank in range(1,18):
    print(f" Rank {rank:2d}: {short_cum_df.loc[rank,'cum_ret']:.2%}")

print("\nMetrics for LONG (1–17):\n", metrics_long.loc[1:17].to_string())
print("\nMetrics for SHORT (1–17):\n", metrics_short.loc[1:17].to_string())
