import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from decimal import Decimal

# ---------------------------------------------------------------------
# 1) PARAMETERS & WINDOWS
# ---------------------------------------------------------------------
START_DATE              = datetime(2001, 1, 1)
FINAL_END               = datetime(2024, 12, 31)
LOOKBACK_YEARS          = 10
TEST_YEARS              = 5        # testing/calibration window
SIM_YEARS               = 5        # simulation window (for illustrative split)
SIG_LEVEL               = 0.05     # DVR significance level for the D dummy
START_VALUE             = 1000.0
ENTRY_COST              = 0.0025   # monthly entry cost (apply only in holdout)
DEBUG_DATE              = None     # e.g., datetime(2011, 1, 1)

# ---------------------------------------------------------------------
# 2) DATE RANGES
# ---------------------------------------------------------------------
LOOKBACK_END = (START_DATE + relativedelta(years=LOOKBACK_YEARS) - pd.offsets.MonthEnd(1))
TEST_SIM_START   = START_DATE + relativedelta(years=LOOKBACK_YEARS)
TEST_SIM_END     = TEST_SIM_START + relativedelta(years=SIM_YEARS) - pd.offsets.MonthEnd(1)
FINAL_SIM_START  = START_DATE + relativedelta(years=LOOKBACK_YEARS) + relativedelta(years=SIM_YEARS)
FINAL_SIM_END    = FINAL_END

print(f"Lookback: {START_DATE.date()} → {LOOKBACK_END.date()}")
print(f"Testing:  {TEST_SIM_START.date()} → {TEST_SIM_END.date()}")
print(f"Holdout:  {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")

# ---------------------------------------------------------------------
# 3) HELPERS
# ---------------------------------------------------------------------
def newey_west_lags(T: int) -> int:
    """Rule of thumb: L = floor(0.75 * T^(1/3)), at least 1."""
    return max(1, int(np.floor(0.75 * (T ** (1/3)))))


def load_returns(root_dir: Path) -> dict[str, pd.Series]:
    """Load *_Monthly_Revenues.csv files, return dict[ticker] -> monthly simple returns Series."""
    out = {}
    for f in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        ticker = f.stem.replace("_Monthly_Revenues", "")
        df = (
            pd.read_csv(f)
              .assign(
                  date=lambda d: pd.to_datetime(d[['year','month']].assign(day=1)),
                  rtn=lambda d: pd.to_numeric(d['return'], errors='coerce')
              )
              .set_index('date')['rtn']
              .sort_index()
        )
        out[ticker] = df
    return out


def sharpe_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0:
        return np.nan
    std = arr.std(ddof=1)
    if std == 0:
        return np.nan
    return arr.mean() / std * np.sqrt(12)


def sortino_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0:
        return np.nan
    neg = arr[arr < 0]
    if neg.size == 0:
        return np.nan
    return arr.mean() / neg.std(ddof=1) * np.sqrt(12)


def calmar_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0:
        return np.nan
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    mdd = abs(dd.min())
    years = len(arr) / 12
    if mdd == 0 or years == 0:
        return np.nan
    cagr = cum[-1] ** (1 / years) - 1
    return cagr / mdd


def gps_harmonic(scores: list[float]) -> float:
    """Simple GPS-like harmonic aggregation over [cum, Sharpe, Sortino, Calmar].
       Requires all 4 strictly positive and non-NaN; else returns sentinel -999."""
    vals = [s for s in scores if (s is not None) and (not np.isnan(s)) and (s > 0)]
    if len(vals) != 4:
        return -999
    return len(vals) / sum(1.0 / s for s in vals)


def build_metrics(cum_df: pd.DataFrame, rets_dict: dict[int, list[Decimal]]) -> pd.DataFrame:
    """Build metrics per rank using NO-COST monthly returns for SR/Sortino/Calmar.
       cum_df contains 'cum_ret' (no-cost) and 'cum_ret_wc' (with-cost)."""
    rows = []
    for prev_rank, cum_row in cum_df.iterrows():
        rets = rets_dict.get(prev_rank, [])
        sr  = sharpe_ratio(rets)
        sor = sortino_ratio(rets)
        cr  = calmar_ratio(rets)
        cum = float(cum_row['cum_ret'])
        score = gps_harmonic([cum, sr, sor, cr])
        rows.append({
            'prev_rank': prev_rank,
            'cum_ret':   cum,                          # no-cost cumulative
            'cum_ret_wc': float(cum_row['cum_ret_wc']),# with-cost cumulative
            'sharpe':    sr,
            'sortino':   sor,
            'calmar':    cr,
            'score':     score
        })
    df = pd.DataFrame(rows).set_index('prev_rank')
    df['new_rank']    = df['score'].rank(ascending=False, method='first')
    df['rank_change'] = df.index - df['new_rank']
    return df.sort_index()


def compute_cum(rankings: dict[pd.Timestamp, list[str]],
                direction: str = 'long',
                start_date: pd.Timestamp | None = None,
                end_date: pd.Timestamp | None = None,
                entry_cost: float = 0.0):
    """
    Compounds monthly returns per previous DVR rank.
    - Uses simple_rets for monthly base returns.
    - Exact short: r_short = 1/(1+r_long) - 1.
    - Entry cost applied monthly ONLY if entry_cost > 0: value *= (1 - entry_cost) * (1 + r).
    Returns:
      cum_df: DataFrame with columns ['cum_ret', 'cum_ret_wc'] indexed by rank
      holdings: dict[rank] -> {date: ticker}
      rets_nc: dict[rank] -> list[Decimal] of NO-COST monthly returns
    """
    holdings: dict[int, dict[pd.Timestamp, str]] = {}
    rets_nc:  dict[int, list[Decimal]] = {}
    rets_wc:  dict[int, list[Decimal]] = {}

    for r in range(1, NUM_T + 1):
        holdings[r] = {}
        rets_nc[r]  = []
        rets_wc[r]  = []

    for dt, order in rankings.items():
        if start_date and dt < start_date:
            continue
        if end_date and dt > end_date:
            continue

        for r, t in enumerate(order, start=1):
            holdings[r][dt] = t

            # base monthly simple return
            raw = Decimal(str(simple_rets[t].get(dt, 0.0)))
            # exact futures short transformation
            if direction == 'short':
                raw = Decimal(1) / (Decimal(1) + raw) - Decimal(1)

            # store no-cost return
            rets_nc[r].append(raw)

            # with-cost path: apply only if entry_cost > 0, else identical to no-cost
            if entry_cost > 0:
                r_wc = (Decimal(1) - Decimal(str(entry_cost))) * (Decimal(1) + raw) - Decimal(1)
            else:
                r_wc = raw
            rets_wc[r].append(r_wc)

    # compound to cumulative returns
    rows = []
    START_D = Decimal(str(START_VALUE))
    for r in range(1, NUM_T + 1):
        vc_nc = START_D
        vc_wc = START_D
        for x_nc, x_wc in zip(rets_nc[r], rets_wc[r]):
            vc_nc *= (Decimal(1) + x_nc)
            vc_wc *= (Decimal(1) + x_wc)
        rows.append({
            'rank':       float(r),
            'cum_ret':    vc_nc / START_D - Decimal(1),   # no-cost cumulative return
            'cum_ret_wc': vc_wc / START_D - Decimal(1)    # with-cost cumulative return
        })
    cum_df = pd.DataFrame(rows).set_index('rank')
    return cum_df, holdings, rets_nc   # keep no-cost rets for risk metrics


# ---------------------------------------------------------------------
# 4) LOAD DATA
# ---------------------------------------------------------------------
base        = Path().resolve().parent.parent / "Complete Data"
# log_rets are not used in compounding; simple_rets feed P&L math (and shorts)
log_rets    = load_returns(base / "All_Monthly_Log_Return_Data")
simple_rets = load_returns(base / "All_Monthly_Return_Data")
tickers     = list(log_rets)
NUM_T       = len(tickers)

# ---------------------------------------------------------------------
# 5) ROLLING DVR REGRESSIONS (with Newey–West) & RANKINGS
# ---------------------------------------------------------------------
long_rankings:  dict[pd.Timestamp, list[str]] = {}
short_rankings: dict[pd.Timestamp, list[str]] = {}

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
                        D=lambda d: (d.month == m).astype(float)))
        X = sm.add_constant(df['D'])
        L = newey_west_lags(len(df))
        res = sm.OLS(df['rtn'], X).fit(cov_type='HAC', cov_kwds={'maxlags': L})

        stats.append({
            'ticker': t,
            'beta':   res.params['D'],
            'pval':   res.pvalues['D'],
            'avg':    df.loc[df.month == m, 'rtn'].mean()
        })

    if not stats:
        # fallback: keep original order if no stats (should be rare)
        long_rankings[cur]   = tickers.copy()
        short_rankings[cur]  = tickers.copy()
    else:
        dfm = pd.DataFrame(stats).set_index('ticker')

        # LONG DVR: significant, positive beta, positive avg
        sigL  = dfm[(dfm.pval <= SIG_LEVEL) & (dfm.beta > 0) & (dfm.avg > 0)]
        restL = dfm.drop(sigL.index, errors='ignore')
        orderL = list(sigL.sort_values('avg', ascending=False).index) + \
                 list(restL.sort_values('avg', ascending=False).index)
        # pad to full set if needed
        orderL += [t for t in tickers if t not in orderL]
        long_rankings[cur] = orderL

        # SHORT DVR: significant, negative beta, negative avg
        sigS  = dfm[(dfm.pval <= SIG_LEVEL) & (dfm.beta < 0) & (dfm.avg < 0)]
        restS = dfm.drop(sigS.index, errors='ignore')
        orderS = list(sigS.sort_values('avg', ascending=True).index) + \
                 list(restS.sort_values('avg', ascending=True).index)
        orderS += [t for t in tickers if t not in orderS]
        short_rankings[cur] = orderS

    if DEBUG_DATE and cur == DEBUG_DATE:
        print(f"--- DEBUG at {cur.date()} ---")
        dbg = pd.DataFrame(stats).set_index('ticker')[['beta', 'pval', 'avg']]
        print(dbg)
        print("Long order:", long_rankings[cur])
        print("Short order:", short_rankings[cur])
        print("-" * 100)

    cur += relativedelta(months=1)

# ---------------------------------------------------------------------
# 6) TEST & HOLDOUT COMPUTATIONS (LONG & SHORT)
# ---------------------------------------------------------------------
# Testing period: NO entry cost
long_test_cum_df, _, long_test_rets = compute_cum(
    long_rankings, 'long', TEST_SIM_START, TEST_SIM_END, entry_cost=0.0
)
metrics_long_test = build_metrics(long_test_cum_df, long_test_rets)

short_test_cum_df, _, short_test_rets = compute_cum(
    short_rankings, 'short', TEST_SIM_START, TEST_SIM_END, entry_cost=0.0
)
metrics_short_test = build_metrics(short_test_cum_df, short_test_rets)

# Holdout (final) period: APPLY entry cost
long_holdout_cum_df, _, _  = compute_cum(
    long_rankings, 'long', FINAL_SIM_START, FINAL_SIM_END, entry_cost=ENTRY_COST
)
short_holdout_cum_df, _, _ = compute_cum(
    short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END, entry_cost=ENTRY_COST
)

# ---------------------------------------------------------------------
# 7) OUTPUT & COMPARISONS
# ---------------------------------------------------------------------
print(f"\nTesting Period: {TEST_SIM_START.date()} → {TEST_SIM_END.date()}")
print("\nMetrics for LONG portfolios (ranks 1–17):")
print(metrics_long_test.loc[1:17].to_string(index=True))
print("\nMetrics for SHORT portfolios (ranks 1–17):")
print(metrics_short_test.loc[1:17].to_string(index=True))
print("-" * 100)

print(f"Final (Holdout) Period: {FINAL_SIM_START.date()} → {FINAL_END.date()}")
print("\nLONG cumulative returns (no-cost & with-cost) for ranks 1–17:")
print(long_holdout_cum_df.loc[1:17].to_string(index=True))
print("\nSHORT cumulative returns (no-cost & with-cost) for ranks 1–17:")
print(short_holdout_cum_df.loc[1:17].to_string(index=True))


def output_comparison(metrics_calib: pd.DataFrame, orig_cum: pd.DataFrame) -> pd.DataFrame:
    """
    Compare: for each previous rank, show holdout return at that same previous rank
    vs. the holdout return at the rank implied by the new_score ordering (new_rank).
    Uses NO-COST cumulative returns for comparability.
    """
    print(f"\nTesting Period: {TEST_SIM_START.date()} → {TEST_SIM_END.date()}")
    print(f"Final Simulation: {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")
    rows = []
    for prev_rank, row in metrics_calib.iterrows():
        pr = int(prev_rank)
        nr = int(row['new_rank'])
        new_ret  = float(orig_cum.loc[pr, 'cum_ret'])
        orig_ret = float(orig_cum.loc[nr, 'cum_ret'])
        diff = new_ret - orig_ret
        rows.append({
            'prev_rank': pr,
            'new_rank':  nr,
            'holdout_ret_at_prev_rank': new_ret,
            'holdout_ret_at_new_rank':  orig_ret,
            'difference':               diff,
            'score':                    float(row['score'])
        })
    return pd.DataFrame(rows).set_index('new_rank').sort_index()


print("\nLong comparison (orig vs new holdout returns):")
long_comp = output_comparison(metrics_long_test, long_holdout_cum_df)
print(long_comp.to_string())

print("\nShort comparison (orig vs new holdout returns):")
short_comp = output_comparison(metrics_short_test, short_holdout_cum_df)
print(short_comp.to_string())
