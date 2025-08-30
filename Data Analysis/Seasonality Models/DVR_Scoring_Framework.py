import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from dateutil.relativedelta import relativedelta

# ----------------------------
# User settings
# ----------------------------
end_date       = "2020-5-20"
LOOKBACK_YEARS = 10
SIGN_LEVEL     = 1

# ----------------------------
# 1) Load existing returns
# ----------------------------
def load_monthly_returns(root_dir: Path) -> dict[str, pd.Series]:
    out = {}
    for path in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        ticker = path.stem.replace("_Monthly_Revenues", "")
        df = pd.read_csv(path)
        df['date']   = pd.to_datetime(df[['year','month']].assign(day=1))
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        out[ticker]  = df.set_index('date')['return'].sort_index()
    return out

base    = Path().resolve().parent.parent / "Complete Data"
returns = load_monthly_returns(base / "All_Monthly_Log_Return_Data")

# ----------------------------
# 2) Prep dates
# ----------------------------
end_dt     = pd.to_datetime(end_date)
start_lb   = end_dt - relativedelta(years=LOOKBACK_YEARS)
next_month = (end_dt + pd.DateOffset(months=1)).month

# ----------------------------
# 3) Build overview
# ----------------------------
rows = []
for tkr, series in returns.items():
    # restrict to lookback window only
    s = series[(series.index > start_lb) & (series.index <= end_dt)]
    print(s)
    if len(s) < 12:
        continue

    # basic stats
    obs_count   = len(s)
    first_date  = s.index.min().date()
    last_date   = s.index.max().date()
    mean_return = s[s.index.month == next_month].mean()
    std_return  = s.std(ddof=1)

    # dummy regression for next_month
    df = s.to_frame('return')
    df['month'] = df.index.month
    df['D']     = (df['month'] == next_month).astype(float)

    X     = sm.add_constant(df['D'])
    y     = df['return']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    beta  = model.params['D']
    pval  = model.pvalues['D']

    rows.append({
        'ticker':       tkr,
        'obs_count':    obs_count,
        'first_date':   first_date,
        'last_date':    last_date,
        'mean_return':  mean_return,
        'std_return':   std_return,
        'coef_diff':    beta,
        'p_value':      pval
    })

overview = pd.DataFrame(rows).set_index('ticker').sort_index()
print(overview.to_string())


# ----------------------------
# 4) Final selection
# ----------------------------
# Long candidates: significant, direction‑positive
longs = overview[
    (overview['p_value'] <= SIGN_LEVEL) &
    (overview['coef_diff'] > 0) &
    (overview['mean_return'] > 0)
]
if not longs.empty:
    best_long = longs['coef_diff'].idxmax()
else:
    best_long = "No Ticker"

# Short candidates: significant, direction‑negative
shorts = overview[
    (overview['p_value'] <= SIGN_LEVEL) &
    (overview['coef_diff'] < 0) &
    (overview['mean_return'] < 0)
]
if not shorts.empty:
    best_short = shorts['coef_diff'].idxmin()
    best_short = f"-{best_short}"
else:
    best_short = "No Ticker"

print(f"\nBest long position for month {next_month}: {best_long}")
print(f"Best short position for month {next_month}: {best_short}")
