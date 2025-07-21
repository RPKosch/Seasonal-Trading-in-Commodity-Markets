import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from dateutil.relativedelta import relativedelta

# ----------------------------
# User‑adjustable parameters
# ----------------------------
TICKER          = "CO"                   # Ticker to analyze
start_date      = "2001-01-01"           # Analysis start date (inclusive)
end_date        = ("2016-12-31")           # Analysis end date (inclusive)
LOOKBACK_YEARS  = 10                      # e.g. 5 for last 5 years only, or None for full history

# ----------------------------
# Load and preprocess data
# ----------------------------
project_root = Path().resolve().parent.parent
data_path    = project_root / "Complete Data" / "All_Monthly_Return_Data" / f"{TICKER}_Monthly_Revenues.csv"

df = pd.read_csv(data_path)
df['return'] = pd.to_numeric(df['return'], errors='coerce')
df = df.dropna(subset=['return'])
df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
df = df.sort_values('date').reset_index(drop=True)

# Filter by absolute time window
df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

# Optionally trim to lookback window ending at end_date
if LOOKBACK_YEARS is not None:
    cutoff = pd.to_datetime(end_date) - relativedelta(years=LOOKBACK_YEARS)
    df = df[df['date'] >= cutoff]

# Extract month
df['month'] = df['date'].dt.month

# ----------------------------
# Month‑by‑month dummy regressions
# ----------------------------
results = []

for m in range(1, 13):
    # dummy for month m
    df['D'] = (df['month'] == m).astype(float)

    # regress return ~ 1 + D
    X     = sm.add_constant(df['D'])
    y     = df['return']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

    beta    = model.params['D']
    t_stat  = model.tvalues['D']
    p_val   = model.pvalues['D']
    ci_low, ci_high = model.conf_int().loc['D']

    avg_m     = df.loc[df['month'] == m, 'return'].mean()
    avg_other = df.loc[df['month'] != m, 'return'].mean()

    results.append({
        'month':             m,
        'avg_return_month':  avg_m,
        'avg_return_other':  avg_other,
        'coef_diff':         beta,
        't_stat':            t_stat,
        'p_value':           p_val,
        'ci_lower':          ci_low,
        'ci_upper':          ci_high
    })

# Compile and display
res_df = pd.DataFrame(results)
res_df['month_name'] = res_df['month'].apply(
    lambda x: pd.to_datetime(str(x), format='%m').strftime('%B')
)
res_df = res_df[[
    'month', 'month_name', 'avg_return_month', 'avg_return_other',
    'coef_diff', 't_stat', 'p_value', 'ci_lower', 'ci_upper'
]]

print(f"Month‑by‑Month Seasonality Tests for Ticker: {TICKER}")
print(f"Using data from {df['date'].min().date()} to {df['date'].max().date()}")
if LOOKBACK_YEARS is not None:
    print(f"  (lookback = last {LOOKBACK_YEARS} years)\n")
else:
    print()

print(res_df.to_string(index=False))
