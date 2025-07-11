import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# ----------------------------
# User-adjustable parameters
# ----------------------------
TICKER = "CO"                  # Ticker to analyze
start_date = "2000-01-01"      # Analysis start date (inclusive)
end_date = "2026-12-31"        # Analysis end date (inclusive)

# ----------------------------
# Load and preprocess data
# ----------------------------
project_root = Path().resolve().parent.parent

data_path = project_root / "Complete Data" / "All_Monthly_Return_Data" / f"{TICKER}_Monthly_Revenues.csv"
df = pd.read_csv(data_path)

# Ensure proper types
df['return'] = pd.to_numeric(df['return'], errors='coerce')
df = df.dropna(subset=['return'])
df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
df = df.sort_values('date').reset_index(drop=True)

# Filter by time window
df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

# Extract month from date
df['month'] = df['date'].dt.month

# ----------------------------
# Month-by-month dummy regressions
# ----------------------------
results = []

for m in range(1, 13):
    # Create dummy: 1 if current month == m, else 0
    df['D'] = (df['month'] == m).astype(float)

    # Regression: return ~ 1 + D
    X = sm.add_constant(df['D'])
    y = df['return']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

    beta = model.params['D']
    t_stat = model.tvalues['D']
    p_val = model.pvalues['D']
    ci_low, ci_high = model.conf_int().loc['D']

    # Average return in month m
    avg_m = df.loc[df['month'] == m, 'return'].mean()
    # Average return in other months
    avg_other = df.loc[df['month'] != m, 'return'].mean()

    results.append({
        'month': m,
        'avg_return_month': avg_m,
        'avg_return_other': avg_other,
        'coef_diff': beta,
        't_stat': t_stat,
        'p_value': p_val,
        'ci_lower': ci_low,
        'ci_upper': ci_high
    })

# Compile results
res_df = pd.DataFrame(results)
res_df['month_name'] = res_df['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
res_df = res_df[['month', 'month_name', 'avg_return_month', 'avg_return_other',
                 'coef_diff', 't_stat', 'p_value', 'ci_lower', 'ci_upper']]

# Display
print(f"Month-by-Month Seasonality Tests for Ticker: {TICKER}")
print(f"Time window: {start_date} to {end_date}\n")
print(res_df.to_string(index=False))
