import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from dateutil.relativedelta import relativedelta

# ----------------------------
# User-adjustable parameters
# ----------------------------
start_date      = "2001-01-01"   # Earliest data to consider (inclusive)
end_date        = "2016-12-20"   # Analysis end date (inclusive)
LOOKBACK_YEARS  = 10              # e.g. 5 to use only last 5 years, or None for full history

# ----------------------------
# Locate all tickers
# ----------------------------
project_root       = Path().resolve().parent.parent
data_dir           = project_root / "Complete Data" / "All_Monthly_Return_Data"
files              = list(data_dir.glob("*_Monthly_Revenues.csv"))

all_results         = []
selection_candidates = []

# Determine the "next month" after end_date
end_dt     = pd.to_datetime(end_date)
next_month = (end_dt + pd.DateOffset(months=1)).month

for file_path in files:
    ticker = file_path.stem.replace("_Monthly_Revenues", "")
    df     = pd.read_csv(file_path)

    # Prepare DataFrame
    df['return'] = pd.to_numeric(df['return'], errors='coerce')
    df           = df.dropna(subset=['return'])
    df['date']   = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df           = df.sort_values('date').reset_index(drop=True)

    # Filter absolute window
    df = df[(df['date'] >= pd.to_datetime(start_date)) &
            (df['date'] <= end_dt)]

    # Apply lookback window if requested
    if LOOKBACK_YEARS is not None:
        cutoff = end_dt - relativedelta(years=LOOKBACK_YEARS)
        df     = df[df['date'] >= cutoff]

    df['month'] = df['date'].dt.month

    # Monthly dummy regression for all months
    results = []
    for m in range(1, 13):
        df['D'] = (df['month'] == m).astype(float)
        X       = sm.add_constant(df['D'])
        y       = df['return']
        model   = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

        beta    = model.params['D']
        pval    = model.pvalues['D']
        tstat   = model.tvalues['D']
        ci_low, ci_high = model.conf_int().loc['D']

        avg_m     = df.loc[df['month'] == m, 'return'].mean()
        avg_other = df.loc[df['month'] != m, 'return'].mean()

        results.append({
            'ticker':           ticker,
            'month':            m,
            'avg_return_month': avg_m,
            'avg_return_other': avg_other,
            'coef_diff':        beta,
            't_stat':           tstat,
            'p_value':          pval,
            'ci_lower':         ci_low,
            'ci_upper':         ci_high
        })

    all_results.append(pd.DataFrame(results))

    # Find candidate for next_month (require p ≤ .05 and positive avg return)
    candidate = [
        r for r in results
        if r['month'] == next_month
        and r['p_value'] <= 0.05
        and r['avg_return_month'] > 0
    ]
    if candidate:
        # pick the one with the largest |coef_diff|
        best   = max(candidate, key=lambda x: abs(x['coef_diff']))
        signal = best['ticker'] if best['coef_diff'] > 0 else f"-{best['ticker']}"
        selection_candidates.append((signal, best['coef_diff']))

# Print month-by-month seasonality for each ticker
for df_res in all_results:
    print(df_res.to_string(index=False))
    print("\n" + "-" * 60 + "\n")

# Selection for next month
if selection_candidates:
    # pick the signal with highest absolute coefficient
    best_signal = max(selection_candidates, key=lambda x: abs(x[1]))[0]
    print(f"Selected Contract for month {next_month}: {best_signal}")
else:
    print("NoContract")
