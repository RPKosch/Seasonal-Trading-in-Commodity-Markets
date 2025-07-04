import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_ind

# 1) Load data
df = pd.read_csv("crude_monthly_close.csv", parse_dates=["Date"], index_col="Date")
price = df["CL=F"].sort_index()

# 2) Log returns RET
RET = np.log(price / price.shift(1)).dropna().rename("RET")
print(f"Log‑returns from {RET.index[0].date()} to {RET.index[-1].date()}, {len(RET)} points")

# 3) Build lagged returns for SEA
lags = [12 * k for k in range(2, 11)]
lagged = pd.concat({f"lag{lag}": RET.shift(lag) for lag in lags}, axis=1)

# 4) SEA_t as mean of lagged returns
SEA = lagged.mean(axis=1).dropna().rename("SEA")
print(f"SEA from {SEA.index[0].date()} to {SEA.index[-1].date()}, {len(SEA)} points")

# 5) Rolling regression for rho_t with 60-month window and manual t-stat
window = 60
rho = pd.Series(index=SEA.index[window:], dtype=float)
rho_tstat = pd.Series(index=SEA.index[window:], dtype=float)

for end in SEA.index[window:]:
    y = RET.loc[:end].iloc[-window:]
    x = SEA.loc[:end].iloc[-window:]
    # Fit slope b and intercept a
    x_bar, y_bar = x.mean(), y.mean()
    Sxx = ((x - x_bar)**2).sum()
    Sxy = ((x - x_bar)*(y - y_bar)).sum()
    b = Sxy / Sxx
    a = y_bar - b * x_bar
    # Residuals and sigma^2
    resid = y - (a + b * x)
    sigma2 = (resid**2).sum() / (window - 2)
    # Standard error of b
    se_b = np.sqrt(sigma2 / Sxx)
    # t-statistic
    t_b = b / se_b

    rho.loc[end] = b
    rho_tstat.loc[end] = t_b

rho.dropna(inplace=True)
rho_tstat.dropna(inplace=True)
print(f"Estimated rho on {len(rho)} dates")

# 6) Test average rho_t significance
tstat_mean, pval_mean = ttest_1samp(rho_tstat, 0)
print(f"Average rho t-stat = {tstat_mean:.2f}, p-value = {pval_mean:.4f}")

# 7) Align and compute expected returns E_t = rho_t * SEA_t
common = SEA.index.intersection(rho.index)
E = rho.loc[common] * SEA.loc[common]

# 8) Month-of-year tests: compare each month vs others
month_tests = []
for m in range(1, 13):
    m_rets = RET[RET.index.month == m]
    other_rets = RET[RET.index.month != m]
    tstat, pval = ttest_ind(m_rets, other_rets, equal_var=False)
    month_tests.append((m, tstat, pval))

month_df = pd.DataFrame(month_tests, columns=["Month", "t-stat", "p-value"]).set_index("Month")
print("\nMonth-of-year return difference tests:")
print(month_df.round(3))

# 9) Plot month-of-year t-statistics
plt.figure(figsize=(8, 4))
plt.bar(month_df.index, month_df["t-stat"], color='coral')
plt.axhline(0, color='gray', linestyle='--')
plt.xticks(range(1, 13))
plt.xlabel("Month")
plt.ylabel("t-statistic")
plt.title("Month-of-Year Return Difference t-Statistics")
plt.tight_layout()
plt.show()
