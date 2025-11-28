# GD — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **86,899**  
- Minimum monthly volume: **1,084**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min       q25   median      q75      max
   292 0.007231 0.046428 -0.091513         0.539997 -0.180978 -0.023067 0.004767 0.036255 0.128779


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.024640    0.005601              0.019039  1.943256 0.061913   *
02-Feb    25      267  0.005242    0.007417             -0.002175 -0.231541 0.818514    
03-Mar    25      267  0.003779    0.007554             -0.003775 -0.446700 0.658229    
04-Apr    25      267  0.011119    0.006867              0.004252  0.433457 0.667937    
05-May    24      268  0.002051    0.007695             -0.005644 -0.596800 0.555469    
06-Jun    24      268 -0.006727    0.008481             -0.015208 -1.472294 0.152577    
07-Jul    24      268  0.007651    0.007193              0.000457  0.054249 0.957103    
08-Aug    24      268  0.016575    0.006394              0.010181  1.061118 0.297827    
09-Sep    24      268  0.002855    0.007623             -0.004768 -0.424268 0.674842    
10-Oct    24      268  0.000376    0.007845             -0.007469 -0.708822 0.484584    
11-Nov    24      268  0.007269    0.007228              0.000042  0.003720 0.997060    
12-Dec    24      268  0.011281    0.006868              0.004412  0.442360 0.661725    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -18.750, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.165, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.9723, F p = 0.9743  
- **White**: LM p = 0.9723, F p = 0.9743
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0617, F p = 0.0588  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** **Mixed/ambiguous** evidence (some tests borderline). Prefer **robust/HAC standard errors** to be safe.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.2524. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 2.77, p = 0.0014, break at index 122 (2011-03-01), left n = 122, right n = 170

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Break suggested at the reported date; parameter shifts before/after that point.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2008-10-01  2008     10 -0.180978   -4.080038  0.041667 0.057122                  True               False            True
2011-12-01  2011     12 -0.105963   -2.593052  0.041667 0.023874                 False               False            True
2008-11-01  2008     11  0.121918    2.534321  0.041667 0.022829                 False               False            True
2011-09-01  2011      9 -0.111653   -2.531132  0.041667 0.022773                 False               False            True
2009-11-01  2009     11  0.120133    2.493973  0.041667 0.022123                 False               False            True
2011-08-01  2011      8  0.128779    2.479085  0.041667 0.021866                 False               False            True
2013-06-01  2013      6 -0.118880   -2.477916  0.041667 0.021846                 False               False            True
2008-08-01  2008      8 -0.094257   -2.448125  0.041667 0.021334                 False               False            True
2004-04-01  2004      4 -0.093567   -2.307654  0.040000 0.018209                 False               False            True
2009-05-01  2009      5  0.103072    2.227358  0.041667 0.017724                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2008-10-01  2008     10 -0.180978 -4.231657             True
2011-08-01  2011      8  0.128779  2.825247            False
2013-06-01  2013      6 -0.118880 -2.816921            False
2008-11-01  2008     11  0.121918  2.668925            False
2011-09-01  2011      9 -0.111653 -2.652276            False
2009-11-01  2009     11  0.120133  2.628255            False
2011-12-01  2011     12 -0.105963 -2.522653            False
2006-04-01  2006      4  0.113474  2.476557            False
2012-01-01  2012      1  0.106351  2.314271            False
2016-02-01  2016      2  0.104114  2.263329            False
