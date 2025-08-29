# ZC — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **117,352**  
- Minimum monthly volume: **11,093**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std      skew  kurtosis_excess       min       q25    median      q75      max
   292 -0.004261 0.076276 -0.003438         0.736086 -0.256831 -0.044879 -0.007586 0.043924 0.227337


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.006801   -0.005297              0.012097  0.887776 0.381500    
02-Feb    25      267  0.010719   -0.005663              0.016382  1.330191 0.192554    
03-Mar    25      267 -0.010372   -0.003689             -0.006683 -0.501164 0.619751    
04-Apr    25      267  0.005611   -0.005185              0.010796  0.843767 0.405018    
05-May    24      268 -0.002735   -0.004398              0.001663  0.118852 0.906205    
06-Jun    24      268 -0.036933   -0.001335             -0.035598 -1.608248 0.120360    
07-Jul    24      268 -0.031586   -0.001814             -0.029772 -1.383400 0.178717    
08-Aug    24      268 -0.004233   -0.004263              0.000030  0.001967 0.998445    
09-Sep    24      268 -0.023617   -0.002528             -0.021089 -1.093718 0.284185    
10-Oct    24      268  0.024325   -0.006821              0.031146  1.747932 0.092110   *
11-Nov    24      268 -0.012741   -0.003501             -0.009240 -0.704347 0.486583    
12-Dec    24      268  0.022390   -0.006648              0.029037  2.260414 0.031041  **

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -10.709, p = 0.0000 ***, lags used = 1, n = 290  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.051, p = 0.1000 , lags used = 2  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.0146, F p = 0.0129  
- **White**: LM p = 0.0146, F p = 0.0129
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0009, F p = 0.0006  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.9296. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.30, p = 0.2152, break at index 174 (2015-07-01), left n = 174, right n = 118

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2012-07-01  2012      7  0.227337    3.583263  0.041667 0.044634                  True               False            True
2011-09-01  2011      9 -0.256831   -3.213558  0.041667 0.036210                  True               False            True
2008-10-01  2008     10 -0.207328   -3.191259  0.041667 0.035727                  True               False            True
2008-06-01  2008      6  0.188917    3.108518  0.041667 0.033960                  True               False            True
2012-06-01  2012      6  0.178510    2.960667  0.041667 0.030902                 False               False            True
2015-06-01  2015      6  0.169373    2.831405  0.041667 0.028336                 False               False            True
2008-07-01  2008      7 -0.221269   -2.597513  0.041667 0.023954                 False               False            True
2009-06-01  2009      6 -0.225116   -2.576466  0.041667 0.023577                 False               False            True
2021-04-01  2021      4  0.191348    2.539953  0.040000 0.021973                 False               False            True
2006-10-01  2006     10  0.197558    2.367485  0.041667 0.019979                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2011-09-01  2011      9 -0.256831 -3.830593             True
2012-07-01  2012      7  0.227337  3.610485             True
2009-06-01  2009      6 -0.225116 -3.343171            False
2008-07-01  2008      7 -0.221269 -3.284051            False
2006-10-01  2006     10  0.197558  3.152815            False
2008-10-01  2008     10 -0.207328 -3.069788            False
2021-04-01  2021      4  0.191348  3.057384            False
2008-06-01  2008      6  0.188917  3.020020            False
2012-06-01  2012      6  0.178510  2.860076            False
2004-07-01  2004      7 -0.189202 -2.791218            False
