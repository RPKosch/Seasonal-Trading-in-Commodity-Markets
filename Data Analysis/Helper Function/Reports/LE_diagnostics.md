# LE — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **18,686**  
- Minimum monthly volume: **3,970**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std      skew  kurtosis_excess      min       q25   median      q75     max
   292 -0.000495 0.042244 -0.828136         2.840031 -0.23336 -0.023247 0.002574 0.026764 0.11208


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.003741   -0.000191             -0.003550 -0.500806 0.619910    
02-Feb    25      267 -0.000646   -0.000481             -0.000165 -0.021044 0.983348    
03-Mar    25      267 -0.010948    0.000484             -0.011432 -1.008137 0.322505    
04-Apr    25      267 -0.005148   -0.000059             -0.005089 -0.576608 0.568698    
05-May    24      268  0.003151   -0.000821              0.003972  0.433882 0.667806    
06-Jun    24      268  0.011632   -0.001581              0.013213  1.631668 0.113733    
07-Jul    24      268  0.009902   -0.001426              0.011328  1.555735 0.130160    
08-Aug    24      268 -0.015534    0.000852             -0.016386 -2.224804 0.033757  **
09-Sep    24      268  0.000827   -0.000613              0.001440  0.133718 0.894664    
10-Oct    24      268  0.006791   -0.001147              0.007938  0.895335 0.378392    
11-Nov    24      268  0.003084   -0.000816              0.003899  0.454678 0.652866    
12-Dec    24      268 -0.004538   -0.000133             -0.004405 -0.400602 0.692034    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -16.574, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.077, p = 0.1000 , lags used = 5  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.7067, F p = 0.7155  
- **White**: LM p = 0.7067, F p = 0.7155
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity** (residual variance depends on months). If detected, report heteroskedasticity-robust standard errors (e.g., HC3) when presenting any coefficients.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic error assumption appears reasonable.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.8907. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.54, p = 0.1092, break at index 43 (2004-08-01), left n = 43, right n = 249

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2003-12-01  2003     12 -0.233360   -5.838082  0.041667 0.110441                  True               False            True
2017-04-01  2017      4  0.112080    2.862084  0.040000 0.027731                 False               False            True
2018-03-01  2018      3 -0.120838   -2.678174  0.040000 0.024368                 False               False            True
2020-02-01  2020      2 -0.106591   -2.579699  0.040000 0.022650                 False               False            True
2015-09-01  2015      9 -0.101937   -2.502672  0.041667 0.022275                 False               False            True
2001-09-01  2001      9 -0.099252   -2.435881  0.041667 0.021126                 False               False            True
2011-05-01  2011      5 -0.094555   -2.376943  0.041667 0.020136                 False               False            True
2006-03-01  2006      3 -0.108248   -2.364772  0.040000 0.019104                 False               False            True
2020-03-01  2020      3 -0.102369   -2.219291  0.040000 0.016865                 False               False            True
2008-10-01  2008     10 -0.081784   -2.150917  0.041667 0.016548                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2003-12-01  2003     12 -0.233360 -6.260796             True
2018-03-01  2018      3 -0.120838 -3.274905            False
2006-03-01  2006      3 -0.108248 -2.940804            False
2017-04-01  2017      4  0.112080  2.905862            False
2020-02-01  2020      2 -0.106591 -2.896844            False
2020-03-01  2020      3 -0.102369 -2.784805            False
2015-09-01  2015      9 -0.101937 -2.773324            False
2001-09-01  2001      9 -0.099252 -2.702083            False
2011-05-01  2011      5 -0.094555 -2.577442            False
2012-03-01  2012      3 -0.090682 -2.474657            False
