# CP — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **33,177**  
- Minimum monthly volume: **1,007**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min       q25   median     q75      max
   292 0.005052 0.074499 -0.751287         5.341274 -0.457833 -0.040956 0.003267 0.05171 0.276515


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.015128    0.004109              0.011019  0.732591 0.469663    
02-Feb    25      267  0.030341    0.002684              0.027656  2.154811 0.038909  **
03-Mar    25      267  0.010754    0.004518              0.006236  0.423689 0.674874    
04-Apr    25      267  0.012602    0.004345              0.008257  0.465817 0.645027    
05-May    24      268 -0.008231    0.006242             -0.014473 -1.078513 0.289536    
06-Jun    24      268 -0.008610    0.006276             -0.014885 -1.085163 0.286735    
07-Jul    24      268  0.018652    0.003834              0.014818  0.997042 0.327268    
08-Aug    24      268 -0.011988    0.006578             -0.018567 -1.902462 0.064612   *
09-Sep    24      268 -0.006832    0.006116             -0.012948 -0.728359 0.472843    
10-Oct    24      268 -0.008659    0.006280             -0.014939 -0.650384 0.521447    
11-Nov    24      268  0.015958    0.004075              0.011883  0.674527 0.505858    
12-Dec    24      268 -0.000515    0.005551             -0.006066 -0.386085 0.702414    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -9.565, p = 0.0000 ***, lags used = 1, n = 290  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.138, p = 0.1000 , lags used = 5  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.6510, F p = 0.6598  
- **White**: LM p = 0.6510, F p = 0.6598
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity** (residual variance depends on months). If detected, report heteroskedasticity-robust standard errors (e.g., HC3) when presenting any coefficients.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic error assumption appears reasonable.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.3267. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.49, p = 0.1273, break at index 30 (2003-07-01), left n = 30, right n = 262

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2008-10-01  2008     10 -0.457833   -6.590874  0.041667 0.136674                  True               False            True
2011-09-01  2011      9 -0.286928   -3.927338  0.041667 0.053146                  True               False            True
2006-04-01  2006      4  0.276515    3.685628  0.040000 0.045138                  True               False            True
2008-12-01  2008     12 -0.161117   -2.211212  0.041667 0.017473                 False               False            True
2009-03-01  2009      3  0.174276    2.250141  0.040000 0.017329                 False               False            True
2016-11-01  2016     11  0.173549    2.169044  0.041667 0.016824                 False               False            True
2010-12-01  2010     12  0.155126    2.141756  0.041667 0.016410                 False               False            True
2011-10-01  2011     10  0.145561    2.121885  0.041667 0.016111                 False               False            True
2008-09-01  2008      9 -0.159100   -2.094618  0.041667 0.015706                 False               False            True
2008-11-01  2008     11 -0.133456   -2.054744  0.041667 0.015123                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2008-10-01  2008     10 -0.457833 -6.626405             True
2011-09-01  2011      9 -0.286928 -4.170353             True
2006-04-01  2006      4  0.276515  3.926812             True
2009-03-01  2009      3  0.174276  2.457554            False
2016-11-01  2016     11  0.173549  2.447106            False
2008-12-01  2008     12 -0.161117 -2.362339            False
2008-09-01  2008      9 -0.159100 -2.333356            False
2004-02-01  2004      2  0.162606  2.289852            False
2010-12-01  2010     12  0.155126  2.182347            False
2008-02-01  2008      2  0.153481  2.158710            False
