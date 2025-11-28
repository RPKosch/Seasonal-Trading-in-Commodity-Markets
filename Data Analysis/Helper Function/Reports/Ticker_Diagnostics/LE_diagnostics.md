# LE — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **18,686**  
- Minimum monthly volume: **3,970**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min       q25   median      q75      max
   292 0.000384 0.041595 -0.585868         1.866212 -0.208131 -0.022979 0.002578 0.027126 0.118602


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.003219    0.000722             -0.003941 -0.562429 0.577724    
02-Feb    25      267  0.000002    0.000420             -0.000418 -0.054095 0.957214    
03-Mar    25      267 -0.009451    0.001305             -0.010756 -0.972712 0.339470    
04-Apr    25      267 -0.004275    0.000821             -0.005096 -0.571645 0.572056    
05-May    24      268  0.004039    0.000057              0.003982  0.438896 0.664222    
06-Jun    24      268  0.012383   -0.000690              0.013073  1.603116 0.120035    
07-Jul    24      268  0.010492   -0.000521              0.011013  1.486447 0.147700    
08-Aug    24      268 -0.014875    0.001751             -0.016626 -2.282008 0.029768  **
09-Sep    24      268  0.002076    0.000233              0.001843  0.173420 0.863678    
10-Oct    24      268  0.007648   -0.000266              0.007914  0.886225 0.383262    
11-Nov    24      268  0.003856    0.000073              0.003782  0.443232 0.661037    
12-Dec    24      268 -0.003293    0.000714             -0.004007 -0.398198 0.693718    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -16.546, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.071, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.6927, F p = 0.7014  
- **White**: LM p = 0.6927, F p = 0.7014
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.2809, F p = 0.2835  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic assumption appears reasonable; **robust SEs** are still a conservative default in finance.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.9133. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.45, p = 0.1425, break at index 43 (2004-08-01), left n = 43, right n = 249

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2003-12-01  2003     12 -0.208131   -5.253319  0.041667 0.091316                  True               False            True
2017-04-01  2017      4  0.118602    3.053322  0.040000 0.031436                  True               False            True
2018-03-01  2018      3 -0.113823   -2.581502  0.040000 0.022681                 False               False            True
2015-09-01  2015      9 -0.096913   -2.447614  0.041667 0.021326                 False               False            True
2020-02-01  2020      2 -0.101107   -2.498979  0.040000 0.021285                 False               False            True
2001-09-01  2001      9 -0.094486   -2.386345  0.041667 0.020293                 False               False            True
2011-05-01  2011      5 -0.090222   -2.328371  0.041667 0.019337                 False               False            True
2006-03-01  2006      3 -0.102595   -2.298215  0.040000 0.018063                 False               False            True
2003-07-01  2003      7  0.097458    2.145066  0.041667 0.016460                 False               False            True
2008-10-01  2008     10 -0.078529   -2.125278  0.041667 0.016162                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2003-12-01  2003     12 -0.208131 -5.589560             True
2018-03-01  2018      3 -0.113823 -3.087801            False
2017-04-01  2017      4  0.118602  3.077826            False
2006-03-01  2006      3 -0.102595 -2.789956            False
2020-02-01  2020      2 -0.101107 -2.750486            False
2020-03-01  2020      3 -0.097304 -2.649596            False
2015-09-01  2015      9 -0.096913 -2.639233            False
2001-09-01  2001      9 -0.094486 -2.574832            False
2003-07-01  2003      7  0.097458  2.516920            False
2011-05-01  2011      5 -0.090222 -2.461740            False
