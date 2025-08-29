# SV — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **32,102**  
- Minimum monthly volume: **1,531**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min       q25    median      q75      max
   292 0.003329 0.087151 -0.110296         0.885415 -0.323058 -0.051699 -0.002185 0.059542 0.266458


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.026835    0.001128              0.025707  1.810518 0.079377   *
02-Feb    25      267  0.009098    0.002789              0.006310  0.368555 0.715091    
03-Mar    25      267  0.006152    0.003064              0.003087  0.172182 0.864494    
04-Apr    25      267 -0.001818    0.003811             -0.005629 -0.265454 0.792660    
05-May    24      268  0.000015    0.003626             -0.003611 -0.167218 0.868494    
06-Jun    24      268 -0.032997    0.006582             -0.039579 -2.528636 0.017051  **
07-Jul    24      268  0.030228    0.000920              0.029308  1.811561 0.080421   *
08-Aug    24      268 -0.000901    0.003708             -0.004608 -0.222130 0.825931    
09-Sep    24      268 -0.018527    0.005286             -0.023813 -1.005683 0.324075    
10-Oct    24      268  0.005349    0.003148              0.002201  0.134318 0.894083    
11-Nov    24      268  0.006344    0.003059              0.003285  0.198936 0.843712    
12-Dec    24      268  0.009044    0.002817              0.006227  0.341559 0.735279    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -18.021, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.114, p = 0.1000 , lags used = 6  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.3885, F p = 0.3938  
- **White**: LM p = 0.3885, F p = 0.3938
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity** (residual variance depends on months). If detected, report heteroskedasticity-robust standard errors (e.g., HC3) when presenting any coefficients.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic error assumption appears reasonable.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.2776. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.71, p = 0.0635, break at index 123 (2011-04-01), left n = 123, right n = 169

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Weak or borderline evidence of a break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2011-09-01  2011      9 -0.323058   -3.641860  0.041667 0.046039                  True               False            True
2008-08-01  2008      8 -0.267453   -3.170101  0.041667 0.035271                  True               False            True
2011-04-01  2011      4  0.266458    3.188478  0.040000 0.034181                  True               False            True
2004-04-01  2004      4 -0.265895   -3.136797  0.040000 0.033119                  True               False            True
2020-07-01  2020      7  0.264928    2.780056  0.041667 0.027345                 False               False            True
2009-05-01  2009      5  0.232638    2.754782  0.041667 0.026864                 False               False            True
2011-05-01  2011      5 -0.227494   -2.692627  0.041667 0.025695                 False               False            True
2008-10-01  2008     10 -0.220880   -2.677089  0.041667 0.025407                 False               False            True
2020-05-01  2020      5  0.204679    2.416276  0.041667 0.020794                 False               False            True
2016-06-01  2016      6  0.147754    2.129067  0.041667 0.016219                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2011-09-01  2011      9 -0.323058 -3.904488             True
2011-04-01  2011      4  0.266458  3.268935            False
2020-07-01  2020      7  0.264928  3.250319            False
2008-08-01  2008      8 -0.267453 -3.227866            False
2004-04-01  2004      4 -0.265895 -3.208910            False
2009-05-01  2009      5  0.232638  2.857399            False
2011-05-01  2011      5 -0.227494 -2.741635            False
2008-10-01  2008     10 -0.220880 -2.661155            False
2020-05-01  2020      5  0.204679  2.517189            False
2020-09-01  2020      9 -0.190700 -2.293914            False
