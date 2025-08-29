# ZS — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **71,053**  
- Minimum monthly volume: **4,537**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min       q25   median      q75      max
   292 0.003907 0.070245 -0.312292         1.051116 -0.247851 -0.038096 0.002084 0.045375 0.192177


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.005225    0.004762             -0.009988 -0.856412 0.398048    
02-Feb    25      267  0.027399    0.001707              0.025692  1.653433 0.109399    
03-Mar    25      267 -0.006113    0.004845             -0.010959 -0.766331 0.449669    
04-Apr    25      267  0.019850    0.002414              0.017436  1.560481 0.128063    
05-May    24      268 -0.003869    0.004603             -0.008473 -0.534880 0.597154    
06-Jun    24      268  0.018389    0.002610              0.015778  0.945756 0.352890    
07-Jul    24      268 -0.018427    0.005907             -0.024334 -1.427756 0.165224    
08-Aug    24      268 -0.005299    0.004731             -0.010031 -0.666425 0.510746    
09-Sep    24      268 -0.027074    0.006681             -0.033755 -1.697280 0.102021    
10-Oct    24      268  0.019162    0.002541              0.016621  1.164647 0.254053    
11-Nov    24      268  0.005591    0.003756              0.001835  0.166747 0.868604    
12-Dec    24      268  0.021658    0.002317              0.019340  1.597411 0.120557    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -7.671, p = 0.0000 ***, lags used = 8, n = 283  
  critical values: 1%=-3.454, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.078, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.1427, F p = 0.1416  
- **White**: LM p = 0.1427, F p = 0.1416
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity** (residual variance depends on months). If detected, report heteroskedasticity-robust standard errors (e.g., HC3) when presenting any coefficients.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic error assumption appears reasonable.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.8201. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.05, p = 0.3997, break at index 174 (2015-07-01), left n = 174, right n = 118

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2008-03-01  2008      3 -0.247851   -3.623456  0.040000 0.043695                  True               False            True
2004-05-01  2004      5 -0.221914   -3.257009  0.041667 0.037160                  True               False            True
2008-09-01  2008      9 -0.219885   -2.868208  0.041667 0.029057                 False               False            True
2018-06-01  2018      6 -0.170722   -2.811591  0.041667 0.027952                 False               False            True
2003-09-01  2003      9  0.159332    2.770268  0.041667 0.027158                 False               False            True
2011-09-01  2011      9 -0.210339   -2.722341  0.041667 0.026251                 False               False            True
2004-07-01  2004      7 -0.194719   -2.616157  0.041667 0.024291                 False               False            True
2005-02-01  2005      2  0.192177    2.439346  0.040000 0.020302                 False               False            True
2008-10-01  2008     10 -0.136308   -2.300889  0.041667 0.018892                 False               False            True
2023-06-01  2023      6  0.170053    2.243553  0.041667 0.017978                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2008-03-01  2008      3 -0.247851 -4.094112             True
2004-05-01  2004      5 -0.221914 -3.669243             True
2008-09-01  2008      9 -0.219885 -3.635996             True
2011-09-01  2011      9 -0.210339 -3.479636            False
2004-07-01  2004      7 -0.194719 -3.223765            False
2005-02-01  2005      2  0.192177  3.113864            False
2018-06-01  2018      6 -0.170722 -2.830678            False
2004-09-01  2004      9 -0.168950 -2.801648            False
2008-02-01  2008      2  0.172929  2.798564            False
2023-06-01  2023      6  0.170053  2.751460            False
