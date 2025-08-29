# HE — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **14,279**  
- Minimum monthly volume: **2,182**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std      skew  kurtosis_excess      min       q25   median      q75      max
   292 -0.008653 0.075411 -0.354634         0.644723 -0.27996 -0.060751 -0.00139 0.041139 0.180057


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.006260   -0.008877              0.002616  0.148201 0.883263    
02-Feb    25      267 -0.006115   -0.008890              0.002775  0.203241 0.840285    
03-Mar    25      267 -0.016032   -0.007962             -0.008071 -0.436777 0.665726    
04-Apr    25      267 -0.008417   -0.008675              0.000257  0.019580 0.984502    
05-May    24      268 -0.016285   -0.007969             -0.008316 -0.675800 0.504106    
06-Jun    24      268 -0.020950   -0.007551             -0.013399 -0.891588 0.380194    
07-Jul    24      268 -0.006034   -0.008887              0.002853  0.163469 0.871389    
08-Aug    24      268 -0.026053   -0.007095             -0.018958 -1.009134 0.322243    
09-Sep    24      268  0.013234   -0.010613              0.023846  1.270891 0.215061    
10-Oct    24      268 -0.004354   -0.009038              0.004683  0.242347 0.810433    
11-Nov    24      268  0.009610   -0.010288              0.019898  1.686643 0.101278    
12-Dec    24      268 -0.016083   -0.007987             -0.008095 -0.577880 0.567797    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -10.979, p = 0.0000 ***, lags used = 2, n = 289  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.058, p = 0.1000 , lags used = 7  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.2658, F p = 0.2681  
- **White**: LM p = 0.2658, F p = 0.2681
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity** (residual variance depends on months). If detected, report heteroskedasticity-robust standard errors (e.g., HC3) when presenting any coefficients.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic error assumption appears reasonable.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.4719. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.22, p = 0.2711, break at index 252 (2022-01-01), left n = 252, right n = 40

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2016-09-01  2016      9 -0.268164   -3.873032  0.041667 0.051761                  True               False            True
2002-08-01  2002      8 -0.279960   -3.477343  0.041667 0.042142                  True               False            True
2020-03-01  2020      3 -0.250531   -3.198473  0.040000 0.034388                  True               False            True
2020-01-01  2020      1 -0.237968   -3.159034  0.040000 0.033574                  True               False            True
2016-12-01  2016     12  0.180057    2.663020  0.041667 0.025147                 False               False            True
2016-07-01  2016      7 -0.194803   -2.560554  0.041667 0.023293                 False               False            True
2007-07-01  2007      7  0.160878    2.258291  0.041667 0.018211                 False               False            True
2019-03-01  2019      3  0.154151    2.301344  0.040000 0.018112                 False               False            True
2008-10-01  2008     10 -0.162587   -2.138890  0.041667 0.016367                 False               False            True
2018-07-01  2018      7 -0.163332   -2.126038  0.041667 0.016174                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2002-08-01  2002      8 -0.279960 -3.809856             True
2016-09-01  2016      9 -0.268164 -3.648524             True
2020-03-01  2020      3 -0.250531 -3.407376            False
2020-01-01  2020      1 -0.237968 -3.235557            False
2016-07-01  2016      7 -0.194803 -2.645214            False
2016-12-01  2016     12  0.180057  2.481540            False
2007-07-01  2007      7  0.160878  2.219240            False
2018-07-01  2018      7 -0.163332 -2.214796            False
2008-10-01  2008     10 -0.162587 -2.204614            False
2002-09-01  2002      9  0.157238  2.169464            False
