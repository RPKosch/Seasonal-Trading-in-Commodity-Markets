# CF — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **15,248**  
- Minimum monthly volume: **4,063**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std     skew  kurtosis_excess       min       q25    median      q75      max
   292 0.002101 0.088848 0.778369         1.711169 -0.229047 -0.055588 -0.010395 0.043923 0.419685


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.020916    0.000340              0.020576  1.178479 0.248034    
02-Feb    25      267  0.008771    0.001477              0.007295  0.307475 0.760881    
03-Mar    25      267 -0.018670    0.004046             -0.022716 -1.290936 0.206800    
04-Apr    25      267  0.011333    0.001237              0.010096  0.603217 0.550881    
05-May    24      268 -0.011233    0.003295             -0.014529 -0.658524 0.516004    
06-Jun    24      268 -0.010151    0.003198             -0.013350 -0.691821 0.494936    
07-Jul    24      268 -0.000832    0.002364             -0.003196 -0.187736 0.852409    
08-Aug    24      268  0.002337    0.002080              0.000257  0.017082 0.986482    
09-Sep    24      268 -0.010673    0.003245             -0.013918 -0.806254 0.426796    
10-Oct    24      268 -0.009596    0.003149             -0.012745 -0.618899 0.541270    
11-Nov    24      268  0.038763   -0.001182              0.039945  1.879386 0.071370   *
12-Dec    24      268  0.003668    0.001961              0.001707  0.104911 0.917162    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -18.353, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.237, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.7321, F p = 0.7406  
- **White**: LM p = 0.7321, F p = 0.7406
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.9945, F p = 0.9951  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic assumption appears reasonable; **robust SEs** are still a conservative default in finance.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.2587. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.74, p = 0.0582, break at index 234 (2020-07-01), left n = 234, right n = 58

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Weak or borderline evidence of a break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2014-02-01  2014      2  0.419685    4.891646  0.040000 0.076796                  True               False            True
2024-11-01  2024     11  0.295256    2.979122  0.041667 0.031277                 False               False            True
2004-05-01  2004      5  0.235379    2.860931  0.041667 0.028914                 False               False            True
2002-03-01  2002      3  0.230108    2.884215  0.040000 0.028149                 False               False            True
2010-06-01  2010      6  0.223534    2.706913  0.041667 0.025962                 False               False            True
2002-10-01  2002     10  0.215668    2.606949  0.041667 0.024124                 False               False            True
2020-01-01  2020      1 -0.207642   -2.643658  0.040000 0.023759                 False               False            True
2004-11-01  2004     11  0.257419    2.528693  0.041667 0.022730                 False               False            True
2008-03-01  2008      3 -0.229047   -2.428724  0.040000 0.020129                 False               False            True
2009-05-01  2009      5  0.187554    2.294359  0.041667 0.018787                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2014-02-01  2014      2  0.419685  5.808668             True
2024-11-01  2024     11  0.295256  4.128123             True
2004-11-01  2004     11  0.257419  3.617108             True
2004-05-01  2004      5  0.235379  3.319431            False
2002-03-01  2002      3  0.230108  3.248234            False
2010-06-01  2010      6  0.223534  3.159449            False
2002-10-01  2002     10  0.215668  3.053217            False
2008-03-01  2008      3 -0.229047 -2.953106            False
2010-12-01  2010     12  0.194438  2.766475            False
2009-05-01  2009      5  0.187554  2.673506            False
