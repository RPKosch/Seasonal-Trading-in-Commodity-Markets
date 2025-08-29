# GD — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **86,899**  
- Minimum monthly volume: **1,084**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min       q25   median      q75      max
   292 0.006139 0.046344 -0.277764         0.922544 -0.199645 -0.023337 0.004756 0.035614 0.121137


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.023335    0.004529              0.018806  1.961842 0.059525   *
02-Feb    25      267  0.004279    0.006313             -0.002034 -0.217907 0.829019    
03-Mar    25      267  0.003034    0.006430             -0.003396 -0.408098 0.686015    
04-Apr    25      267  0.010013    0.005776              0.004237  0.433407 0.667970    
05-May    24      268  0.001130    0.006588             -0.005457 -0.583635 0.564151    
06-Jun    24      268 -0.007906    0.007397             -0.015303 -1.468013 0.153772    
07-Jul    24      268  0.006916    0.006070              0.000846  0.101538 0.919809    
08-Aug    24      268  0.015503    0.005301              0.010202  1.077682 0.290453    
09-Sep    24      268  0.001489    0.006556             -0.005066 -0.450372 0.656159    
10-Oct    24      268 -0.000895    0.006769             -0.007664 -0.692463 0.494730    
11-Nov    24      268  0.005931    0.006158             -0.000227 -0.020590 0.983728    
12-Dec    24      268  0.010168    0.005778              0.004390  0.438466 0.664517    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -18.834, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.159, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.9676, F p = 0.9699  
- **White**: LM p = 0.9676, F p = 0.9699
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity** (residual variance depends on months). If detected, report heteroskedasticity-robust standard errors (e.g., HC3) when presenting any coefficients.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic error assumption appears reasonable.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.2871. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 2.74, p = 0.0016, break at index 122 (2011-03-01), left n = 122, right n = 170

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Break suggested at the reported date; parameter shifts before/after that point.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2008-10-01  2008     10 -0.199645   -4.507064  0.041667 0.068851                  True               False            True
2011-12-01  2011     12 -0.112008   -2.709901  0.041667 0.026018                 False               False            True
2011-09-01  2011      9 -0.118392   -2.657710  0.041667 0.025050                 False               False            True
2013-06-01  2013      6 -0.126561   -2.629836  0.041667 0.024540                 False               False            True
2008-08-01  2008      8 -0.099000   -2.535655  0.041667 0.022852                 False               False            True
2008-11-01  2008     11  0.115040    2.413630  0.041667 0.020750                 False               False            True
2009-11-01  2009     11  0.113447    2.377683  0.041667 0.020148                 False               False            True
2004-04-01  2004      4 -0.098238   -2.392170  0.040000 0.019540                 False               False            True
2011-08-01  2011      8  0.121137    2.335239  0.041667 0.019449                 False               False            True
2009-05-01  2009      5  0.098099    2.140387  0.041667 0.016389                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2008-10-01  2008     10 -0.199645 -4.690068             True
2013-06-01  2013      6 -0.126561 -3.013125            False
2011-09-01  2011      9 -0.118392 -2.825693            False
2011-12-01  2011     12 -0.112008 -2.679200            False
2011-08-01  2011      8  0.121137  2.670416            False
2008-11-01  2008     11  0.115040  2.530509            False
2009-11-01  2009     11  0.113447  2.493970            False
2008-08-01  2008      8 -0.099000 -2.380728            False
2004-04-01  2004      4 -0.098238 -2.363251            False
2006-04-01  2006      4  0.107485  2.357163            False
