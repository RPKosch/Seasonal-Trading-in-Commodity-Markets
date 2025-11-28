# HO — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **35,989**  
- Minimum monthly volume: **9,283**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min       q25   median      q75      max
   292 0.006129 0.089593 -0.092854         0.673972 -0.314933 -0.045664 0.004018 0.061267 0.275128


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.009878    0.005778              0.004100  0.234198 0.816445    
02-Feb    25      267  0.034515    0.003471              0.031044  1.636662 0.112695    
03-Mar    25      267  0.009005    0.005860              0.003146  0.144636 0.886063    
04-Apr    25      267  0.016681    0.005141              0.011540  0.566687 0.575471    
05-May    24      268  0.010441    0.005743              0.004698  0.236525 0.814817    
06-Jun    24      268  0.023378    0.004584              0.018793  1.628713 0.111359    
07-Jul    24      268  0.003616    0.006354             -0.002738 -0.150021 0.881831    
08-Aug    24      268  0.015216    0.005315              0.009901  0.632041 0.532137    
09-Sep    24      268 -0.005906    0.007207             -0.013113 -0.657757 0.516292    
10-Oct    24      268 -0.016317    0.008139             -0.024457 -1.205767 0.238493    
11-Nov    24      268 -0.028487    0.009229             -0.037716 -1.810582 0.081597   *
12-Dec    24      268 -0.000369    0.006711             -0.007080 -0.344803 0.732955    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -13.945, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.083, p = 0.1000 , lags used = 3  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.6886, F p = 0.6973  
- **White**: LM p = 0.6886, F p = 0.6973
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0551, F p = 0.0522  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** **Mixed/ambiguous** evidence (some tests borderline). Prefer **robust/HAC standard errors** to be safe.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.5494. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.14, p = 0.3269, break at index 260 (2022-09-01), left n = 260, right n = 32

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2020-03-01  2020      3 -0.314933   -3.768024  0.040000 0.047079                  True               False            True
2020-11-01  2020     11  0.275128    3.523948  0.041667 0.043231                  True               False            True
2008-10-01  2008     10 -0.289421   -3.156434  0.041667 0.034978                  True               False            True
2004-09-01  2004      9  0.218549    2.579276  0.041667 0.023627                 False               False            True
2009-05-01  2009      5  0.233802    2.566400  0.041667 0.023397                 False               False            True
2023-07-01  2023      7  0.226028    2.555236  0.041667 0.023199                 False               False            True
2022-04-01  2022      4  0.241831    2.585154  0.040000 0.022743                 False               False            True
2020-01-01  2020      1 -0.196169   -2.361223  0.040000 0.019048                 False               False            True
2005-08-01  2005      8  0.204396    2.166453  0.041667 0.016784                 False               False            True
2015-12-01  2015     12 -0.187816   -2.146278  0.041667 0.016478                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2020-03-01  2020      3 -0.314933 -3.835714             True
2008-10-01  2008     10 -0.289421 -3.528907             True
2020-11-01  2020     11  0.275128  3.260379            False
2022-04-01  2022      4  0.241831  2.859953            False
2009-05-01  2009      5  0.233802  2.763387            False
2023-07-01  2023      7  0.226028  2.669898            False
2003-02-01  2003      2  0.221210  2.611957            False
2004-09-01  2004      9  0.218549  2.579965            False
2005-08-01  2005      8  0.204396  2.409761            False
2020-01-01  2020      1 -0.196169 -2.407462            False
