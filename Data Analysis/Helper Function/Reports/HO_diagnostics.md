# HO — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **35,989**  
- Minimum monthly volume: **9,283**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min      q25  median      q75      max
   292 0.002077 0.090583 -0.474448         1.238319 -0.378238 -0.04674 0.00401 0.059463 0.243046


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.006501    0.001662              0.004839  0.273675 0.786236    
02-Feb    25      267  0.030237   -0.000560              0.030797  1.673262 0.105031    
03-Mar    25      267  0.003184    0.001973              0.001211  0.052175 0.958775    
04-Apr    25      267  0.012057    0.001142              0.010915  0.542069 0.592054    
05-May    24      268  0.006328    0.001696              0.004632  0.236957 0.814468    
06-Jun    24      268  0.021997    0.000293              0.021704  1.910033 0.063200   *
07-Jul    24      268  0.000186    0.002246             -0.002060 -0.113730 0.910262    
08-Aug    24      268  0.012770    0.001119              0.011650  0.764670 0.450285    
09-Sep    24      268 -0.010145    0.003171             -0.013316 -0.669600 0.508802    
10-Oct    24      268 -0.021275    0.004168             -0.025442 -1.181204 0.248110    
11-Nov    24      268 -0.033678    0.005279             -0.038956 -1.856887 0.074501   *
12-Dec    24      268 -0.005062    0.002716             -0.007778 -0.368029 0.715782    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -13.719, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.076, p = 0.1000 , lags used = 3  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.6061, F p = 0.6147  
- **White**: LM p = 0.6061, F p = 0.6147
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0100, F p = 0.0084  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.5409. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.09, p = 0.3644, break at index 229 (2020-02-01), left n = 229, right n = 63

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2020-03-01  2020      3 -0.378238   -4.432508  0.040000 0.063960                  True               False            True
2008-10-01  2008     10 -0.341675   -3.688732  0.041667 0.047176                  True               False            True
2020-11-01  2020     11  0.243046    3.166336  0.041667 0.035191                  True               False            True
2020-01-01  2020      1 -0.218367   -2.555153  0.040000 0.022231                 False               False            True
2004-09-01  2004      9  0.197661    2.359351  0.041667 0.019845                 False               False            True
2009-05-01  2009      5  0.210100    2.312662  0.041667 0.019082                 False               False            True
2023-07-01  2023      7  0.203779    2.310590  0.041667 0.019048                 False               False            True
2015-12-01  2015     12 -0.208029   -2.303350  0.041667 0.018931                 False               False            True
2022-04-01  2022      4  0.216587    2.319370  0.040000 0.018391                 False               False            True
2020-04-01  2020      4 -0.189577   -2.285922  0.040000 0.017874                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2020-03-01  2020      3 -0.378238 -4.701141             True
2008-10-01  2008     10 -0.341675 -4.251461             True
2020-11-01  2020     11  0.243046  2.939831            False
2020-01-01  2020      1 -0.218367 -2.734936            False
2022-04-01  2022      4  0.216587  2.614417            False
2015-12-01  2015     12 -0.208029 -2.607790            False
2018-11-01  2018     11 -0.202175 -2.535803            False
2009-05-01  2009      5  0.210100  2.534635            False
2008-11-01  2008     11 -0.200244 -2.512054            False
2023-07-01  2023      7  0.203779  2.456899            False
