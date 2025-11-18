# SV — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **32,102**  
- Minimum monthly volume: **1,531**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std     skew  kurtosis_excess       min       q25    median     q75      max
   292 0.007129 0.087671 0.248574         0.753476 -0.276068 -0.050385 -0.002182 0.06135 0.305332


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.029335    0.005049              0.024285  1.654377 0.107761    
02-Feb    25      267  0.012368    0.006638              0.005730  0.327347 0.745734    
03-Mar    25      267  0.009733    0.006885              0.002848  0.155999 0.877124    
04-Apr    25      267  0.003284    0.007489             -0.004205 -0.194978 0.846863    
05-May    24      268  0.005135    0.007307             -0.002173 -0.098152 0.922570    
06-Jun    24      268 -0.030012    0.010455             -0.040466 -2.596771 0.014503  **
07-Jul    24      268  0.033552    0.004762              0.028790  1.646020 0.110934    
08-Aug    24      268  0.003665    0.007439             -0.003774 -0.184695 0.854877    
09-Sep    24      268 -0.012475    0.008884             -0.021359 -0.946968 0.352479    
10-Oct    24      268  0.008075    0.007044              0.001031  0.064167 0.949273    
11-Nov    24      268  0.009212    0.006942              0.002270  0.134735 0.893766    
12-Dec    24      268  0.012580    0.006640              0.005939  0.326589 0.746443    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -17.918, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.124, p = 0.1000 , lags used = 6  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.5416, F p = 0.5496  
- **White**: LM p = 0.5416, F p = 0.5496
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0011, F p = 0.0007  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.2504. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.70, p = 0.0670, break at index 124 (2011-05-01), left n = 124, right n = 168

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Weak or borderline evidence of a break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2011-04-01  2011      4  0.305332    3.580899  0.040000 0.042720                  True               False            True
2020-07-01  2020      7  0.303337    3.186502  0.041667 0.035624                  True               False            True
2011-09-01  2011      9 -0.276068   -3.110802  0.041667 0.034008                  True               False            True
2009-05-01  2009      5  0.261924    3.027837  0.041667 0.032275                  True               False            True
2008-08-01  2008      8 -0.234673   -2.803903  0.041667 0.027804                 False               False            True
2004-04-01  2004      4 -0.233480   -2.782379  0.040000 0.026249                 False               False            True
2020-05-01  2020      5  0.227131    2.606791  0.041667 0.024122                 False               False            True
2011-05-01  2011      5 -0.203473   -2.446087  0.041667 0.021300                 False               False            True
2008-10-01  2008     10 -0.198187   -2.418005  0.041667 0.020823                 False               False            True
2016-06-01  2016      6  0.159228    2.214787  0.041667 0.017528                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2011-04-01  2011      4  0.305332  3.723136             True
2020-07-01  2020      7  0.303337  3.698976             True
2011-09-01  2011      9 -0.276068 -3.315978            False
2009-05-01  2009      5  0.261924  3.197583            False
2008-08-01  2008      8 -0.234673 -2.814807            False
2004-04-01  2004      4 -0.233480 -2.800361            False
2020-05-01  2020      5  0.227131  2.776338            False
2011-02-01  2011      2  0.204630  2.503920            False
2011-05-01  2011      5 -0.203473 -2.437055            False
2008-10-01  2008     10 -0.198187 -2.373061            False
