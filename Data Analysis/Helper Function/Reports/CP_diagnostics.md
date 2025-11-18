# CP — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **33,177**  
- Minimum monthly volume: **1,007**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min       q25   median     q75      max
   292 0.007803 0.073623 -0.108346         2.888205 -0.367347 -0.040128 0.003272 0.05307 0.318526


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.017726    0.006874              0.010852  0.718989 0.477915    
02-Feb    25      267  0.032608    0.005480              0.027128  2.018236 0.052425   *
03-Mar    25      267  0.013212    0.007296              0.005916  0.391715 0.698139    
04-Apr    25      267  0.016375    0.007000              0.009375  0.497767 0.622702    
05-May    24      268 -0.006390    0.009074             -0.015464 -1.164980 0.253357    
06-Jun    24      268 -0.006672    0.009099             -0.015772 -1.161684 0.254788    
07-Jul    24      268  0.021158    0.006607              0.014551  0.969178 0.340819    
08-Aug    24      268 -0.011081    0.009494             -0.020575 -2.125577 0.040046  **
09-Sep    24      268 -0.003570    0.008821             -0.012391 -0.742140 0.464487    
10-Oct    24      268 -0.003323    0.008799             -0.012122 -0.600858 0.553279    
11-Nov    24      268  0.019486    0.006757              0.012729  0.708400 0.484983    
12-Dec    24      268  0.002077    0.008316             -0.006239 -0.398325 0.693487    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -9.692, p = 0.0000 ***, lags used = 1, n = 290  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.178, p = 0.1000 , lags used = 5  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.6412, F p = 0.6500  
- **White**: LM p = 0.6412, F p = 0.6500
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.3075, F p = 0.3109  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic assumption appears reasonable; **robust SEs** are still a conservative default in finance.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.2379. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.55, p = 0.1055, break at index 30 (2003-07-01), left n = 30, right n = 262

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2008-10-01  2008     10 -0.367347   -5.276211  0.041667 0.092041                  True               False            True
2006-04-01  2006      4  0.318526    4.308994  0.040000 0.060664                  True               False            True
2011-09-01  2011      9 -0.249434   -3.470673  0.041667 0.041987                  True               False            True
2009-03-01  2009      3  0.190384    2.473236  0.040000 0.020858                 False               False            True
2016-11-01  2016     11  0.189519    2.373636  0.041667 0.020081                 False               False            True
2010-12-01  2010     12  0.167805    2.312364  0.041667 0.019077                 False               False            True
2011-10-01  2011     10  0.156688    2.231145  0.041667 0.017784                 False               False            True
2008-12-01  2008     12 -0.148808   -2.101820  0.041667 0.015813                 False               False            True
2003-10-01  2003     10  0.145910    2.078444  0.041667 0.015469                 False               False            True
2008-11-01  2008     11 -0.124934   -2.010424  0.041667 0.014487                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2008-10-01  2008     10 -0.367347 -5.287493             True
2006-04-01  2006      4  0.318526  4.497623             True
2011-09-01  2011      9 -0.249434 -3.605276             True
2009-03-01  2009      3  0.190384  2.669464            False
2016-11-01  2016     11  0.189519  2.657121            False
2004-02-01  2004      2  0.176573  2.472433            False
2010-12-01  2010     12  0.167805  2.347332            False
2008-02-01  2008      2  0.165885  2.319950            False
2001-11-01  2001     11  0.158228  2.210703            False
2011-10-01  2011     10  0.156688  2.188733            False
