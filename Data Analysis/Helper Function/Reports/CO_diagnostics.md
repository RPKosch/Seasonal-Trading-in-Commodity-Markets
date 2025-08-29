# CO — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **225,595**  
- Minimum monthly volume: **48,893**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std      skew  kurtosis_excess       min       q25   median      q75      max
   292 -0.001387 0.105687 -1.314126        10.196096 -0.763149 -0.055138 0.009932 0.060679 0.476392


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.003137   -0.001810              0.004947  0.256911 0.798964    
02-Feb    25      267  0.026730   -0.004019              0.030749  1.837438 0.075022   *
03-Mar    25      267 -0.010236   -0.000558             -0.009678 -0.281901 0.780291    
04-Apr    25      267  0.005458   -0.002028              0.007486  0.355124 0.725041    
05-May    24      268  0.011677   -0.002557              0.014233  0.472986 0.640311    
06-Jun    24      268  0.023622   -0.003626              0.027249  1.899758 0.065289   *
07-Jul    24      268  0.001138   -0.001613              0.002751  0.141312 0.888594    
08-Aug    24      268 -0.004175   -0.001137             -0.003038 -0.207962 0.836420    
09-Sep    24      268 -0.012861   -0.000359             -0.012502 -0.636946 0.529147    
10-Oct    24      268 -0.027311    0.000935             -0.028246 -1.118376 0.273548    
11-Nov    24      268 -0.036463    0.001754             -0.038217 -1.631174 0.114528    
12-Dec    24      268  0.001369   -0.001633              0.003002  0.135644 0.893091    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -13.638, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.125, p = 0.1000 , lags used = 1  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.6076, F p = 0.6162  
- **White**: LM p = 0.6076, F p = 0.6162
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0001, F p = 0.0001  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.3322. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.51, p = 0.1212, break at index 221 (2019-06-01), left n = 221, right n = 71

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2020-03-01  2020      3 -0.763149   -8.009582  0.040000 0.181759                  True               False            True
2020-05-01  2020      5  0.476392    4.629783  0.041667 0.072380                  True               False            True
2008-10-01  2008     10 -0.403349   -3.697607  0.041667 0.047392                  True               False            True
2020-11-01  2020     11  0.244657    2.734870  0.041667 0.026487                 False               False            True
2020-04-01  2020      4 -0.249544   -2.472714  0.040000 0.020849                 False               False            True
2009-05-01  2009      5  0.244637    2.256890  0.041667 0.018189                 False               False            True
2015-07-01  2015      7 -0.230586   -2.244699  0.041667 0.017996                 False               False            True
2008-12-01  2008     12 -0.221708   -2.159501  0.041667 0.016678                 False               False            True
2014-12-01  2014     12 -0.217760   -2.120667  0.041667 0.016093                 False               False            True
2018-11-01  2018     11 -0.244397   -2.010713  0.041667 0.014491                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2020-03-01  2020      3 -0.763149 -8.890393             True
2020-05-01  2020      5  0.476392  5.364272             True
2008-10-01  2008     10 -0.403349 -4.752713             True
2020-04-01  2020      4 -0.249544 -2.983956            False
2018-11-01  2018     11 -0.244397 -2.924766            False
2015-07-01  2015      7 -0.230586 -2.765946            False
2020-11-01  2020     11  0.244657  2.699329            False
2009-05-01  2009      5  0.244637  2.699096            False
2008-11-01  2008     11 -0.222592 -2.674017            False
2008-12-01  2008     12 -0.221708 -2.663843            False
