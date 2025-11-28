# HE — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **14,279**  
- Minimum monthly volume: **2,182**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std      skew  kurtosis_excess       min       q25    median      q75      max
   292 -0.005826 0.074142 -0.085699         0.270937 -0.244186 -0.058943 -0.001389 0.041997 0.197285


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.002844   -0.006106              0.003262  0.191112 0.849832    
02-Feb    25      267 -0.004112   -0.005987              0.001875  0.136169 0.892587    
03-Mar    25      267 -0.012132   -0.005236             -0.006896 -0.380451 0.706567    
04-Apr    25      267 -0.006609   -0.005753             -0.000855 -0.066659 0.947273    
05-May    24      268 -0.014686   -0.005033             -0.009653 -0.793697 0.433341    
06-Jun    24      268 -0.018423   -0.004698             -0.013725 -0.927482 0.361590    
07-Jul    24      268 -0.002801   -0.006097              0.003296  0.191987 0.849216    
08-Aug    24      268 -0.022055   -0.004373             -0.017682 -0.986221 0.333069    
09-Sep    24      268  0.017063   -0.007876              0.024940  1.370336 0.182298    
10-Oct    24      268 -0.000303   -0.006321              0.006018  0.312654 0.757072    
11-Nov    24      268  0.011018   -0.007335              0.018353  1.542738 0.132789    
12-Dec    24      268 -0.013934   -0.005100             -0.008833 -0.620568 0.539803    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -11.107, p = 0.0000 ***, lags used = 2, n = 289  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.063, p = 0.1000 , lags used = 7  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.2306, F p = 0.2318  
- **White**: LM p = 0.2306, F p = 0.2318
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.5036, F p = 0.5118  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic assumption appears reasonable; **robust SEs** are still a conservative default in finance.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.4672. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.28, p = 0.2282, break at index 252 (2022-01-01), left n = 252, right n = 40

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2016-09-01  2016      9 -0.235217   -3.516988  0.041667 0.043067                  True               False            True
2002-08-01  2002      8 -0.244186   -3.081368  0.041667 0.033389                  True               False            True
2016-12-01  2016     12  0.197285    2.925225  0.041667 0.030189                 False               False            True
2020-03-01  2020      3 -0.221613   -2.897832  0.040000 0.028407                 False               False            True
2020-01-01  2020      1 -0.211772   -2.889961  0.040000 0.028258                 False               False            True
2007-07-01  2007      7  0.174541    2.445027  0.041667 0.021282                 False               False            True
2019-03-01  2019      3  0.166667    2.463351  0.040000 0.020695                 False               False            True
2016-07-01  2016      7 -0.177003   -2.400827  0.041667 0.020535                 False               False            True
2002-09-01  2002      9  0.170274    2.106604  0.041667 0.015884                 False               False            True
2008-10-01  2008     10 -0.150058   -2.058367  0.041667 0.015176                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2002-08-01  2002      8 -0.244186 -3.279033            False
2016-09-01  2016      9 -0.235217 -3.157910            False
2020-03-01  2020      3 -0.221613 -2.974178            False
2020-01-01  2020      1 -0.211772 -2.841278            False
2016-12-01  2016     12  0.197285  2.683132            False
2007-07-01  2007      7  0.174541  2.375971            False
2016-07-01  2016      7 -0.177003 -2.371718            False
2002-09-01  2002      9  0.170274  2.318344            False
2019-03-01  2019      3  0.166667  2.269624            False
2009-10-01  2009     10  0.148936  2.030169            False
