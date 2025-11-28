# ZW — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **46,592**  
- Minimum monthly volume: **6,957**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean     std     skew  kurtosis_excess       min       q25    median      q75      max
   292 -0.005282 0.08451 0.431613         1.556803 -0.262826 -0.063107 -0.008347 0.044279 0.380282


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.015718   -0.004305             -0.011413 -0.968280 0.339089    
02-Feb    25      267  0.003424   -0.006097              0.009521  0.543314 0.591095    
03-Mar    25      267 -0.024028   -0.003527             -0.020502 -1.366908 0.181460    
04-Apr    25      267 -0.002168   -0.005573              0.003406  0.214562 0.831553    
05-May    24      268  0.007140   -0.006394              0.013535  0.727547 0.473162    
06-Jun    24      268 -0.017768   -0.004164             -0.013604 -0.507681 0.616201    
07-Jul    24      268  0.012200   -0.006847              0.019047  0.762870 0.452708    
08-Aug    24      268 -0.010849   -0.004783             -0.006066 -0.376020 0.709672    
09-Sep    24      268 -0.004238   -0.005375              0.001137  0.057826 0.954321    
10-Oct    24      268  0.003802   -0.006095              0.009897  0.593829 0.557359    
11-Nov    24      268 -0.012954   -0.004595             -0.008359 -0.593991 0.556825    
12-Dec    24      268 -0.001501   -0.005620              0.004120  0.276572 0.784010    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -14.078, p = 0.0000 ***, lags used = 1, n = 290  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.064, p = 0.1000 , lags used = 5  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.0078, F p = 0.0066  
- **White**: LM p = 0.0078, F p = 0.0066
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.7509, F p = 0.7604  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.7828. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.38, p = 0.1749, break at index 158 (2014-03-01), left n = 158, right n = 134

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2010-07-01  2010      7  0.380282    4.551497  0.041667 0.070121                  True               False            True
2015-06-01  2015      6  0.272211    3.536260  0.041667 0.043520                  True               False            True
2011-06-01  2011      6 -0.262826   -2.969496  0.041667 0.031081                 False               False            True
2011-09-01  2011      9 -0.227330   -2.696034  0.041667 0.025759                 False               False            True
2008-10-01  2008     10 -0.215148   -2.644711  0.041667 0.024811                 False               False            True
2022-02-01  2022      2  0.216938    2.575175  0.040000 0.022572                 False               False            True
2007-08-01  2007      8  0.194915    2.481815  0.041667 0.021913                 False               False            True
2017-06-01  2017      6  0.184685    2.441009  0.041667 0.021213                 False               False            True
2015-07-01  2015      7 -0.187551   -2.407746  0.041667 0.020651                 False               False            True
2021-04-01  2021      4  0.185081    2.252216  0.040000 0.017360                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2010-07-01  2010      7  0.380282  4.855457             True
2015-06-01  2015      6  0.272211  3.505238             True
2011-06-01  2011      6 -0.262826 -3.179419            False
2022-02-01  2022      2  0.216938  2.814672            False
2011-09-01  2011      9 -0.227330 -2.735939            False
2008-10-01  2008     10 -0.215148 -2.583737            False
2007-08-01  2007      8  0.194915  2.539522            False
2009-05-01  2009      5  0.186685  2.436698            False
2021-04-01  2021      4  0.185081  2.416650            False
2017-06-01  2017      6  0.184685  2.411703            False
