# ZW — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **46,592**  
- Minimum monthly volume: **6,957**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std     skew  kurtosis_excess       min       q25    median      q75      max
   292 -0.008859 0.084499 0.033478         1.096956 -0.304932 -0.065186 -0.008382 0.043326 0.322288


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.017261   -0.008073             -0.009189 -0.754049 0.455671    
02-Feb    25      267  0.000199   -0.009707              0.009906  0.581088 0.565649    
03-Mar    25      267 -0.026777   -0.007182             -0.019596 -1.289617 0.206768    
04-Apr    25      267 -0.004847   -0.009235              0.004388  0.277828 0.783035    
05-May    24      268  0.003584   -0.009974              0.013558  0.746366 0.461849    
06-Jun    24      268 -0.026190   -0.007307             -0.018883 -0.692550 0.495101    
07-Jul    24      268  0.005765   -0.010169              0.015934  0.670412 0.508704    
08-Aug    24      268 -0.013589   -0.008436             -0.005153 -0.320724 0.750742    
09-Sep    24      268 -0.008600   -0.008882              0.000282  0.013881 0.989030    
10-Oct    24      268  0.000732   -0.009718              0.010450  0.600560 0.553020    
11-Nov    24      268 -0.015013   -0.008308             -0.006705 -0.476521 0.637042    
12-Dec    24      268 -0.003762   -0.009316              0.005554  0.370290 0.713785    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -14.130, p = 0.0000 ***, lags used = 1, n = 290  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.065, p = 0.1000 , lags used = 5  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.0036, F p = 0.0029  
- **White**: LM p = 0.0036, F p = 0.0029
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.6548, F p = 0.6646  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.6914. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.38, p = 0.1744, break at index 158 (2014-03-01), left n = 158, right n = 134

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2010-07-01  2010      7  0.322288    3.876833  0.041667 0.051858                  True               False            True
2011-06-01  2011      6 -0.304932   -3.393625  0.041667 0.040217                  True               False            True
2015-06-01  2015      6  0.240756    3.244479  0.041667 0.036885                  True               False            True
2011-09-01  2011      9 -0.257904   -3.022766  0.041667 0.032171                  True               False            True
2008-10-01  2008     10 -0.242260   -2.943840  0.041667 0.030562                 False               False            True
2015-07-01  2015      7 -0.207702   -2.577021  0.041667 0.023587                 False               False            True
2017-06-01  2017      6  0.169477    2.357653  0.041667 0.019817                 False               False            True
2022-06-01  2022      6 -0.218836   -2.320538  0.041667 0.019210                 False               False            True
2022-02-01  2022      2  0.196338    2.361361  0.040000 0.019050                 False               False            True
2007-08-01  2007      8  0.178075    2.308488  0.041667 0.019014                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2010-07-01  2010      7  0.322288  4.014779             True
2011-06-01  2011      6 -0.304932 -3.600517             True
2011-09-01  2011      9 -0.257904 -3.029533            False
2015-06-01  2015      6  0.240756  3.024876            False
2008-10-01  2008     10 -0.242260 -2.839601            False
2022-06-01  2022      6 -0.218836 -2.555193            False
2022-02-01  2022      2  0.196338  2.485579            False
2015-07-01  2015      7 -0.207702 -2.420015            False
2009-06-01  2009      6 -0.204949 -2.386585            False
2024-06-01  2024      6 -0.196103 -2.279187            False
