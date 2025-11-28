# NG — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **71,479**  
- Minimum monthly volume: **9,241**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std     skew  kurtosis_excess       min       q25    median      q75      max
   292 -0.016912 0.140824 0.493896         1.242833 -0.345665 -0.107307 -0.019821 0.061626 0.522744


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.042769   -0.014491             -0.028278 -0.865756 0.394082    
02-Feb    25      267 -0.004116   -0.018110              0.013993  0.438794 0.664196    
03-Mar    25      267  0.005684   -0.019028              0.024711  0.824342 0.416595    
04-Apr    25      267  0.011990   -0.019618              0.031608  1.415561 0.166127    
05-May    24      268 -0.013759   -0.017194              0.003436  0.168026 0.867541    
06-Jun    24      268 -0.016320   -0.016965              0.000645  0.022237 0.982418    
07-Jul    24      268 -0.008372   -0.017677              0.009305  0.280304 0.781433    
08-Aug    24      268 -0.015959   -0.016997              0.001038  0.028398 0.977564    
09-Sep    24      268  0.002138   -0.018618              0.020756  0.641441 0.526733    
10-Oct    24      268 -0.020198   -0.016618             -0.003580 -0.154080 0.878538    
11-Nov    24      268 -0.042489   -0.014621             -0.027868 -0.933510 0.358728    
12-Dec    24      268 -0.060373   -0.013020             -0.047353 -1.377082 0.180210    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -4.722, p = 0.0001 ***, lags used = 11, n = 280  
  critical values: 1%=-3.454, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.055, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.5484, F p = 0.5564  
- **White**: LM p = 0.5484, F p = 0.5564
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0055, F p = 0.0044  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.8757. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.85, p = 0.0407, break at index 47 (2004-12-01), left n = 47, right n = 245

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Break suggested at the reported date; parameter shifts before/after that point.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2003-02-01  2003      2  0.522744    3.881225  0.040000 0.049804                  True               False            True
2022-07-01  2022      7  0.469727    3.508523  0.041667 0.042869                  True               False            True
2005-08-01  2005      8  0.434000    3.293734  0.041667 0.037971                  True               False            True
2018-11-01  2018     11  0.387485    3.142138  0.041667 0.034673                  True               False            True
2022-01-01  2022      1  0.371412    3.020161  0.040000 0.030779                  True               False            True
2002-03-01  2002      3  0.353814    2.526435  0.040000 0.021745                 False               False            True
2022-06-01  2022      6 -0.342545   -2.366270  0.041667 0.019959                 False               False            True
2020-08-01  2020      8  0.309109    2.357709  0.041667 0.019818                 False               False            True
2021-09-01  2021      9  0.317243    2.284072  0.041667 0.018622                 False               False            True
2003-12-01  2003     12  0.254103    2.279431  0.041667 0.018547                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2003-02-01  2003      2  0.522744  4.329717             True
2022-07-01  2022      7  0.469727  3.906631             True
2005-08-01  2005      8  0.434000  3.621529             True
2018-11-01  2018     11  0.387485  3.250335            False
2022-01-01  2022      1  0.371412  3.122075            False
2002-03-01  2002      3  0.353814  2.981641            False
2021-09-01  2021      9  0.317243  2.689797            False
2020-08-01  2020      8  0.309109  2.624888            False
2022-12-01  2022     12 -0.345665 -2.600264            False
2022-06-01  2022      6 -0.342545 -2.575373            False
