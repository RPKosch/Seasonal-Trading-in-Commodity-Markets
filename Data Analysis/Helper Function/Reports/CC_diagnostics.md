# CC — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **13,575**  
- Minimum monthly volume: **2,857**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std     skew  kurtosis_excess       min       q25   median      q75      max
   292 0.008535 0.091455 0.493713            2.427 -0.293859 -0.046537 0.008357 0.059426 0.467345


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.024717    0.007020              0.017697  0.903443 0.373887    
02-Feb    25      267  0.020770    0.007389              0.013381  0.637402 0.529093    
03-Mar    25      267 -0.000442    0.009375             -0.009818 -0.355450 0.725136    
04-Apr    25      267  0.014908    0.007938              0.006970  0.408702 0.685635    
05-May    24      268 -0.019474    0.011043             -0.030517 -1.666262 0.106824    
06-Jun    24      268  0.013483    0.008092              0.005391  0.348572 0.729789    
07-Jul    24      268  0.008692    0.008521              0.000171  0.009498 0.992488    
08-Aug    24      268  0.013660    0.008076              0.005584  0.333465 0.741159    
09-Sep    24      268  0.002886    0.009041             -0.006155 -0.339216 0.736966    
10-Oct    24      268 -0.026045    0.011632             -0.037677 -2.312316 0.027896  **
11-Nov    24      268  0.034562    0.006204              0.028358  1.243681 0.224759    
12-Dec    24      268  0.013626    0.008079              0.005547  0.313215 0.756401    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -19.096, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.210, p = 0.1000 , lags used = 7  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.3544, F p = 0.3589  
- **White**: LM p = 0.3544, F p = 0.3589
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity** (residual variance depends on months). If detected, report heteroskedasticity-robust standard errors (e.g., HC3) when presenting any coefficients.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic error assumption appears reasonable.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.3666. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.88, p = 0.0372, break at index 32 (2003-09-01), left n = 32, right n = 260

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Break suggested at the reported date; parameter shifts before/after that point.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2024-03-01  2024      3  0.467345    5.468983  0.040000 0.094134                  True               False            True
2024-11-01  2024     11  0.313295    3.154415  0.041667 0.034935                  True               False            True
2003-05-01  2003      5 -0.293859   -3.103497  0.041667 0.033854                  True               False            True
2001-01-01  2001      1  0.283768    2.921973  0.040000 0.028868                 False               False            True
2001-11-01  2001     11  0.268207    2.630243  0.041667 0.024547                 False               False            True
2011-03-01  2011      3 -0.224498   -2.517551  0.040000 0.021595                 False               False            True
2024-02-01  2024      2  0.239435    2.455653  0.040000 0.020569                 False               False            True
2024-12-01  2024     12  0.216102    2.272361  0.041667 0.018435                 False               False            True
2008-10-01  2008     10 -0.226547   -2.249809  0.041667 0.018077                 False               False            True
2003-08-01  2003      8  0.208207    2.181829  0.041667 0.017019                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2024-03-01  2024      3  0.467345  5.763376             True
2024-11-01  2024     11  0.313295  3.829014             True
2003-05-01  2003      5 -0.293859 -3.794826             True
2001-01-01  2001      1  0.283768  3.458259            False
2001-11-01  2001     11  0.268207  3.262856            False
2008-10-01  2008     10 -0.226547 -2.949620            False
2011-03-01  2011      3 -0.224498 -2.923882            False
2024-02-01  2024      2  0.239435  2.901584            False
2024-12-01  2024     12  0.216102  2.608596            False
2003-08-01  2003      8  0.208207  2.509465            False
