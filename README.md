# Seasonal-Trading-in-Commodity-Markets

Code and data for the bachelor thesis  
**“Seasonality-Based Trading in Commodity Futures: A Risk-Aware Comparison of Dummy Variable Regression, SSA and RLSSA”** by Ralph Kosch.

The project builds seasonal trading strategies in commodity futures using three models:

- **DVR – Dummy Variable Regression**
- **SSA – Singular Spectrum Analysis**
- **RLSSA – Robust L1 Singular Spectrum Analysis**

From historical futures prices, the pipeline

1. builds continuous contract series and monthly (log) returns,
2. diagnoses data quality and dependence structure,
3. estimates seasonality signals with DVR, SSA and RLSSA,
4. converts these signals into long/short portfolios, and  
5. evaluates them in a Monte Carlo framework with a risk-aware overlay.

The complete thesis is provided as `Bachelor_Thesis_Ralph_Kosch.pdf` in the `/Bachelor Thesis` folder.

---

## Repository Structure

### `Complete Data/`

Holds the input data used by all Python scripts.

- `All_Daily_Contract_Data/`  
  Daily continuous-contract futures prices for all tickers, one file per contract.

- `All_Monthly_Log_Return_Data/`  
  Monthly **log** returns generated from the daily data.

- `All_Monthly_Return_Data/`  
  Monthly **simple** returns generated from the daily data.

- `File_Counter.py`  
  Small helper to check file counts / coverage in the data folders.

- `Single_Ticker_Data_Scrapper.py`  
  Downloads and stores the full history for a single futures ticker.

- `Ticker_Data_Scrapper.py`  
  Batch scraper that downloads data for all tickers in the universe.

- `Ticker_Transformation_Dictionary.py`  
  Maps raw exchange tickers to the internal ticker names used throughout the project
  (ensures consistent naming between the data folders and the analysis scripts).

> **Data linkage:**  
> All analysis scripts expect the daily and monthly data to live in these folders.
> As long as you keep this structure and run the scripts from the project root,
> the relative paths inside the Python files will correctly find the data.

---

### `Data Analysis/`

Higher-level analysis and helper utilities.

#### `/Helper Function`

Scripts that create intermediate datasets and diagnostic plots.

- `Compute_Monthly_Returns.py`  
  Reads daily futures prices from `Complete Data/All_Daily_Contract_Data/`  
  and writes monthly simple and log returns to  
  `Complete Data/All_Monthly_Return_Data/` and  
  `Complete Data/All_Monthly_Log_Return_Data/`.

- `Correlation_Matrix_Creator.py`  
  Builds correlation matrices of monthly returns across all tickers and saves this in the `/Reports` Folder.

- `Data_Diagnostics_Creator.py`  
  Runs the main diagnostics from Chapter 3 (stationarity, volatility clustering,
  outliers, etc.) and saves this in the `/Reports` Folder.

- `Model_Scaling_Experiment.py`  
  Experiments with volatility scaling / normalisation of returns
  (EWMA scaling used in the risk-aware evaluation).

- `Plotting_Ticker_Returns.py`  
  Produces time series plots of long-only indices and monthly returns for each ticker.

- `Ticker_Robustness_Checker.py`  
  Checks robustness of seasonal signals ticker by ticker
  (e.g. sub-sample stability of DVR/SSA/RLSSA coefficients).

---

### `Seasonality Models/`

Implements the three seasonality models and their evaluation on **original** data and on **Monte Carlo–simulated** return paths.

Subfolders:

- `Monte_Carlo_Outputs/`  
  Contains the **100 Monte Carlo return paths** per model together with their performance statistics.  
  These are created by slightly perturbing the original return series with the MEB approach and show the **main quantitative results** of the thesis.

- `Outputs/`  
  Holds the **baseline results on the original historical data**:  

- `plots/`  
  Figures built from `Plotting_Ticker_Returns.py`   
  (Show performance charts).

Model scripts:

- `DVR_Monte_Carlo_Portfolios.py`  
  Reads the DVR-based portfolio returns from `Outputs/`, generates 100 slightly perturbed (bootstrapped) return paths, evaluates each, and stores the simulated equity curves and performance statistics in `Monte_Carlo_Outputs/`.

- `SSA_Monte_Carlo_Portfolios.py`  
  Takes the SSA-based portfolio returns from `Outputs/`, runs the same Monte Carlo procedure (100 adjusted time series), and saves the corresponding paths and metrics to `Monte_Carlo_Outputs/`.

- `RLSSA_Monte_Carlo_Portfolios.py`  
  Uses the RLSSA-based portfolio returns from `Outputs/` as input, performs 100 Monte Carlo simulations, and writes the simulated paths and summary statistics to `Monte_Carlo_Outputs/`.

The corresponding `*_Scoring_Framework.py` scripts  
(`DVR_Scoring_Framework.py`, `SSA_Scoring_Framework.py`, `RLSSA_Scoring_Framework.py`)  
implement the seasonality models themselves on the **original** return series. They compute seasonal scores and baseline portfolio returns and are mainly used to explore and understand how each seasonality algorithm behaves before running the Monte Carlo evaluation.

---

## Typical Workflow

1. **(Optional) Download raw futures data**  
   - Use `Complete Data/Ticker_Data_Scrapper.py` or  
     `Single_Ticker_Data_Scrapper.py` to pull daily data from the data vendor.  
   - Store the resulting files in `Complete Data/All_Daily_Contract_Data/`.

2. **Create monthly return panels**  
   - Run `Data Analysis/Helper Function/Reports/Compute_Monthly_Returns.py`.  
   - This script populates  
     `All_Monthly_Return_Data/` and `All_Monthly_Log_Return_Data/`,
     which are then used by all seasonality model scripts.

3. **Run diagnostics (optional but recommended)**  
   - `Data_Diagnostics_Creator.py`, `Correlation_Matrix_Creator.py`,
     `Plotting_Ticker_Returns.py`, etc.  
   - These scripts only **read** from the `Complete Data` folders
     and write their reports/plots into their own subdirectories.

4. **Monte Carlo portfolio evaluation**  
   - Run `DVR_Monte_Carlo_Portfolios.py`, `SSA_Monte_Carlo_Portfolios.py`,
     and `RLSSA_Monte_Carlo_Portfolios.py`.  
   - These scripts load the scoring files from `Outputs/`,
     build long/short portfolios, and store performance statistics plus
     simulated equity curves in `Monte_Carlo_Outputs/` and `plots/`.

---

## Dependencies

The code uses the standard scientific Python stack, e.g.

- `python >= 3.9`
- `numpy`, `pandas`, `scipy`
- `statsmodels`
- `scikit-learn`
- `matplotlib` / `seaborn`

Install them with your preferred environment manager (e.g. `pip` or `conda`).

---

## Citation

If you use this code or build on the ideas in your own work, please cite:

> Ralph Kosch (2025): *Seasonality-Based Trading in Commodity Futures:  
> A Risk-Aware Comparison of Dummy Variable Regression, SSA and RLSSA.*  
> Bachelor’s Thesis, University of Zurich.
