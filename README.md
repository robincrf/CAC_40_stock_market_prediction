# Stock Market Prediction & Analysis — FinanceJR Project

## Project Objective
This project aims to simulate and predict the price evolution of real-world stocks (such as CAC 40 companies or major tech stocks) using historical data. 

The main objective is to understand how to model stock prices by combining quantitative finance concepts with modern **Machine Learning algorithms**, and to thoroughly evaluate the quality and accuracy of these predictions against real market trajectories.

---

## Key Concepts
- **Logarithmic Returns**: Measures the relative price variation from one day to the next.
- **Volatility**: Standard deviation of returns, reflecting market uncertainty and risk.
- **Technical Indicators (Feature Engineering)**: RSI, MACD, Simple Moving Averages (SMA), and Momentum to capture market trends.
- **Machine Learning for Time Series**: Using supervised learning to predict future prices and unsupervised learning (Clustering) to identify market regimes.

---

## Methodology

1. **Data Collection**  
   Using the `yfinance` library to download historical market data (Adjusted Close, Open, High, Low, Volume).

2. **Preparation & Feature Engineering**  
   Calculating daily returns and building comprehensive features (Rolling Volatility, Price Ratios, Oscillators). Cleaning data and handling Missing Values.

3. **Modeling**  
   Training robust Machine Learning algorithms to uncover complex market patterns. We deploy three main approaches:
   - **Random Forest**: An ensemble method using regression trees to capture non-linear relationships and assess feature importance.
   - **XGBoost**: A highly optimized gradient boosting algorithm for accurate numerical predictions.
   - **K-Means**: An unsupervised clustering model to identify hidden market states or regimes (e.g., highly volatile vs. steady growth).

4. **Evaluation**  
   Comparing predicted prices with actual prices over the testing period.
   - **RMSE** (Root Mean Squared Error)
   - **MAE** (Mean Absolute Error)
   - **R² Score** (Coefficient of determination)

---

## Project Structure

The project is divided into **5 sequential Jupyter Notebooks** to ensure a clean, step-by-step approach:

- **Notebook 1**: Setup, Data Extraction & Cleaning (`yfinance`) 
- **Notebook 2**: Exploratory Data Analysis (EDA) & Trend Analysis
- **Notebook 3**: Modeling & Prediction using **Random Forest**
- **Notebook 4**: Modeling & Prediction using **XGBoost**
- **Notebook 5**: Market Regime Detection using **K-Means Clustering**

---

## Expected Results
- **Visualizations**: Clear plots overlaying the real price vs. the predicted price trajectory, alongside residual error graphs.
- **Performance Evaluation**:

  | Metric | Description | Interpretation |
  |-----------|-------------|----------------|
  | **RMSE** | Root Mean Square Error | Lower is better |
  | **MAE** | Mean Absolute Error    | Lower is better |
  | **R²**   | Variance explained by the model | Closer to 1.0 = highly accurate |

---

## Technologies Stack
- **Python 3.11+**
- **Core Libraries**:
  - `pandas` — Data manipulation & Time Series
  - `numpy` — Numerical computations
  - `matplotlib` / `seaborn` — Data visualization
  - `yfinance` — Market data retrieval
  - `scikit-learn` — Machine learning (Random Forest, K-Means, Metrics)
  - `xgboost` — Gradient boosting framework
