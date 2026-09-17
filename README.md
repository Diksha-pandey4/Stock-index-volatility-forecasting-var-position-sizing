# Stock Index Volatility Forecasting & VaR-Based Position Sizing

A quantitative finance project focused on **forecasting next-day S&P 500 volatility** using econometric, statistical, and machine learning approaches, followed by **Value-at-Risk (VaR)** estimation and volatility-based position sizing.

The project compares three modeling philosophies:

* **HAR-RV** — a multi-horizon realized volatility benchmark
* **Random Forest** — a non-linear machine learning model
* **GARCH(1,1)** with **Student-t innovations** — a classical econometric volatility model

The resulting volatility forecasts are then used to demonstrate how risk estimates can be incorporated into **VaR calculations and position sizing**.

> **Disclaimer:** This is a forecasting and risk-management illustration, not a trading strategy or investment recommendation.

---

## Project Objective

Predicting the exact next-day stock price is difficult and often less useful for risk management. Volatility, however, exhibits **persistence and clustering**, making it a more meaningful quantity to forecast.

The objective of this project is to:

1. Analyze the volatility characteristics of the S&P 500.
2. Engineer features that capture short-, medium-, and longer-horizon volatility.
3. Compare statistical, machine learning, and econometric forecasting approaches.
4. Evaluate their out-of-sample volatility forecasts.
5. Use volatility forecasts to estimate **VaR**.
6. Demonstrate a **volatility-targeted position-sizing framework**.
7. Compare the resulting framework with a simple buy-and-hold benchmark.

---

## Dataset

* **Index:** S&P 500 (`^GSPC`)
* **Frequency:** Daily
* **Period:** 2010–2023
* **Source:** Yahoo Finance via `yfinance`
* **Primary variable:** Adjusted/automatically adjusted closing price

Daily log returns are calculated as:

[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
]

The project uses annualized absolute log returns as a daily realized-volatility proxy:

[
\sigma_t^{proxy} = |r_t|\sqrt{252}
]

The final modeling dataset contains **3,491 observations** after rolling-window and target construction.

---

## Exploratory Data Analysis

The initial analysis focuses on three important characteristics of financial returns:

### 1. Volatility Clustering

Large market movements tend to be followed by further large movements, while relatively calm periods tend to persist.

This provides the motivation for modeling volatility rather than directly predicting returns.

### 2. Fat-Tailed Return Distribution

The return distribution contains more extreme observations than would be expected under a Normal distribution.

This motivates the use of **Student-t innovations in the GARCH model** and a corresponding fat-tail-aware risk analysis.

### 3. Autocorrelation of Squared Returns

The autocorrelation of squared returns demonstrates persistence in volatility, providing statistical evidence for volatility clustering.

---

## Feature Engineering

The project creates features representing different volatility horizons and market conditions.

### HAR-RV Features

| Feature      | Description                       |
| ------------ | --------------------------------- |
| `RV_Daily`   | Daily volatility proxy            |
| `RV_Weekly`  | 5-day rolling average volatility  |
| `RV_Monthly` | 22-day rolling average volatility |

These features represent different investment horizons and form the basis of the HAR-RV model.

### Machine Learning Features

Additional features are created for the Random Forest model:

* `Negative_Shock` — captures the magnitude of negative lagged returns
* `Roll_Std_10` — 10-day rolling standard deviation
* `Roll_Std_30` — 30-day rolling standard deviation
* `Return_Lag_1` — previous day's log return

The negative-shock feature is designed to allow the model to capture an asymmetric response of volatility to negative market movements.

### Prediction Target

The target is the **next day's volatility proxy**:

```python
data['Target_Vol'] = data['Daily_Vol_Proxy'].shift(-1)
```

Thus, the models perform a genuine **one-step-ahead forecasting task** rather than predicting volatility using information from the future.

---

## Models

### 1. HAR-RV

The **Heterogeneous Autoregressive Realized Volatility** model uses volatility information from multiple horizons:

* Daily
* Weekly
* Monthly

It provides a simple and interpretable statistical benchmark.

---

### 2. Random Forest

A **Random Forest Regressor** is used to investigate whether non-linear relationships can improve volatility forecasting.

It uses:

* HAR-RV features
* Negative shock
* Rolling volatility measures

This provides a machine-learning benchmark against the simpler linear HAR-RV specification.

---

### 3. GARCH(1,1)

A **GARCH(1,1)** model is used to capture the persistence of conditional volatility.

The model is estimated with **Student-t innovations** rather than Gaussian innovations to account for the fat-tailed nature of financial returns.

The fitted degrees-of-freedom parameter is also incorporated into the subsequent VaR analysis.

---

## Train-Test Methodology

Because this is a time-series forecasting problem, the data is **not randomly shuffled**.

An **80/20 chronological split** is used:

```text
Earlier observations ───────────────► Training
                                    │
                                    ▼
Later observations ────────────────► Testing
```

This preserves the temporal ordering of the data and avoids using future observations to train the model.

For GARCH specifically, the model parameters are estimated using **training data only** and then held fixed when generating the volatility path for the test period.

---

## Risk Management: VaR

The volatility forecasts are subsequently incorporated into a **Value-at-Risk framework**.

VaR provides an estimate of the potential loss threshold over a specified horizon and confidence level under the assumed return distribution.

The project uses the Student-t characteristics from the GARCH model to account for heavier tails rather than relying solely on a Normal-return assumption.

---

## Position Sizing

The project demonstrates how a volatility forecast can be translated into a **volatility-targeted position size**.

The basic intuition is:

> Higher forecast volatility → smaller position
> Lower forecast volatility → larger position

This creates a link between:

```text
Volatility Forecast
        ↓
Risk Estimate / VaR
        ↓
Position Size
        ↓
Portfolio Risk Management
```

The position-sizing exercise is illustrative and is not intended to represent a deployable trading strategy.

---

## Key Results

The models were evaluated on a chronological held-out test set using RMSE and QLIKE.

| Model                |       RMSE |      QLIKE |
| -------------------- | ---------: | ---------: |
| GARCH(1,1)-Student-t |     0.1176 | **1.4470** |
| Random Forest        | **0.1111** |     1.7097 |
| HAR-RV               |     0.1127 |     1.7126 |

### Interpretation

* **Random Forest achieved the lowest RMSE**, indicating the smallest average prediction error under the RMSE criterion.
* **GARCH achieved the lowest QLIKE**, making it the selected model for the subsequent risk-management illustration.
* The difference between the models highlights that model performance depends on the evaluation criterion rather than there being a universally best model.
* The selected volatility forecast was subsequently used to illustrate 95% Student-t VaR and volatility-targeted position sizing.

> **Important:** These results come from a single chronological train/test split and should not be interpreted as evidence of live trading performance.


## Model Comparison

The project evaluates the different approaches using out-of-sample forecasting performance.

The comparison is designed to answer:

* Does a simple multi-horizon volatility model provide a strong baseline?
* Does a non-linear Random Forest improve upon the statistical benchmark?
* How does the classical GARCH framework compare with the machine-learning approach?
* How can the resulting volatility estimates be incorporated into risk management?

---

## Key Concepts Demonstrated

* Financial time-series analysis
* Log returns
* Volatility clustering
* Fat-tailed distributions
* Realized volatility proxies
* HAR-RV modeling
* GARCH(1,1)
* Student-t innovations
* Random Forest regression
* Time-series train/test splitting
* Out-of-sample forecasting
* Value-at-Risk (VaR)
* Volatility targeting
* Position sizing
* Quantitative risk management

---

## Tech Stack

**Languages & Tools**

* Python
* Google Colab
* NumPy
* Pandas
* Matplotlib
* Seaborn

**Statistical & Econometric Modeling**

* Statsmodels
* ARCH

**Machine Learning**

* Scikit-learn

**Financial Data**

* yfinance

---

## Project Structure

```text
Stock-Index-Volatility-Forecasting/
│
├── Stock_Index_Volatility_Forecasting_VaR_Position_Sizing.ipynb
├── README.md
└── requirements.txt
```

---

## Limitations & Future Improvements

This project is intended as an educational quantitative-finance exercise and has several limitations.

### Current Limitations

* The volatility target is based on a **single-day absolute-return proxy**, which is inherently noisy.
* The analysis uses daily data rather than high-frequency intraday observations.
* Model evaluation uses a single chronological 80/20 split.
* The position-sizing component is illustrative rather than a live trading system.
* Transaction costs, slippage, liquidity constraints, and market impact are not modeled.

### Possible Extensions

* Use multi-day realized volatility targets.
* Implement rolling or walk-forward validation.
* Compare additional GARCH variants such as EGARCH and GJR-GARCH.
* Incorporate realized volatility from intraday data.
* Compare additional ML models such as XGBoost or LightGBM.
* Perform formal VaR backtesting.
* Add Expected Shortfall (CVaR).
* Incorporate transaction costs and portfolio constraints.
* Extend the framework to multiple assets or indices.

---

## Conclusion

This project demonstrates an end-to-end workflow connecting **financial time-series modeling with practical risk management**:

**Market Data → Return Analysis → Volatility Features → Forecasting → VaR → Position Sizing**

Rather than focusing only on predictive accuracy, the project explores how volatility forecasts can be transformed into interpretable risk measures and used as an input to portfolio decision-making.

---

## Disclaimer

This project is for **educational and research purposes only**. The analysis does not constitute financial advice, an investment recommendation, or a claim of profitability or safety of any trading strategy.

##  Author

**Diksha Pandey**

M.Sc. Industrial Engineering & Operations Research
Indian Institute of Technology Bombay
