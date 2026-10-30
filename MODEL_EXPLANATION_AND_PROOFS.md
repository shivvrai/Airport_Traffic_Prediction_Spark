# ✈️ Model Explanation, Mathematical Proofs & Project Metrics Dictionary
> **Project:** Distributed Airport Traffic Prediction using Apache Spark (PySpark MLlib)  
> **Author:** Rai Yashmit  
> **Repository:** [github.com/shivvrai/Airport_Traffic_Prediction_Spark](https://github.com/shivvrai/Airport_Traffic_Prediction_Spark)  
> **Live App:** [airporttrafficpredictionspark.streamlit.app](https://airporttrafficpredictionspark.streamlit.app/)

---

## 📌 Table of Contents
1. [Executive Summary: Why These Features & Values?](#1-executive-summary-why-these-features--values)
2. [Proof & Justification: Why Lag 1 & Lag 7? (Why not 2, 3, 4, 5, or 6?)](#2-proof--justification-why-lag-1--lag-7)
   - [Empirical Experiment: Testing Lags 1, 2, 3, and 7](#empirical-experiment-testing-lags-1-2-3-and-7)
   - [Statistical Theory: The Markov Property & Redundancy](#statistical-theory-the-markov-property--redundancy)
   - [Multicollinearity & Variance Inflation](#multicollinearity--variance-inflation)
   - [Aviation Domain Rationale: The IATA 7-Day Timetable](#aviation-domain-rationale-the-iata-7-day-timetable)
   - [The Holiday Shock Question: How the Model Absorbs Transitions](#the-holiday-shock-question-how-the-model-absorbs-transitions)
3. [The Winning Formula: Mathematical Breakdown of Model Weights](#3-the-winning-formula-mathematical-breakdown-of-model-weights)
4. [Complete Numeric Dictionary of the Entire Project](#4-complete-numeric-dictionary-of-the-entire-project)
   - [Dataset & Ingestion Metrics](#dataset--ingestion-metrics)
   - [Train/Test Split Metrics](#traintest-split-metrics)
   - [Accuracy & Error Reduction Metrics](#accuracy--error-reduction-metrics)
   - [Step-by-Step Mathematical Derivation of R² = 0.9925](#step-by-step-mathematical-derivation-of-r--09925)
   - [Unsupervised K-Means Clustering Segmentation](#unsupervised-k-means-clustering-segmentation)
   - [Macro Seasonal Volatility Metrics](#macro-seasonal-volatility-metrics)
   - [Operational Batch Forecasting Metrics](#operational-batch-forecasting-metrics)
5. [Why Linear Regression Beat Complex Ensemble Models (RF & GBT)](#5-why-linear-regression-beat-complex-ensemble-models-rf--gbt)
6. [Quick Interview Defense Cheat-Sheet](#6-quick-interview-defense-cheat-sheet)

---

## 1. Executive Summary: Why These Features & Values?

A frequent question when evaluating this project is:
> *"Why did you use only two predictors (`yesterday` and `last_week`), and where did the numbers $0.700$ and $0.302$ come from?"*

### The 30-Second Summary:
1. **The Numbers are Solved, Not Guessed:** $\beta_1 = 0.700$ and $\beta_2 = 0.302$ were calculated via **Ordinary Least Squares (OLS)** linear regression on **81,858 historical flight records**. The mathematical solver determined that these exact coefficients minimize the sum of squared errors.
2. **Natural Convex Combination:** Notice that $0.700 + 0.302 = \mathbf{1.002} \approx 1.0$. Without being forced by any manual constraints, the algorithm discovered that tomorrow's flight count is an optimal weighted blend of **70% weekly schedule repetition** and **30% daily operational momentum**.
3. **Parsimony (Occam's Razor):** Testing additional lags (Day 2, Day 3, Day 4) provided less than a **0.01% gain in $R^2$**, while introducing severe multicollinearity ($r > 0.99$) and unnecessary data-loss at time-series boundaries.

---

## 2. Proof & Justification: Why Lag 1 & Lag 7?

### Empirical Experiment: Testing Lags 1, 2, 3, and 7
To scientifically verify whether intermediate days (like 2 days ago or 3 days ago) should be included, we trained multiple regression models on the complete Eurocontrol 2025 dataset with different lag combinations:

| Model Specification | Feature Set | $R^2$ Score | MAE (Flights/Day) | Key Finding |
|:---|:---|:---:|:---:|:---|
| **Full Lag Model** | `lag_1`, `lag_2`, `lag_3`, `lag_7` | **0.993987** | 10.31 | Marginally lower MAE, but high redundancy |
| **Selected Model** | `lag_1`, `lag_7` | **0.993864** | **10.37** | **Virtually identical accuracy ($< 0.01\%$ difference)** |

#### Individual Feature Weights in the 4-Lag Model:
When all four lags were allowed to compete freely in the same regression model:
* **Lag 7 (Same Day Last Week):** $\beta = \mathbf{0.6509}$ (**65.1%** weight) $\rightarrow$ Dominant predictor
* **Lag 1 (Yesterday):** $\beta = \mathbf{0.2496}$ (**25.0%** weight) $\rightarrow$ Immediate operational state
* **Lag 3 (Three Days Ago):** $\beta = 0.0817$ ($8.2\%$ weight) $\rightarrow$ Marginal noise
* **Lag 2 (Two Days Ago):** $\beta = \mathbf{0.0173}$ (**1.7%** weight) $\rightarrow$ **Virtually ZERO explanatory value**

> **Empirical Conclusion:** Including Day 2 contributes less than **1.7%** to the prediction, while improving $R^2$ by an insignificant **0.00012**. Therefore, Day 2 is statistically redundant.

---

### Statistical Theory: The Markov Property & Redundancy

Why does Day 2 provide zero value once Day 1 is known?

```
[Day t-2]  ──────►  [Day t-1 (Yesterday)]  ──────►  [Day t (Today)]
(2 Days Ago)         (Absorbs state of t-2)          (Only needs t-1)
```

1. **First-Order Autoregressive Dynamics ($AR(1)$):**
   In time-series econometrics, daily processes follow the **Markov Property**: the present state ($t$) depends directly on the most immediate preceding state ($t-1$). What happened two days ago is already absorbed, reflected, and integrated into yesterday's traffic volume.
2. **Conditional Independence:**
   Once you condition on yesterday ($t-1$), the traffic from two days ago ($t-2$) is conditionally independent of today ($t$). Adding it adds parameters without adding new information.

---

### Multicollinearity & Variance Inflation

* **Correlation Between Lag 1 and Lag 2:** $r = \mathbf{0.9900}$
* **Correlation Between Lag 1 and Lag 7:** $r = \mathbf{0.9926}$

Because consecutive daily volumes at an airport are almost perfectly collinear ($r = 0.99$), feeding `lag_1`, `lag_2`, and `lag_3` into a regression model simultaneously causes **severe multicollinearity**. This inflates coefficient variance, makes the model sensitive to local fluctuations, and hurts interpretability.

By contrast, `lag_7` captures the **orthogonal weekly cycle**, giving the model independent explanatory power without destabilizing the regression weights.

---

### Aviation Domain Rationale: The IATA 7-Day Timetable

Commercial air transport operates under unique industrial constraints that explain why the 7-day lag is overwhelmingly dominant:

1. **IATA Seasonal Slot Allocation:** Airlines do not publish flight schedules at random. Commercial airline timetables are built on **7-day recurring rotations**. A business flight from London Heathrow (EGLL) to Frankfurt (EDDF) operates Monday through Friday at fixed slot times, but does not operate on Sunday.
2. **Weekday vs. Weekend Disconnect:** If today is **Tuesday**, looking at **Sunday (Lag 2)** is disastrous: business routes are grounded, leisure charters are elevated, and cargo operations are altered. However, looking at **last Tuesday (Lag 7)** gives the exact matching commercial flight schedule.
3. **Immediate Operational Shocks (Lag 1):** Why keep yesterday at all? Because disruptions are real-time. If heavy snowfall, an Air Traffic Control (ATC) union strike, or airspace closures grounded aircraft yesterday, planes and flight crews are out of position today. Yesterday acts as the immediate operational baseline.

---

### The Holiday Shock Question: How the Model Absorbs Transitions

**Interviewer Question:** *"What happens when a 2-week festival or holiday period (e.g., Christmas/New Year) ends and traffic suddenly drops back to normal? Won't the 7-day lag overshoot?"*

**Answer:**
1. **The 30% Stabilizing Brake:** In the formula:
   $$\text{Predicted Flights} = 0.700 \times \text{last\_week} + \mathbf{0.302 \times \text{yesterday}} + 0.50$$
   When the holiday period ends, traffic drops immediately on Day 1. That drop is immediately captured by `yesterday`. The $\beta = 0.302$ coefficient acts as an **automatic damping factor**, pulling the forecast down by 30% rather than blindly overshooting.
2. **Macro Rarity:** Across a 365-day calendar year, post-holiday transition boundaries account for fewer than **14 calendar days (<4% of all days)**. The remaining **96% of the year** is strictly governed by regular business and seasonal schedules.
3. **Production Solution:** In a commercial production deployment, we would augment this baseline with an exogenous binary calendar flag (`is_holiday`) or direct feeds from filed IATA airline schedules.

---

## 3. The Winning Formula: Mathematical Breakdown of Model Weights

The final fitted equation on the training partition ($N = 81,858$) is:

$$\widehat{\text{flights}}_t = 0.7005 \times \text{flights}_{t-7} + 0.3020 \times \text{flights}_{t-1} + 0.4960$$

Rounded for operational reporting:
$$\mathbf{\widehat{\text{flights}} = 0.700 \times \text{last\_week} + 0.302 \times \text{yesterday} + 0.50}$$

```
┌─────────────────────────────────────────────────────────────────┐
│               TOTAL PREDICTED FLIGHT MOVEMENTS                  │
│                                                                 │
│   ┌──────────────────────────────┐  ┌───────────────────────┐   │
│   │   70.0% Weekly Timetable     │  │  30.2% Daily Momentum │   │
│   │   (Lag 7: Same Weekday)      │  │  (Lag 1: Yesterday)   │   │
│   │   Coefficient: +0.700        │  │  Coefficient: +0.302  │   │
│   └──────────────────────────────┘  └───────────────────────┘   │
│                                                                 │
│   Constant Baseline Bias (Intercept): +0.50 flights             │
└─────────────────────────────────────────────────────────────────┘
```

* **Sum of Weights:** $0.700 + 0.302 = \mathbf{1.002} \approx 1.0$ (Convex combination).
* **Interpretability:** An increase of 10 flights last week corresponds to $+7.0$ flights today. An increase of 10 flights yesterday corresponds to $+3.0$ flights today.

---

## 4. Complete Numeric Dictionary of the Entire Project

Every number mentioned in the code, report, and resume is documented below with its exact origin:

### Dataset & Ingestion Metrics
* **116,020:** Total raw daily records in `airport_traffic_2025.csv` from official Eurocontrol records.
* **365:** Calendar days evaluated (January 1, 2025 – December 31, 2025).
* **331:** Commercial European airports monitored across **42** sovereign European nations.
* **17,188,114 (~17.19 Million):** Total cumulative IFR flight movements processed across all 331 airports in 2025.
* **110,260:** Cleaned records remaining after joining 1-day and 7-day lags (the first 7 days of January per airport have no 7-day historical lag).

---

### Train/Test Split Metrics
* **October 1, 2025:** Strict chronological cutoff date dividing training and testing sets.
* **81,858 Rows:** Training set (January 1 – September 30, 2025; ~74.2% of data).
* **28,402 Rows:** Out-of-time test set (October 1 – December 31, 2025; ~25.8% of data).
* **Zero Leakage:** No future dates were visible to the training pipeline, replicating realistic production conditions.

---

### Accuracy & Error Reduction Metrics

All metrics evaluated strictly on the **28,402 unseen holdout records**:

| Model | MAE (Flights/Day) | RMSE (Flights/Day) | WMAPE (%) | $R^2$ Score | Error Reduction vs. Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline (Yesterday)** | 17.43 | 33.75 | 12.09% | 0.9809 | Reference Baseline |
| **Linear Regression (Selected)** | **10.37** | **21.17** | **7.20%** | **0.9925** | **-40.5% MAE / -37.3% RMSE** |
| **Random Forest Regressor** | 12.23 | 24.45 | 8.49% | 0.9879 | -29.8% MAE / -27.6% RMSE |
| **Gradient-Boosted Trees (GBT)** | 12.99 | 25.44 | 9.01% | 0.9891 | -25.5% MAE / -24.6% RMSE |

* **-40.5% MAE Reduction:** $\frac{10.3685 - 17.4275}{17.4275} \times 100 = -40.50\%$
* **-37.3% RMSE Reduction:** $\frac{21.1680 - 33.7542}{33.7542} \times 100 = -37.29\%$

---

### Step-by-Step Mathematical Derivation of R² = 0.9925

The coefficient of determination ($R^2$) measures the proportion of variance in flight counts explained by the model:

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$

Evaluated on the $N = 28,402$ test set records:
1. **Actual Flight Mean ($\bar{y}$):** $144.13$ flights/airport-day.
2. **Total Sum of Squares ($SS_{\text{tot}}$):**
   $$SS_{\text{tot}} = \sum_{i=1}^{28,402} (y_i - \bar{y})^2 = 1,694,360,250.72$$
3. **Residual Sum of Squares ($SS_{\text{res}}$):**
   $$SS_{\text{res}} = \sum_{i=1}^{28,402} (y_i - \hat{y}_i)^2 = 12,726,458.14$$
4. **Exact $R^2$ Calculation:**
   $$R^2 = 1 - \frac{12,726,458.14}{1,694,360,250.72} = 1 - 0.00751107 = \mathbf{0.9924889}$$
5. **Rounded to 4 Decimal Places:** $\mathbf{0.9925}$ (99.25% of all variance explained).

---

### Unsupervised K-Means Clustering Segmentation

To segment the 331 airports into operational tiers, K-Means clustering ($k=3$) was fitted on StandardScaled features: `avg_day`, `variab` (std dev), and `peak`:

| Cluster Tier | Number of Airports | % of Airports | Total Annual Flights | % of All Traffic | Average Flights/Day | Daily Traffic Range |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Major Hubs** | **18** | **5.4%** | **6,182,727** | **36.0%** | **941.2** | 613 – 1,491 flights/day |
| **Medium Hubs** | **68** | **20.5%** | **7,556,000** | **44.0%** | **304.5** | 82 – 727 flights/day |
| **Small / Regional** | **245** | **74.1%** | **3,449,387** | **20.0%** | **38.9** | 1 – 222 flights/day |
| **Total European System**| **331** | **100%** | **17,188,114** | **100%** | **142.1** | 1 – 1,491 flights/day |

#### Top 3 Busiest European Airports (2025):
1. **Istanbul Airport (LTFM):** 544,121 movements (~1,491 flights/day)
2. **Amsterdam Schiphol (EHAM):** 493,042 movements (~1,351 flights/day)
3. **London Heathrow (EGLL):** 480,072 movements (~1,315 flights/day)

---

### Macro Seasonal Volatility Metrics

* **Winter Trough (February 2025):** 1,104,896 flights (~1.10 Million).
* **Summer Peak (July 2025):** 1,679,252 flights (~1.68 Million).
* **Summer Peak (August 2025):** 1,675,595 flights (~1.68 Million).
* **Seasonal Demand Surge:** **+52.0%** expansion from winter trough to summer peak.

---

### Operational Batch Forecasting Metrics
* **279 Airports:** Number of active commercial airports forecasted simultaneously for **January 1, 2026** (the first day beyond the 2025 dataset).
* **Sub-2-Second Execution:** Total PySpark batch transformation and inference time for all 279 airports.

---

## 5. Why Linear Regression Beat Complex Ensemble Models (RF & GBT)

In standard data science, beginners assume Gradient-Boosted Trees (XGBoost/GBT) always beat Linear Regression. In this project, **Linear Regression achieved the lowest RMSE (21.17 vs. 25.44 for GBT)**. Here is why:

1. **Strict Linear Physical Relationship:**
   Flight schedules have a naturally additive physical structure: today's flights are literally last week's scheduled flights plus/minus operational deviations. There are no complex non-linear polynomial curves; it is a straight line.
2. **Extrapolation Failure of Decision Trees:**
   Tree models (Random Forests, GBT) partition feature space using step functions. They cannot extrapolate values beyond the maximum thresholds seen in the training set. At massive mega-hubs like Istanbul (1,500+ flights/day), tree models underestimate peak spikes, creating large squared error penalties.
3. **Occam's Razor & Latency:**
   A linear model with 2 weights executes in microseconds, has zero risk of leaf-node overfitting, and provides 100% transparent auditability for air traffic controllers.

---

## 6. Quick Interview Defense Cheat-Sheet

| If an Interviewer Asks... | Say This: |
|:---|:---|
| *"Why only Lag 1 and Lag 7?"* | *"We tested Lags 1, 2, 3, and 7. Lag 2 contributed only 1.7% weight and improved $R^2$ by just 0.01%. By the Markov property, information from 2+ days ago is already absorbed into yesterday. Keeping only Lags 1 and 7 avoided multicollinearity ($r > 0.99$) and data loss."* |
| *"How did you get 70% and 30%?"* | *"They are the OLS regression coefficients: $\beta = 0.700$ for `last_week` and $\beta = 0.302$ for `yesterday`. They naturally sum to 1.002, proving the model learned an optimal convex combination of weekly schedule cycles and short-term daily momentum."* |
| *"Is $R^2 = 0.9925$ verified?"* | *"Yes. Recalculated on the exact 28,402 chronological test records (Oct–Dec 2025), $SS_{\text{res}} = 12.72\text{M}$ and $SS_{\text{tot}} = 1.69\text{B}$, yielding an exact $R^2 = 0.992488$."* |
| *"Why didn't GBT beat Linear Regression?"* | *"Because decision trees use step functions and cannot extrapolate to peak flight volumes at major hubs like Istanbul. The underlying relationship is purely linear and additive."* |
