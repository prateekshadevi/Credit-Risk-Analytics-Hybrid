# Loan Default Prediction: A Hybrid Statistical & Machine Learning Framework

![Project Overview](https://img.shields.io/badge/Project-Data_Science-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-orange)

## 📌 Problem Statement
Loan defaults can greatly impact financial institutions' profitability and sustainability. While traditional credit scoring models offer some predictive power, they often overlook intricate patterns hidden within a borrower's holistic profile. 

The core objective of this project is to build a reliable, data-driven framework to predict loan default risks early[cite: 2]. By combining credit risk indicators (e.g., credit score, credit lines) with borrower profile features (e.g., income, debt-to-income ratio, employment type, and loan terms), this framework enables lenders to optimize loan approvals while minimizing losses.

---

## 🛠️ Tech Stack & Libraries
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Statistical Modeling:** `statsmodels`, `scipy`
* **Machine Learning & Dimensionality Reduction:** `scikit-learn`, `xgboost`

---

## 📊 Exploratory Data Analysis (EDA) & Key Findings
The project utilizes a real-world dataset comprising **255,347 loan records** with 17 features. 

### 1. Structure & Quality Check
* **Shape:** 255,347 rows, 17 columns.
* **Missing Values:** 0 missing values across all fields, simplifying preprocessing.
* **Outliers:** Boxplot analysis on key numerical columns (`CreditScore`, `InterestRate`, `Income`) confirmed no extreme outliers that would skew our models.

### 2. Class Imbalance
The target variable (`Default`) exhibits a severe class imbalance:
* **Non-Default (0):** 225,694 records (~88.4%)
* **Default (1):** 29,653 records (~11.6%)

*Note: This imbalance was handled across the models using stratified splitting, class-weight adjustments, and robust optimization techniques.*

---

## 🔬 Hypothesis Testing & Modeling Approach

We formulated and tested three risk-tiered hypotheses using a hybrid approach of traditional statistical methods, dimensionality reduction, and advanced ensemble models[cite: 2]. Feature selection was performed using Recursive Feature Elimination (RFE) to narrow down the predictors to the most informative ten.

### Hypothesis 1 (Low-Risk): Do borrowers with lower credit scores have a higher likelihood of loan default?
* **Methods Used:** Logistic Regression, XGBoost, PCA + Logistic Regression
* **Result:** **Proven.** Logistic Regression yielded a negative coefficient ($-0.0007$, $p < 0.0001$), confirming default risk decreases as credit score increases. 
* **Top Performer:** **XGBoost** achieved the highest overall predictive power with a Spearman correlation of $\rho = -0.112$ ($p < 0.001$) and an **ROC AUC of 0.7370**.

### Hypothesis 2 (Moderate-Risk): How does the interest rate impact the likelihood of default across different income levels?
* **Methods Used:** Interaction Logistic Regression, Linear Discriminant Analysis (LDA), Random Forest
* **Result:** **Proven.** 
* **Top Performer:** **Random Forest** outperformed other methods with an **Accuracy of 74.31%** and an **ROC AUC of 0.7167**, capturing complex socioeconomic interactions showing that low-income tiers are highly vulnerable to interest rate hikes.

### Hypothesis 3 (High-Risk): How will economic downturns or policy changes affect loan default rates over the next decade?
* **Methods Used:** Logistic Regression, Ridge Regression, Bootstrap Resampling
* **Result:** **Proven.** Since future unseen data isn't available, split testing datasets and bootstrap techniques simulated these environments effectively.
* **Top Performer:** **Bootstrap Resampling + Logistic Regression** provided the most stable framework for simulating future/uncertain economic conditions, maintaining an **ROC AUC of 0.7557** and a strong **Recall of 0.7026**.

---

## 📈 Model Comparison Metrics

| Hypothesis / Model | Accuracy | Recall | F1-Score | ROC AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hypothesis 1** | | | | | |
| ↳ Logistic Regression | 0.515 | 0.5268 | 0.201 | 0.5307 | *Failed*|
| ↳ **XGBoost** | **0.722** | **0.6140** | **0.339** | **0.7370** | **Best** |
| ↳ PCA + Logistic Regression | 0.672 | 0.6713 | 0.323 | 0.6734 | *Stable* |
| **Hypothesis 2** | | | | | |
| ↳ Multiple LogReg | 0.658 | 0.5286 | 0.264 | 0.6461 | *Moderate* |
| ↳ LDA | 0.675 | 0.4967 | 0.261 | 0.6428 | *Failed* |
| ↳ **Random Forest** | **0.743** | **0.5345** | **0.325** | **0.7167** | **Best** |
| **Hypothesis 3** | | | | | |
| ↳ Logistic Regression | 0.682 | 0.7026 | 0.339 | 0.7557 | *Good* |
| ↳ Ridge Regression | 0.771 | 0.3929 | 0.287 | 0.6086 | *Failed* |
| ↳ **Bootstrap Resampling** | **0.682** | **0.7026** | **0.339** | **0.7557** | **Best** |

---

## 🛑 Lessons Learned (Failed Methods)
* **Single-Feature Logistic Regression (H1):** Dropping all contextual variables and relying purely on `CreditScore` failed to cleanly separate classes, yielding a poor ROC AUC ($0.5307$).
* **LDA (H2):** Attempting to squeeze non-linear socioeconomic interactions into a strict linear boundary resulted in a low recall ($0.4967$) and contradictory direction signs
* **Ridge Regression (H3):** Since Ridge is built primarily for continuous tracking (regression) rather than clear-cut risk grouping (classification), it struggled to capture the boundary lines of default patterns under simulated stress test rules ($AUC = 0.6086$).

---


