# OASIS INFOBYTE Data Analytics Internship
## Level 2 Task 1: House Price Prediction (Machine Learning & Regression Workflow)

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-blue.svg)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## 📌 Project Overview & Objective
This repository contains the complete implementation for **OASIS INFOBYTE Data Analytics Level 2 Task 1: House Price Prediction**. The objective of this project is to build an end-to-end Machine Learning Regression System that predicts continuous residential property prices (`SalePrice`) using structural, physical, quality, and location-based predictors.

Key Machine Learning & Statistical Pillars:
- **Exploratory Data Analysis (EDA)**: Profiling features, missing value ratios, and target variable distribution.
- **Multicollinearity & Redundancy Mitigation**: Identifying and removing redundant linear combination features (`1stFlrSF`, `2ndFlrSF` vs `GrLivArea`) to preserve Ordinary Least Squares (OLS) stability.
- **Data Leakage Prevention**: Encapsulating numerical scaling (`StandardScaler`) and categorical one-hot encoding (`OneHotEncoder(drop='first')`) in scikit-learn `Pipeline` and `ColumnTransformer` fitted **strictly on 80% training data**.
- **Model Training & Evaluation**: Training Ordinary Least Squares (OLS) Linear Regression alongside regularized L2 (Ridge) and L1 (Lasso) models evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ Score.
- **Diagnostic Residual & Coefficient Analysis**: Evaluating residual homoscedasticity and identifying top positive/negative standardized coefficient price drivers.

---

## 📊 Dataset Details
- **Dataset Name**: Ames Housing Benchmark Dataset (compiled by Dean De Cock)
- **Source URL**: OpenML `house_prices` (`https://www.openml.org/d/42165`)
- **Volume**: 1,460 Property Records
- **Predictors**: 80 explanatory variables describing structural characteristics, location, quality ratings, and amenities.
- **Target Variable**: `SalePrice` (Continuous USD monetary valuation).
- **Target Summary**: Mean = $180,921.20 | Median = $163,000.00 | Range = $34,900.00 to $755,000.00.

---

## 🛠️ Data Preprocessing & Feature Engineering Pipeline
1. **High-Null Feature Removal**: Dropped features exceeding 80% missing data (`PoolQC`, `MiscFeature`, `Alley`, `Fence`).
2. **Structural Imputation**: Imputed missing categorical attributes (e.g., "No Basement", "No Garage") with constant `'Missing'`. Imputed numerical columns with training median values.
3. **Domain Feature Engineering**: Derived `HouseAge` (`YrSold - YearBuilt`) and `RemodAge` (`YrSold - YearRemodAdd`).
4. **Multicollinearity Elimination**: Dropped redundant component columns (`1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `YearBuilt`, `YrSold`, `YearRemodAdd`, `GarageArea`) to ensure well-conditioned design matrices.
5. **One-Hot Encoding**: Applied `OneHotEncoder(drop='first', handle_unknown='ignore')` to convert categorical variables into binary indicators without triggering the dummy variable trap.

---

## 🧮 Feature Pipeline & Mathematical Formulation
Ordinary Least Squares (OLS) minimizes the Residual Sum of Squares (RSS):

$$\text{RSS}(\beta) = \sum_{i=1}^{n} \left( y_i - \left( \beta_0 + \sum_{j=1}^{p} \beta_j X_{ij} \right) \right)^2$$

Coefficient of Determination ($R^2$):

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

- **Data Leakage Prevention**: Integrated into a Scikit-Learn `Pipeline`, fitting imputation medians, scaling means/variances, and one-hot categories **strictly on 80% training data**.

---

## 📈 Model Performance & Evaluation Results

| Model Classifier | Train MSE ($) | Train RMSE ($) | Train $R^2$ | Test MSE ($) | Test RMSE ($) | Test $R^2$ | Best Model Selection |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | **7.67 × 10⁸** | **$27,698.29** | **0.8714** | **9.93 × 10⁸** | **$31,515.35** | **0.8705** | **BEST MODEL (Selected)** |
| **Ridge Regression ($\alpha=10.0$)** | 8.06 × 10⁸ | $28,385.25 | 0.8649 | 1.01 × 10⁹ | $31,732.01 | 0.8687 | Regularized L2 Benchmark |
| **Lasso Regression ($\alpha=100.0$)** | 8.01 × 10⁸ | $28,306.60 | 0.8656 | 9.96 × 10⁸ | $31,553.89 | 0.8702 | Regularized L1 Benchmark |

---

## 🏆 Best Performing Model
- **Selected Model**: **Linear Regression (OLS)**
- **Performance**: Test $R^2 = 0.8705$ (~87.1% variance explained), Test RMSE = $31,515.35.
- **Key Advantages**: High interpretability, perfectly balanced Train vs Test $R^2$ (0.8714 vs 0.8705), indicating zero overfitting and clean generalizability.
- **Model Binary Location**: Persisted to [`models/linear_regression_pipeline.joblib`](file:///C:/Users/srava/.gemini/antigravity/scratch/OIBSIP/DataAnalytics-L2-HousePricePrediction/models/linear_regression_pipeline.joblib).

---

## 🔍 Error Analysis & Residual Highlights
Analyzing prediction errors across the test set revealed key diagnostic insights:
1. **Homoscedasticity across Standard Range**: Residuals scatter randomly around zero with constant variance across the $100k–$300k valuation range.
2. **Luxury Estate Heteroscedasticity**: Higher prediction variance occurs at high valuations (>$400,000) due to custom luxury features not captured in standard linear terms.
3. **Primary Positive Drivers**: `Neighborhood_NoRidge` (+ $50,032), `Neighborhood_StoneBr` (+ $46,480), `GrLivArea` (+ $26,578 / std dev), `OverallQual` (+ $16,378 / std dev).
4. **Primary Negative Drivers**: `BldgType_Twnhs` (- $44,906), `HouseAge` (- $12,328 / std dev), `KitchenQual_TA` (- $29,080).

---

## 🌐 Real-World Applications
1. **Automated Valuation Models (AVM)**: Empower real estate platforms (Zillow, Redfin) to estimate property values instantly.
2. **Mortgage & Underwriting Assessment**: Assist banks and mortgage lenders in validating property collateral values before loan origination.
3. **Property Tax Assessment**: Help municipal tax authorities perform fair property tax assessments based on physical attributes.

---

## 📂 Project Directory Structure

```
DataAnalytics-L2-HousePricePrediction/
│
├── README.md                                     # Project documentation & summary report
├── requirements.txt                              # Python package dependencies
├── .gitignore                                    # Git tracking rules
│
├── data/
│   ├── raw/
│   │   ├── README.md                             # Raw dataset schema & download notes
│   │   └── house_prices_raw.csv                  # Raw benchmark dataset (Downloaded automatically)
│   └── processed/
│       └── house_prices_cleaned.csv              # Cleaned dataset artifact
│
├── notebooks/
│   └── House_Price_Linear_Regression.ipynb       # Fully executed 16-section Jupyter Notebook
│
├── src/
│   ├── __init__.py                               # Package initializer
│   ├── preprocessing.py                          # Data loading & preprocessing helper module
│   └── model_utils.py                            # Model evaluation, coefficient & viz utilities
│
├── models/
│   └── linear_regression_pipeline.joblib         # Persisted trained model pipeline
│
└── outputs/
    ├── figures/                                  # Visual diagnostic outputs
    │   ├── 01_target_distribution.png
    │   ├── 02_correlation_heatmap.png
    │   ├── 03_actual_vs_predicted.png
    │   ├── 04_residual_plot.png
    │   └── 05_top_coefficients.png
    └── tables/                                   # Tabular outputs
        ├── model_performance_comparison.csv
        └── top_coefficients.csv
```

---

## 🚀 Installation & Execution Instructions

### Prerequisites
- Python 3.10+ installed on Windows / macOS / Linux.
- `pip` package manager.

### 1. Environment Setup
Navigate to the task folder in PowerShell or Terminal:
```bash
cd OIBSIP/DataAnalytics-L2-HousePricePrediction
python -m venv venv
```
Activate virtual environment:
- **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
- **Linux/macOS**: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pipeline Execution
Run the automated preprocessing and model execution pipeline:
```bash
python -c "from src.preprocessing import fetch_and_save_raw_data, clean_house_data; raw = fetch_and_save_raw_data('data/raw/house_prices_raw.csv'); clean_house_data(raw).to_csv('data/processed/house_prices_cleaned.csv', index=False)"
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/House_Price_Linear_Regression.ipynb
```
Select `Run All Cells` to view the full regression analysis notebook.

---

## ⚠️ Project Limitations
- **Linearity Assumption**: Ordinary Least Squares assumes linear relationships between features and target; non-linear algorithms (Random Forest, XGBoost) could capture non-linear feature interactions.
- **Geographic Scope**: Model parameters are calibrated specifically for Ames, Iowa property data and require re-calibration for other geographic regions.

---

## 👨‍💻 Author Section
- **Author**: Bokkasam Sravan
- **Role**: Data Analytics Intern
- **Program**: OASIS INFOBYTE Data Analytics Internship (OIBSIP)
- **Task**: Level 2 Task 1 - House Price Prediction
- **Repository**: `OIBSIP`
- **Folder**: `DataAnalytics-L2-HousePricePrediction/`
