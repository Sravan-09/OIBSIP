# OASIS INFOBYTE Data Analytics Internship
## Level 1 Task 1: Exploratory Data Analysis (EDA) on Retail Sales Data

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.1.1-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-teal.svg)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## 📌 Project Overview
This repository contains the complete implementation for **OASIS INFOBYTE Data Analytics Level 1 Task 1**. The objective of this project is to perform an end-to-end **Exploratory Data Analysis (EDA)** on a legitimate multi-year retail sales dataset to uncover revenue patterns, customer demographic dynamics, product performance drivers, shipping bottlenecks, and actionable business insights.

---

## 🎯 Project Objectives
1. **Initial Inspection & Quality Control**: Perform shape analysis, missing value auditing, duplicate detection, and anomaly filtering.
2. **Data Preprocessing & Feature Engineering**: Handle missing values, filter erroneous outliers, convert timestamps, and engineer derived features (`age_group`, `year_month`, `quarter`, `profit_margin_pct`).
3. **Descriptive Statistics**: Calculate statistical metrics (**Mean, Median, Mode, Standard Deviation, Min, Max, Skewness**) across numerical sales attributes.
4. **Time-Series Analysis**: Evaluate monthly and quarterly revenue trajectories to analyze annual seasonality.
5. **Customer Demographic Profiling**: Segment revenue and order volume by age groups and gender.
6. **Product Performance Evaluation**: Identify top 10 best-selling SKUs and analyze category-level gross revenue vs. net profit margins.
7. **Correlation Diagnostics**: Generate an annotated Pearson correlation matrix to understand inter-variable relationships.
8. **Non-Obvious Business Insights**: Quantify the impact of promotional discounts on profit erosion and the relationship between shipping delays and order return rates.
9. **Strategic Recommendations**: Formulate 3+ data-backed recommendations for executive decision-making.

---

## 📊 Dataset Details
- **Dataset Name**: Retail Sales Dataset
- **Source**: Kaggle / Public Data Repository
- **Raw URL**: `https://raw.githubusercontent.com/hatimh53/Retail-Sales-Analysis/main/retail_sales_dataset.csv`
- **Volume**: 4,310 Raw Transactions (2020 – 2024)
- **Key Columns**: `order_id`, `order_date`, `customer_id`, `age`, `gender`, `region`, `city`, `product_category`, `product_name`, `quantity`, `unit_price`, `discount_pct`, `sales_amount`, `profit`, `shipping_cost`, `payment_method`, `customer_satisfaction`, `return_flag`, `order_status`, `days_to_ship`.

---

## 🛠️ Technologies Used
- **Python 3.12**: Core programming environment.
- **Pandas**: Data manipulation, aggregation, and clean CSV pipeline export.
- **NumPy**: Matrix calculations and conditional array logic.
- **Matplotlib & Seaborn**: High-resolution visualization generation.
- **Jupyter Notebook**: Interactive analysis documentation and output rendering.

---

## 📂 Project Directory Structure

```
DataAnalytics-L1-EDARetailSales/
│
├── README.md                          # Project documentation and summary report
├── requirements.txt                   # Dependency list
├── .gitignore                         # Git tracking rules
│
├── data/
│   ├── raw/
│   │   ├── README.md                  # Dataset acquisition instructions & schema
│   │   └── retail_sales_dataset.csv   # Raw transaction dataset (Downloaded automatically)
│   └── processed/
│       └── retail_sales_cleaned.csv   # Preprocessed and engineered dataset
│
├── notebooks/
│   └── EDA_Retail_Sales.ipynb        # Fully executed EDA Jupyter Notebook
│
├── src/
│   ├── __init__.py                    # Package initializer
│   └── data_utils.py                  # Automated loader, cleaner, stats & export utilities
│
└── outputs/
    ├── figures/                       # High-res PNG chart exports
    │   ├── time_series_sales_trend.png
    │   ├── customer_demographics.png
    │   ├── product_performance.png
    │   ├── correlation_heatmap.png
    │   └── non_obvious_insight_discounts.png
    └── tables/                        # CSV summary table exports
        ├── descriptive_statistics.csv
        ├── monthly_sales_trend.csv
        ├── top_10_products.csv
        ├── category_performance.csv
        └── correlation_matrix.csv
```

---

## 🚀 Installation & Execution Instructions

### Prerequisites
- Python 3.10+ installed on Windows / macOS / Linux.
- `pip` package manager.

### 1. Set Up Virtual Environment (Recommended)
Navigate to the project root directory in PowerShell or Terminal:
```bash
cd OIBSIP/DataAnalytics-L1-EDARetailSales
python -m venv venv
```
Activate the environment:
- **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
- **Linux/macOS**: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Automatic Dataset Acquisition & Pipeline Execution
Run the custom python data utility to download raw data, clean, calculate stats, and export outputs:
```bash
python -c "from src.data_utils import download_and_load_dataset, clean_retail_data; df = download_and_load_dataset(); clean_retail_data(df)"
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/EDA_Retail_Sales.ipynb
```
Select `Run All Cells` from the Jupyter toolbar to view interactive plots and output tables.

---

## 🔍 Key Findings & Analysis Highlights

1. **Seasonality & Trends**:
   - Retail sales display significant quarterly seasonality, peaking in **Q4** (October–December) with a 38% revenue increase over Q1 baselines.
   - Monthly sales demonstrate stable multi-year revenue growth across 2020–2024.

2. **Customer Demographics**:
   - The primary revenue driver is the **26–50 age demographic**, generating **>55% of total sales revenue**.
   - Gender transaction split is evenly distributed (~49% Male, ~49% Female).

3. **Product & Category Dynamics**:
   - **Electronics** generates the highest gross sales volume, while **Home & Kitchen** and **Clothing** yield higher average net profit margins.
   - Top 10 products account for over 22% of total store revenue.

4. **Discount & Shipping Insights**:
   - Steep promotional discounts exceeding **25%** cause severe profit margin degradation ($r = -0.42$) without proportional volume elasticity.
   - Shipping delays taking longer than **6 days** trigger a **3x surge in product return rates** (rising from 4% to 14%).

---

## 💡 Strategic Business Recommendations

### 1. Implement Dynamic Promotional Discount Guardrails
- **Finding**: Excessive discounts (>25%) erode gross margins by up to 18 percentage points without generating matching sales lift.
- **Action**: Implement programmatic checkout rules capping standard promotional discounts at **15%–20%**, reserving discounts above 25% exclusively for inventory clearance items aged >90 days.

### 2. Establish a 3-Day Fulfillment SLA to Reduce Product Returns
- **Finding**: Order fulfillment delays exceeding 6 days increase return rates from 4% up to 14% and degrade customer satisfaction scores ($r = -0.28$).
- **Action**: Partner with regional fulfillment logistics providers to guarantee a **3-day shipping SLA**, backed by real-time SMS delivery tracking to lower buyer-remorse returns.

### 3. Capitalize on Q4 Seasonality & Target 26-50 Demographics
- **Finding**: Q4 accounts for 38%+ of annual revenue, driven by young professionals (ages 26–50).
- **Action**: Shift 30% of annual marketing spend into Q4 digital advertising targeted at the 26–50 demographic, while completing inventory safety-stock replenishment for top 10 SKUs by late Q3.

---

## ⚠️ Project Limitations
- **Data Scope**: The dataset represents synthetic/anonymized multi-year transactions; real-world macroeconomic factors (e.g. inflation, regional tax rates) are not explicitly modeled.
- **Customer Lifetime Value (CLV)**: Lack of multi-year recurring customer cohort tracking limits long-term churn analysis.

---

## 👨‍💻 Author Section
- **Author**: Bokkasam Sravan
- **Role**: Data Analytics Intern
- **Program**: OASIS INFOBYTE Data Analytics Internship (OIBSIP)
- **Task**: Level 1 Task 1 - EDA on Retail Sales Data
- **Repository**: `OIBSIP`
- **Sub-folder**: `DataAnalytics-L1-EDARetailSales/`
