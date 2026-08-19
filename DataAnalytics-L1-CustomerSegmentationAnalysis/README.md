# OASIS INFOBYTE Data Analytics Internship
## Level 1 Task 2: Customer Segmentation Analysis (RFM & K-Means Clustering)

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.1.1-013243.svg)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## 📌 Project Overview
This repository contains the complete implementation for **OASIS INFOBYTE Data Analytics Level 1 Task 2: Customer Segmentation Analysis**. The project utilizes the benchmark **UCI Online Retail Dataset** (541,909 raw transactions) to build an end-to-end Machine Learning customer segmentation pipeline incorporating **RFM (Recency, Frequency, Monetary) Feature Engineering**, **`StandardScaler` Normalization**, **Elbow Method & Silhouette Score Optimization**, and **K-Means Unsupervised Clustering**.

---

## 🎯 Business Objective
To group e-commerce customers into distinct behavioral segments based on historical purchasing activity, enabling data-driven marketing customization, customer retention optimization, and targeted revenue growth.

---

## 📊 Dataset Details
- **Dataset Name**: UCI Online Retail Dataset
- **Source**: UCI Machine Learning Repository / Public Mirror
- **Raw URL**: `https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv`
- **Volume**: 541,909 Raw Transactions (526,818 valid records; 4,338 unique customers)
- **Key Columns**: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

---

## 💡 Key Analytical Frameworks

### 1. RFM (Recency, Frequency, Monetary) Analysis
- **Recency ($R$)**: Days elapsed between the customer's last transaction and the dataset snapshot date ($T_{snapshot} - T_{max\_purchase}$). Lower values indicate active customers.
- **Frequency ($F$)**: Count of unique completed orders/invoices per customer. Higher values signify high brand engagement.
- **Monetary ($M$)**: Total cumulative financial spend ($\sum Quantity \times UnitPrice$). Higher values identify key revenue accounts.

### 2. Feature Normalization & K-Means Clustering
- **StandardScaler**: Distance-based clustering algorithms (K-Means using Euclidean Distance $d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum (u_i - v_i)^2}$) are sensitive to feature magnitude differences. Applying a log-transform ($\log(1+x)$) followed by `StandardScaler` ($\mu=0, \sigma=1$) prevents high-dollar monetary attributes from distorting cluster boundaries.
- **Elbow Method & Silhouette Coefficient**: Evaluated $K \in [2, 10]$ clusters to select **$K=4$** based on the inertia bend and silhouette score peak.

---

## 📂 Project Directory Structure

```
DataAnalytics-L1-CustomerSegmentationAnalysis/
│
├── README.md                                 # Comprehensive project report & setup guide
├── requirements.txt                          # Project dependency specification
├── .gitignore                                # Git tracking rules
│
├── data/
│   ├── raw/
│   │   ├── README.md                         # Dataset acquisition notes & column schema
│   │   └── Online_Retail.csv                 # Raw dataset (Downloaded automatically)
│   └── processed/
│       └── customer_rfm_cleaned.csv          # Cleaned RFM metrics table per customer
│
├── notebooks/
│   └── Customer_Segmentation_RFM_KMeans.ipynb # Fully executed Jupyter Notebook
│
├── src/
│   ├── __init__.py                           # Package initializer
│   └── segmentation_utils.py                 # Pipeline data loader, cleaner, scaling & clustering utils
│
└── outputs/
    ├── figures/                              # High-resolution chart exports
    │   ├── elbow_method_curve.png
    │   ├── rfm_cluster_scatter_plots.png
    │   └── cluster_customer_count_distribution.png
    └── tables/                               # Summary table CSV exports
        ├── descriptive_statistics.csv
        ├── clustering_evaluation_metrics.csv
        ├── cluster_profile_summary.csv
        └── cluster_customer_counts.csv
```

---

## 🚀 Installation & Execution Instructions

### Prerequisites
- Python 3.10+ installed on Windows / macOS / Linux.
- `pip` package manager.

### 1. Environment Setup
Navigate to the task folder in PowerShell or Terminal:
```bash
cd OIBSIP/DataAnalytics-L1-CustomerSegmentationAnalysis
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
Run the data utility to fetch data, compute RFM metrics, fit K-Means, and export summary files:
```bash
python -c "from src.segmentation_utils import download_and_load_data, clean_transaction_data, compute_rfm_metrics; df = download_and_load_data(); cleaned = clean_transaction_data(df); compute_rfm_metrics(cleaned)"
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/Customer_Segmentation_RFM_KMeans.ipynb
```
Select `Run All Cells` to view interactive cluster scatter plots and profile tables.

---

## 📊 Cluster Profiling & Results Summary ($K=4$)

| Cluster ID | Segment Name | Customer Share | Recency (Days) | Frequency (Orders) | Monetary Spend (£) | Avg Order Value (£) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **High-Value Champions** | 16.4% | 15.2 | 11.4 | £5,820.45 | £510.50 |
| **1** | **Recent Loyal Buyers** | 37.4% | 35.8 | 4.2 | £1,420.10 | £338.10 |
| **2** | **Dormant / Low-Engagement** | 19.3% | 248.5 | 1.3 | £290.40 | £223.38 |
| **3** | **At-Risk / Slipping** | 26.9% | 92.4 | 2.1 | £680.15 | £323.88 |

---

## 💡 Strategic Marketing Recommendations

### 1. 🏆 High-Value Champions (Cluster 0 - 16.4%)
- **Characteristics**: VIP purchasers with low recency, high frequency, and high monetary spend.
- **Action**: Invite to an exclusive **VIP Royalty Loyalty Tier** with early product access, dedicated phone support, and personalized appreciation gifts. Avoid heavy discounting.

### 2. 💙 Recent Loyal Buyers (Cluster 1 - 37.4%)
- **Characteristics**: Steady, engaged shoppers with recent activity and consistent purchase history.
- **Action**: Implement **Cross-Selling & Product Bundling** campaigns ("Buy 2 Get 1 30% Off") to increase average order values and convert them into Champions.

### 3. ⚠️ At-Risk / Slipping Customers (Cluster 3 - 26.9%)
- **Characteristics**: Formerly active buyers who have not purchased in 3-5 months.
- **Action**: Deploy automated **Win-Back Email Triggers** offering time-sensitive discount codes (e.g. 15% off next order within 7 days) and satisfaction feedback surveys.

### 4. 💤 Dormant / Low-Engagement (Cluster 2 - 19.3%)
- **Characteristics**: Inactive buyers (>200 days recency) with minimal transaction history.
- **Action**: Run low-cost automated email re-engagement campaigns showcasing trending products. Prune permanently unresponsive contacts after 12 months to protect email sender reputation.

---

## ⚠️ Project Limitations
- **Geographic Representation**: Dataset is heavily skewed toward UK domestic customers (~90%).
- **Product Categorization**: Product descriptions are free-text strings without explicit hierarchical taxonomy.

---

## 👨‍💻 Author Section
- **Author**: Bokkasam Sravan
- **Role**: Data Science Intern
- **Program**: OASIS INFOBYTE Data Analytics Internship (OIBSIP)
- **Task**: Level 1 Task 2 - Customer Segmentation Analysis
- **Repository**: `OIBSIP`
- **Folder**: `DataAnalytics-L1-CustomerSegmentationAnalysis/`
