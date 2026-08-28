# OASIS INFOBYTE Data Analytics Internship
## Level 1 Task 3: Cleaning Data (Professional Workflow)

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.1.1-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-blue.svg)](https://scipy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## 📌 Project Overview & Objective
This repository contains the complete implementation for **OASIS INFOBYTE Data Analytics Level 1 Task 3: Cleaning Data**. The objective of this project is to demonstrate a professional-grade data cleaning workflow by converting a raw, deliberately imperfect dataset containing missing values, unstandardized string formats, whitespace defects, and numerical outliers into a clean, analysis-ready dataset.

---

## 📊 Dataset Details
- **Dataset Name**: Titanic Machine Learning Dataset
- **Source**: DataScienceDojo / Kaggle / UCI Mirror
- **Raw URL**: `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
- **Volume**: 891 Passenger Records, 12 Raw Features
- **Key Columns**: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

---

## 🔍 Data Quality Problems Discovered

1. **High Missingness in `Cabin`**: 687 missing cells (**77.10% missingness**).
2. **Moderate Missingness in `Age`**: 177 missing cells (**19.87% missingness**).
3. **Minor Missingness in `Embarked`**: 2 missing cells (**0.22% missingness**).
4. **Formatting Inconsistencies**: Unstripped leading/trailing whitespace in string fields and inconsistent gender casing (`'male'`, `'Female'`).
5. **Extreme Price Outliers in `Fare`**: Extreme luxury ticket prices up to £512.33 distorting numerical distributions.

---

## 🛠️ Data Cleaning Decisions & Rationale

| Quality Problem | Affected Feature | Cleaning Decision & Strategy | Justification / Rationale |
| :--- | :--- | :--- | :--- |
| **Missing Age** | `Age` | Median Imputation by `Pclass` & `Sex` | Preserves demographic differences across ticket classes and gender (e.g. 1st class passengers were older). |
| **Missing Embarked** | `Embarked` | Mode Imputation (`'S'`) | Southampton (`S`) represents >72% of all embarkations; fills 2 missing records accurately. |
| **High Missing Cabin** | `Cabin` | Created `Cabin_Known` (0/1) & Filled `'Unknown'` | Dropping 77% of rows would destroy data; imputing mode would inject massive bias. Flagging preserves signal. |
| **String Formatting** | `Sex`, `Embarked`, `Name` | Stripped whitespace, unified casing, extracted `Title` | Standardizes categorical levels (`'Male'`, `'Female'`) and extracts passenger honorifics (`'Mr'`, `'Mrs'`, `'Miss'`, `'Master'`). |
| **Extreme Outliers** | `Fare` | Upper IQR Capping ($Q_3 + 1.5 \times \text{IQR} = £65.63$) | Caps extreme prices at £65.63 to prevent gradient skew without dropping valid passenger records. |
| **Data Types** | All Columns | Enforced explicit dtypes (`PassengerId` as str, `Age`/`Fare` as float, `Sex`/`Embarked`/`Title` as category) | Ensures memory efficiency and correct downstream analysis behavior. |

---

## 📈 Before-vs-After Data Quality Comparison

| Data Quality Metric | Before Cleaning (Raw Dataset) | After Cleaning (Processed Dataset) | Improvement / Resolution Note |
| :--- | :---: | :---: | :--- |
| **Total Row Count** | 891 | 891 | Preserved 100% of rows (zero record loss) |
| **Total Column Count** | 12 | 14 | Engineered 2 features (`Cabin_Known`, `Title`) |
| **Total Missing Cells** | 866 | 0 | **100% missing value resolution** |
| **Duplicate Rows** | 0 | 0 | Zero duplicate records remaining |
| **String Casing & Whitespace** | Unstripped whitespace & casing variations | Cleaned (`'Male'`, `'Female'`) | Standardized categorical levels |
| **Extreme Outliers (`Fare`)** | Max Fare = £512.33 | Capped Max Fare = £65.63 | Capped IQR upper bound without row deletion |

---

## 📂 Project Directory Structure

```
DataAnalytics-L1-CleaningData/
│
├── README.md                                     # Project documentation & summary report
├── requirements.txt                              # Python package dependencies
├── .gitignore                                    # Git tracking rules
│
├── data/
│   ├── raw/
│   │   ├── README.md                             # Raw dataset schema & source details
│   │   └── raw_titanic.csv                       # Raw dataset (Downloaded automatically)
│   └── processed/
│       └── cleaned_dataset.csv                   # Final analysis-ready cleaned dataset
│
├── notebooks/
│   └── Data_Cleaning_Professional_Workflow.ipynb # Fully executed 17-section Jupyter Notebook
│
├── src/
│   ├── __init__.py                               # Package initializer
│   └── cleaning_utils.py                         # Automated cleaning pipeline helper module
│
└── outputs/
    ├── figures/                                  # PNG chart exports
    │   ├── missing_data_heatmap.png
    │   └── outlier_boxplots.png
    └── tables/                                   # Summary CSV exports
        ├── data_quality_report.csv
        └── before_after_comparison.csv
```

---

## 🚀 Installation & Execution Instructions

### Prerequisites
- Python 3.10+ installed on Windows / macOS / Linux.
- `pip` package manager.

### 1. Environment Setup
Navigate to the task folder in PowerShell or Terminal:
```bash
cd OIBSIP/DataAnalytics-L1-CleaningData
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
Run the custom cleaning utility to download raw data, audit quality, handle missing values, cap outliers, and export the cleaned dataset:
```bash
python -c "from src.cleaning_utils import download_and_load_data, handle_missing_values, standardize_categorical_features, detect_and_treat_outliers_iqr, enforce_data_types, save_cleaned_data; raw = download_and_load_data(); cleaned = enforce_data_types(detect_and_treat_outliers_iqr(standardize_categorical_features(handle_missing_values(raw)))[0]); save_cleaned_data(cleaned)"
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/Data_Cleaning_Professional_Workflow.ipynb
```
Select `Run All Cells` to view the 17-section data cleaning workflow.

---

## 📍 Output Dataset Location
- **Path**: [`data/processed/cleaned_dataset.csv`](file:///C:/Users/srava/.gemini/antigravity/scratch/OIBSIP/DataAnalytics-L1-CleaningData/data/processed/cleaned_dataset.csv)
- **Shape**: 891 Rows, 14 Columns
- **Quality**: 0 Missing Cells, 0 Duplicates, 100% Enforced Data Types

---

## ⚠️ Project Limitations
- **Synthetic/Historical Context**: The dataset reflects 1912 passenger manifests; missing cabin details reflect historical recording gaps rather than modern system errors.

---

## 👨‍💻 Author Section
- **Author**: Bokkasam Sravan
- **Role**: Data Analytics Intern
- **Program**: OASIS INFOBYTE Data Analytics Internship (OIBSIP)
- **Task**: Level 1 Task 3 - Cleaning Data
- **Repository**: `OIBSIP`
- **Folder**: `DataAnalytics-L1-CleaningData/`
