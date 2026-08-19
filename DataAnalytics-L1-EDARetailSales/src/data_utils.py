"""
data_utils.py - Utility functions for EDA on Retail Sales Data
Includes data loading, automated acquisition, quality inspection,
cleaning, feature engineering, descriptive statistics, and output export.
"""

import os
import sys
import warnings
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATASET_URL = "https://raw.githubusercontent.com/hatimh53/Retail-Sales-Analysis/main/retail_sales_dataset.csv"

def get_project_root():
    """Returns absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def download_and_load_dataset(base_dir=None, raw_filename="retail_sales_dataset.csv", url=DATASET_URL):
    """
    Downloads the dataset from the specified URL if not locally present in data/raw/
    and returns a pandas DataFrame.
    """
    if base_dir is None:
        base_dir = get_project_root()
    
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, raw_filename)
    
    if not os.path.exists(raw_filepath):
        print(f"Dataset not found locally. Downloading from {url} ...")
        urllib.request.urlretrieve(url, raw_filepath)
        print(f"Dataset successfully downloaded and saved to {raw_filepath}")
    else:
        print(f"Loading dataset from local cache: {raw_filepath}")
        
    df = pd.read_csv(raw_filepath)
    return df

def get_initial_inspection(df):
    """
    Performs initial data quality and structure inspection.
    Returns a dictionary of metrics and a DataFrame summary of missing values and data types.
    """
    info_dict = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns)
    }
    
    missing_summary = pd.DataFrame({
        "Data_Type": df.dtypes,
        "Missing_Values": df.isnull().sum(),
        "Missing_Percentage": (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    return info_dict, missing_summary

def clean_retail_data(df, base_dir=None):
    """
    Cleans raw retail data:
    - Removes completely empty / order_id null records
    - Removes duplicate rows
    - Parses order_date to datetime
    - Filters out impossible outliers (negative/excessive age, negative/excessive quantity)
    - Imputes remaining missing values with median/mode values
    - Engineers time-series, demographic, and financial features
    - Saves cleaned dataset to data/processed/retail_sales_cleaned.csv
    """
    cleaned_df = df.copy()
    
    # 1. Drop rows missing primary identifiers (order_id, order_date)
    cleaned_df = cleaned_df.dropna(subset=["order_id", "order_date"]).reset_index(drop=True)
    
    # 2. Drop exact duplicate records
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    dropped_dups = initial_rows - len(cleaned_df)
    
    # 3. Parse Order Date
    cleaned_df["order_date"] = pd.to_datetime(cleaned_df["order_date"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["order_date"]).reset_index(drop=True)
    cleaned_df["year"] = cleaned_df["order_date"].dt.year
    cleaned_df["month"] = cleaned_df["order_date"].dt.month
    cleaned_df["year_month"] = cleaned_df["order_date"].dt.to_period("M").astype(str)
    cleaned_df["quarter"] = cleaned_df["order_date"].dt.to_period("Q").astype(str)
    cleaned_df["day_of_week"] = cleaned_df["order_date"].dt.day_name()

    # 4. Filter or clean numeric anomalies (outliers/erroneous data)
    # Age range validation (15 to 90 years old)
    valid_age_mask = (cleaned_df["age"] >= 15) & (cleaned_df["age"] <= 90)
    median_age = cleaned_df.loc[valid_age_mask, "age"].median()
    cleaned_df["age"] = np.where(valid_age_mask, cleaned_df["age"], median_age)
    cleaned_df["age"] = cleaned_df["age"].fillna(median_age).astype(int)
    
    # Age group binning
    bins = [0, 25, 35, 50, 65, 100]
    labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
    cleaned_df["age_group"] = pd.cut(cleaned_df["age"], bins=bins, labels=labels, right=True)

    # Quantity range validation (1 to 50 items)
    valid_qty_mask = (cleaned_df["quantity"] > 0) & (cleaned_df["quantity"] <= 50)
    median_qty = cleaned_df.loc[valid_qty_mask, "quantity"].median()
    cleaned_df["quantity"] = np.where(valid_qty_mask, cleaned_df["quantity"], median_qty)
    cleaned_df["quantity"] = cleaned_df["quantity"].fillna(median_qty).astype(int)

    # Discount pct handling
    cleaned_df["discount_pct"] = cleaned_df["discount_pct"].fillna(0.0)

    # Unit price & recalculation of Sales Amount & Profit
    cleaned_df["unit_price"] = cleaned_df["unit_price"].fillna(cleaned_df["unit_price"].median())
    
    # Recalculate Sales Amount = quantity * unit_price * (1 - discount_pct)
    cleaned_df["sales_amount"] = (cleaned_df["unit_price"] * cleaned_df["quantity"] * (1 - cleaned_df["discount_pct"])).round(2)
    
    # Profit validation (if missing or invalid ratio, estimate margin)
    if "profit" in cleaned_df.columns:
        mean_margin = 0.20  # 20% default margin
        cleaned_df["profit"] = cleaned_df["profit"].fillna(cleaned_df["sales_amount"] * mean_margin).round(2)
        # Cap unrealistically inflated profits > sales_amount
        cleaned_df["profit"] = np.where(cleaned_df["profit"] > cleaned_df["sales_amount"], cleaned_df["sales_amount"] * 0.35, cleaned_df["profit"])

    cleaned_df["profit_margin_pct"] = np.where(
        cleaned_df["sales_amount"] > 0,
        (cleaned_df["profit"] / cleaned_df["sales_amount"]) * 100,
        0.0
    ).round(2)

    # Customer satisfaction & Shipping
    if "customer_satisfaction" in cleaned_df.columns:
        mean_sat = cleaned_df["customer_satisfaction"].dropna().mean()
        cleaned_df["customer_satisfaction"] = cleaned_df["customer_satisfaction"].fillna(round(mean_sat, 1))

    if "days_to_ship" in cleaned_df.columns:
        valid_ship_mask = (cleaned_df["days_to_ship"] >= 0) & (cleaned_df["days_to_ship"] <= 30)
        median_ship = cleaned_df.loc[valid_ship_mask, "days_to_ship"].median()
        cleaned_df["days_to_ship"] = np.where(valid_ship_mask, cleaned_df["days_to_ship"], median_ship)
        cleaned_df["days_to_ship"] = cleaned_df["days_to_ship"].fillna(median_ship).astype(int)

    # 5. Handle Categorical Missing Values
    cat_cols = ["gender", "region", "city", "product_category", "product_name", "payment_method", "order_status", "return_flag"]
    for col in cat_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(object).fillna("Unknown")

    # 6. Save Cleaned Data
    if base_dir is None:
        base_dir = get_project_root()
    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    proc_filepath = os.path.join(proc_dir, "retail_sales_cleaned.csv")
    cleaned_df.to_csv(proc_filepath, index=False)
    print(f"Cleaned data saved to {proc_filepath} (Rows: {len(cleaned_df)})")
    
    return cleaned_df

def calculate_descriptive_stats(df, num_cols):
    """
    Computes summary descriptive statistics for numerical columns:
    Mean, Median, Mode, Standard Deviation, Min, Max, Skewness.
    """
    stats_list = []
    for col in num_cols:
        if col in df.columns:
            col_series = df[col].dropna()
            mode_val = col_series.mode()[0] if not col_series.mode().empty else np.nan
            stats_list.append({
                "Column": col,
                "Mean": round(float(col_series.mean()), 2),
                "Median": round(float(col_series.median()), 2),
                "Mode": round(float(mode_val), 2),
                "Std_Dev": round(float(col_series.std()), 2),
                "Min": round(float(col_series.min()), 2),
                "Max": round(float(col_series.max()), 2),
                "Skewness": round(float(col_series.skew()), 2)
            })
    return pd.DataFrame(stats_list)

def save_figure(fig, filename, base_dir=None):
    """
    Saves a Matplotlib/Seaborn figure to outputs/figures/ directory.
    """
    if base_dir is None:
        base_dir = get_project_root()
    fig_dir = os.path.join(base_dir, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {filepath}")

def save_table(df_table, filename, base_dir=None):
    """
    Saves a summary pandas table to outputs/tables/ directory.
    """
    if base_dir is None:
        base_dir = get_project_root()
    tab_dir = os.path.join(base_dir, "outputs", "tables")
    os.makedirs(tab_dir, exist_ok=True)
    filepath = os.path.join(tab_dir, filename)
    df_table.to_csv(filepath, index=False)
    print(f"Table saved to: {filepath}")
