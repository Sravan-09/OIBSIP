"""
segmentation_utils.py - Utility functions for Customer Segmentation Analysis (RFM & KMeans)
Includes dataset acquisition, transaction cleaning, RFM feature engineering,
StandardScaler normalization, Elbow & Silhouette diagnostic evaluation,
KMeans clustering, cluster profiling, and output exports.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATASET_URL = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv"

def download_and_load_data(base_dir=None, filename="Online_Retail.csv", url=DATASET_URL):
    """
    Downloads the UCI Online Retail dataset if not locally present in data/raw/
    and returns a pandas DataFrame.
    """
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, filename)
    
    if not os.path.exists(raw_filepath):
        print(f"Dataset not found locally. Downloading from {url} ...")
        urllib.request.urlretrieve(url, raw_filepath)
        print(f"Dataset saved to {raw_filepath}")
    else:
        print(f"Loading dataset from local cache: {raw_filepath}")
        
    df = pd.read_csv(raw_filepath, encoding="latin1")
    return df

def inspect_data(df):
    """
    Performs comprehensive structural inspection of transaction dataset.
    """
    info_dict = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_customer_ids": int(df["CustomerID"].isnull().sum()),
        "cancellation_orders": int(df["InvoiceNo"].astype(str).str.startswith("C").sum()) if "InvoiceNo" in df.columns else 0
    }
    
    missing_table = pd.DataFrame({
        "Data_Type": df.dtypes,
        "Missing_Count": df.isnull().sum(),
        "Missing_Pct": (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    return info_dict, missing_table

def clean_transaction_data(df, base_dir=None):
    """
    Cleans e-commerce transaction data:
    - Drops records without CustomerID
    - Drops cancellation orders (InvoiceNo starting with 'C' or Quantity <= 0)
    - Drops invalid unit prices (UnitPrice <= 0)
    - Drops duplicate rows
    - Converts InvoiceDate to DateTime
    - Calculates TotalSpend = Quantity * UnitPrice
    - Saves cleaned transaction data to data/processed/
    """
    cleaned_df = df.copy()
    
    # 1. Remove missing CustomerID
    cleaned_df = cleaned_df.dropna(subset=["CustomerID"]).reset_index(drop=True)
    cleaned_df["CustomerID"] = cleaned_df["CustomerID"].astype(int).astype(str)
    
    # 2. Filter out cancellations and invalid quantities/prices
    cleaned_df = cleaned_df[~cleaned_df["InvoiceNo"].astype(str).str.startswith("C")]
    cleaned_df = cleaned_df[cleaned_df["Quantity"] > 0]
    cleaned_df = cleaned_df[cleaned_df["UnitPrice"] > 0]
    
    # 3. Drop exact duplicates
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # 4. Parse Invoice Date & calculate total spend
    cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"], format="mixed", errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["InvoiceDate"]).reset_index(drop=True)
    cleaned_df["TotalSpend"] = (cleaned_df["Quantity"] * cleaned_df["UnitPrice"]).round(2)
    
    return cleaned_df

def compute_rfm_metrics(cleaned_df, snapshot_date=None, base_dir=None):
    """
    Calculates Recency, Frequency, and Monetary (RFM) metrics per customer:
    - Recency: Days since last purchase relative to snapshot_date (max date + 1 day)
    - Frequency: Number of unique transaction invoices
    - Monetary: Total sum of spending ($)
    """
    if snapshot_date is None:
        snapshot_date = cleaned_df["InvoiceDate"].max() + pd.Timedelta(days=1)
        
    rfm_df = cleaned_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (snapshot_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalSpend", "sum"),
        AvgOrderValue=("TotalSpend", "mean")
    ).reset_index()
    
    # Round metrics for readability
    rfm_df["Monetary"] = rfm_df["Monetary"].round(2)
    rfm_df["AvgOrderValue"] = rfm_df["AvgOrderValue"].round(2)
    
    # Save cleaned RFM customer table
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    rfm_filepath = os.path.join(proc_dir, "customer_rfm_cleaned.csv")
    rfm_df.to_csv(rfm_filepath, index=False)
    print(f"RFM metrics computed for {len(rfm_df)} customers and saved to {rfm_filepath}")
    
    return rfm_df

def scale_features(rfm_df, features=["Recency", "Frequency", "Monetary"], use_log=True):
    """
    Normalizes/standardizes features using StandardScaler.
    Applies np.log1p log transformation prior to scaling if use_log=True to reduce skewness.
    """
    df_transformed = rfm_df[features].copy()
    
    if use_log:
        df_transformed = np.log1p(df_transformed)
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_transformed)
    
    scaled_df = pd.DataFrame(X_scaled, columns=[f"{col}_Scaled" for col in features])
    return X_scaled, scaler, scaled_df

def evaluate_elbow_and_silhouette(X_scaled, min_k=2, max_k=10, random_state=42):
    """
    Evaluates KMeans clustering across K range [min_k, max_k] using Inertia (SSE) and Silhouette Score.
    """
    results = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X_scaled, labels)
        results.append({
            "K": k,
            "Inertia_SSE": round(inertia, 2),
            "Silhouette_Score": round(sil_score, 4)
        })
    return pd.DataFrame(results)

def fit_kmeans_clustering(X_scaled, n_clusters=4, random_state=42):
    """
    Fits KMeans model on scaled features with specified number of clusters and random_state.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return cluster_labels, kmeans

def compute_cluster_profiles(rfm_df, cluster_labels):
    """
    Computes summary cluster profiles (mean, median, customer count) across RFM metrics.
    Maps cluster labels to intuitive customer segment names based on RFM attributes.
    """
    df_clustered = rfm_df.copy()
    df_clustered["Cluster"] = cluster_labels
    
    # Compute Cluster Means
    profile_mean = df_clustered.groupby("Cluster").agg(
        Customer_Count=("CustomerID", "count"),
        Mean_Recency_Days=("Recency", "mean"),
        Mean_Frequency_Orders=("Frequency", "mean"),
        Mean_Monetary_Spend=("Monetary", "mean"),
        Mean_AvgOrderValue=("AvgOrderValue", "mean")
    ).reset_index()
    
    # Determine segment naming based on relative RFM ranking
    segment_names = {}
    for cluster_id in profile_mean["Cluster"]:
        r = profile_mean.loc[profile_mean["Cluster"] == cluster_id, "Mean_Recency_Days"].values[0]
        f = profile_mean.loc[profile_mean["Cluster"] == cluster_id, "Mean_Frequency_Orders"].values[0]
        m = profile_mean.loc[profile_mean["Cluster"] == cluster_id, "Mean_Monetary_Spend"].values[0]
        
        if m > profile_mean["Mean_Monetary_Spend"].mean() and r < profile_mean["Mean_Recency_Days"].mean():
            segment_names[cluster_id] = "High-Value Champions"
        elif r < profile_mean["Mean_Recency_Days"].mean() and f >= profile_mean["Mean_Frequency_Orders"].median():
            segment_names[cluster_id] = "Recent Loyal Buyers"
        elif r > profile_mean["Mean_Recency_Days"].mean() and m < profile_mean["Mean_Monetary_Spend"].mean():
            segment_names[cluster_id] = "Dormant / Low-Engagement Customers"
        else:
            segment_names[cluster_id] = "At-Risk / Slipping Customers"
            
    df_clustered["Segment_Name"] = df_clustered["Cluster"].map(segment_names)
    profile_mean["Segment_Name"] = profile_mean["Cluster"].map(segment_names)
    
    # Round numerical metrics
    profile_mean["Mean_Recency_Days"] = profile_mean["Mean_Recency_Days"].round(1)
    profile_mean["Mean_Frequency_Orders"] = profile_mean["Mean_Frequency_Orders"].round(1)
    profile_mean["Mean_Monetary_Spend"] = profile_mean["Mean_Monetary_Spend"].round(2)
    profile_mean["Mean_AvgOrderValue"] = profile_mean["Mean_AvgOrderValue"].round(2)
    profile_mean["Percentage_Customers"] = (profile_mean["Customer_Count"] / len(df_clustered) * 100).round(2)
    
    return df_clustered, profile_mean

def save_figure(fig, filename, base_dir=None):
    """
    Saves Matplotlib figure to outputs/figures/ directory.
    """
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fig_dir = os.path.join(base_dir, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {filepath}")

def save_table(df_table, filename, base_dir=None):
    """
    Saves summary table to outputs/tables/ directory.
    """
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tab_dir = os.path.join(base_dir, "outputs", "tables")
    os.makedirs(tab_dir, exist_ok=True)
    filepath = os.path.join(tab_dir, filename)
    df_table.to_csv(filepath, index=False)
    print(f"Table saved to: {filepath}")
