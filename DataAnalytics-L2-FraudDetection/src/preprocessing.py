"""
preprocessing.py - Data fetching, cleaning, feature engineering,
and scaling utilities for Credit Card Fraud Detection.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


def fetch_and_save_raw_data(output_path: str) -> pd.DataFrame:
    """
    Fetches the benchmark Kaggle Credit Card Fraud Detection dataset
    from OpenML (Data ID 1597) and saves raw CSV to output_path.
    """
    print("Fetching benchmark Credit Card Fraud dataset from OpenML (Data ID 1597)...")
    bunch = fetch_openml(data_id=1597, as_frame=True, parser='auto')
    df = bunch.frame
    
    # Ensure Class is integer (0 or 1)
    df['Class'] = df['Class'].astype(int)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raw dataset saved successfully to {output_path}. Shape: {df.shape}")
    return df


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Loads dataset from local CSV file path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    df = pd.read_csv(data_path)
    df['Class'] = df['Class'].astype(int)
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and engineers features:
    - Derives 'HourOfDay' from 'Time' seconds column: (Time / 3600) % 24
    - Ensures clean column names and data types
    """
    df_clean = df.copy()
    df_clean['Class'] = df_clean['Class'].astype(int)
    
    if 'Time' in df_clean.columns:
        df_clean['HourOfDay'] = (df_clean['Time'] / 3600.0) % 24.0
        
    return df_clean
