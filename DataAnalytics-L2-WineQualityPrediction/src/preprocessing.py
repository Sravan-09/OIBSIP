"""
preprocessing.py - Data fetching, cleaning, feature engineering,
and preprocessing functions for Wine Quality Prediction.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def fetch_and_save_raw_data(output_path: str) -> pd.DataFrame:
    """
    Downloads the benchmark UCI Wine Quality (Red) dataset
    and saves the raw CSV to output_path.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    print(f"Fetching raw Wine Quality dataset from UCI Repository ({url})...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    
    # Read semicolon separated CSV
    df = pd.read_csv(output_path, sep=';')
    
    # Clean column names (replace spaces with underscores)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    # Save standard CSV
    df.to_csv(output_path, index=False)
    print(f"Raw dataset saved to {output_path}. Shape: {df.shape}")
    return df


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Loads dataset from local CSV file path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    return pd.read_csv(data_path)


def engineer_quality_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """
    Engineers the target variable:
    Converts discrete quality score (3-8) into binary class:
    - 1 (Good): quality >= threshold (7 or higher)
    - 0 (Normal/Bad): quality < threshold (below 7)
    """
    df_engineered = df.copy()
    df_engineered['quality_binary'] = (df_engineered['quality'] >= threshold).astype(int)
    return df_engineered


def clean_wine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dataset:
    - Standardizes column names
    - Handles missing values if present
    - Drops duplicates if needed
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_')
    
    # Drop rows with missing target or missing features
    df_clean = df_clean.dropna()
    return df_clean
