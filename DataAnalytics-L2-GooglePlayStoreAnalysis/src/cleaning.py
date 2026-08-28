"""
cleaning.py - Data fetching, cleaning, type conversions, and feature
engineering utilities for the Google Play Store Analysis project.
"""

import os
import io
import shutil
import urllib.request
import pandas as pd
import numpy as np


def fetch_and_save_raw_data(raw_dir: str):
    """
    Ensures raw Google Play Store Apps and User Reviews CSV datasets exist in raw_dir.
    If files exist and are non-empty (>100,000 bytes), retains existing files.
    """
    os.makedirs(raw_dir, exist_ok=True)
    apps_path = os.path.join(raw_dir, "googleplaystore.csv")
    reviews_path = os.path.join(raw_dir, "googleplaystore_user_reviews.csv")
    
    if (os.path.exists(apps_path) and os.path.getsize(apps_path) > 100000 and
        os.path.exists(reviews_path) and os.path.getsize(reviews_path) > 100000):
        print(f"Raw datasets verified at {raw_dir}")
        return
        
    apps_url = "https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv"
    reviews_url = "https://raw.githubusercontent.com/Shubhi550/The-Android-App-Market-on-Google-Play/master/googleplaystore_user_reviews.csv"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        if not os.path.exists(apps_path) or os.path.getsize(apps_path) < 100000:
            req1 = urllib.request.Request(apps_url, headers=headers)
            with urllib.request.urlopen(req1) as resp:
                content = resp.read()
            if len(content) > 100000:
                with open(apps_path, 'wb') as f:
                    f.write(content)
        if not os.path.exists(reviews_path) or os.path.getsize(reviews_path) < 100000:
            req2 = urllib.request.Request(reviews_url, headers=headers)
            with urllib.request.urlopen(req2) as resp:
                content = resp.read()
            if len(content) > 100000:
                with open(reviews_path, 'wb') as f:
                    f.write(content)
        print(f"Raw datasets successfully downloaded to {raw_dir}")
    except Exception as e:
        print(f"Download note: {e}")


def parse_size_in_mb(size_str):
    """Converts size string (e.g. '19M', '850k') into float Megabytes (MB)."""
    if pd.isna(size_str) or size_str == 'Varies with device':
        return np.nan
    size_str = str(size_str).strip()
    if size_str.endswith('M') or size_str.endswith('m'):
        try:
            return float(size_str[:-1])
        except ValueError:
            return np.nan
    elif size_str.endswith('k') or size_str.endswith('K'):
        try:
            return float(size_str[:-1]) / 1024.0
        except ValueError:
            return np.nan
    else:
        try:
            return float(size_str)
        except ValueError:
            return np.nan


def clean_apps_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Apps dataset:
    - Coerces Rating to numeric and filters out corrupted rows (Rating > 5.0)
    - Converts Installs to int64
    - Converts Price to float64
    - Standardizes Size to Megabytes (MB)
    - Converts Reviews to int64
    - Converts Last Updated to datetime
    - Deduplicates by keeping the entry with maximum Reviews per App
    - Computes Estimated_Revenue = Installs * Price
    """
    if df.empty:
        return df
    df_clean = df.copy()
    
    # 1. Clean Rating first to handle corrupted row (e.g. Rating = 19.0)
    if 'Rating' in df_clean.columns:
        df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce')
        df_clean = df_clean[(df_clean['Rating'].isna()) | (df_clean['Rating'] <= 5.0)].copy()
        
    # 2. Clean Installs
    if 'Installs' in df_clean.columns:
        df_clean['Installs'] = (
            df_clean['Installs']
            .astype(str)
            .str.replace('+', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        df_clean['Installs'] = pd.to_numeric(df_clean['Installs'], errors='coerce').fillna(0).astype('int64')
        
    # 3. Clean Price
    if 'Price' in df_clean.columns:
        df_clean['Price'] = (
            df_clean['Price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.strip()
        )
        df_clean['Price'] = pd.to_numeric(df_clean['Price'], errors='coerce').fillna(0.0).astype('float64')
        
    # 4. Clean Size (MB)
    if 'Size' in df_clean.columns:
        df_clean['Size_MB'] = df_clean['Size'].apply(parse_size_in_mb)
        
    # 5. Clean Reviews
    if 'Reviews' in df_clean.columns:
        df_clean['Reviews'] = pd.to_numeric(df_clean['Reviews'], errors='coerce').fillna(0).astype('int64')
        
    # 6. Clean Last Updated
    if 'Last Updated' in df_clean.columns:
        df_clean['Last Updated'] = pd.to_datetime(df_clean['Last Updated'], errors='coerce')
        
    # 7. Deduplicate by App Name (retain record with highest Reviews)
    if 'App' in df_clean.columns and 'Reviews' in df_clean.columns:
        df_clean = df_clean.sort_values(by='Reviews', ascending=False)
        df_clean = df_clean.drop_duplicates(subset=['App'], keep='first').reset_index(drop=True)
        
    # 8. Compute Estimated Revenue (Installs * Price)
    if 'Installs' in df_clean.columns and 'Price' in df_clean.columns:
        df_clean['Estimated_Revenue'] = df_clean['Installs'] * df_clean['Price']
        
    return df_clean


def clean_reviews_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw User Reviews dataset:
    - Drops rows where Translated_Review is missing
    - Cleans whitespace
    - Removes duplicate review records
    """
    if df.empty:
        return df
    df_clean = df.copy()
    
    if 'Translated_Review' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['Translated_Review']).copy()
        df_clean['Translated_Review'] = df_clean['Translated_Review'].astype(str).str.strip()
        df_clean = df_clean[df_clean['Translated_Review'] != ''].copy()
        
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    return df_clean
