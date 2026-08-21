"""
Data preprocessing, pipeline construction, and evaluation utilities
for House Price Prediction (Oasis Infobyte Level 2 Task 1).
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score


def fetch_and_save_raw_data(output_path: str) -> pd.DataFrame:
    """
    Fetch the benchmark Ames House Prices dataset from OpenML
    and save the raw CSV to output_path.
    """
    print("Fetching raw Ames Housing dataset from OpenML...")
    bunch = fetch_openml(name='house_prices', as_frame=True, parser='auto')
    df = bunch.frame
    
    if 'SalePrice' in df.columns:
        df['SalePrice'] = df['SalePrice'].astype(float)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raw data saved successfully to {output_path}. Shape: {df.shape}")
    return df


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load dataset from CSV file path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    return pd.read_csv(data_path)


def clean_house_data(df: pd.DataFrame, missing_threshold: float = 0.80) -> pd.DataFrame:
    """
    Clean the dataset:
    - Drop 'Id' column if present.
    - Drop columns exceeding missing_threshold (e.g. PoolQC, MiscFeature, Alley, Fence).
    - Create domain features (HouseAge, RemodAge).
    """
    df_clean = df.copy()
    
    if 'Id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Id'])
        
    if 'SalePrice' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['SalePrice'])
        
    null_ratios = df_clean.isnull().mean()
    high_null_cols = null_ratios[null_ratios > missing_threshold].index.tolist()
    if high_null_cols:
        print(f"Dropping columns with >{missing_threshold*100:.0f}% missing values: {high_null_cols}")
        df_clean = df_clean.drop(columns=high_null_cols)
        
    if 'YrSold' in df_clean.columns and 'YearBuilt' in df_clean.columns:
        df_clean['HouseAge'] = (df_clean['YrSold'] - df_clean['YearBuilt']).clip(lower=0)
    if 'YrSold' in df_clean.columns and 'YearRemodAdd' in df_clean.columns:
        df_clean['RemodAge'] = (df_clean['YrSold'] - df_clean['YearRemodAdd']).clip(lower=0)
        
    return df_clean


def build_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build a leak-free scikit-learn ColumnTransformer for numerical and categorical features.
    - Numeric: Median Imputation -> Standard Scaling
    - Categorical: Constant 'Missing' Imputation -> One-Hot Encoding (drop='first' to prevent dummy variable trap)
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def evaluate_regression_model(model, X_train, y_train, X_test, y_test) -> dict:
    """
    Evaluate regression model performance on train and test sets.
    Returns dictionary with MSE, RMSE, R2.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)
    
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    
    return {
        'Train MSE': mse_train,
        'Train RMSE': rmse_train,
        'Train R2': r2_train,
        'Test MSE': mse_test,
        'Test RMSE': rmse_test,
        'Test R2': r2_test
    }
