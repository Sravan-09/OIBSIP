"""
model_utils.py - Model building, evaluation, coefficient analysis,
error analysis, and output saving utilities for House Price Prediction.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def get_project_root():
    """Returns absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def evaluate_regression_model(model, X_train, y_train, X_test, y_test, model_name="Linear Regression") -> dict:
    """
    Evaluates a regression model on train and test sets.
    Returns metrics dictionary with MSE, RMSE, R2.
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
        "Model": model_name,
        "Train MSE": round(float(mse_train), 2),
        "Train RMSE": round(float(rmse_train), 2),
        "Train R2": round(float(r2_train), 4),
        "Test MSE": round(float(mse_test), 2),
        "Test RMSE": round(float(rmse_test), 2),
        "Test R2": round(float(r2_test), 4)
    }


def extract_top_coefficients(model, feature_names: list, top_pos=8, top_neg=7) -> pd.DataFrame:
    """
    Extracts top positive and top negative standardized coefficients.
    Returns DataFrame containing feature names and coefficient values.
    """
    if hasattr(model, 'coef_'):
        coefs = model.coef_
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('regressor'), 'coef_'):
        coefs = model.named_steps['regressor'].coef_
    else:
        raise ValueError("Model does not expose coef_ attribute.")
        
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs
    })
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    
    pos_top = coef_df.sort_values(by='Coefficient', ascending=False).head(top_pos)
    neg_top = coef_df.sort_values(by='Coefficient', ascending=True).head(top_neg)
    
    top_combined = pd.concat([pos_top, neg_top]).sort_values(by='Coefficient', ascending=True)
    return top_combined


def extract_largest_residuals(model, X_test, y_test, n_samples=5) -> pd.DataFrame:
    """
    Extracts top n_samples test properties with largest absolute prediction errors.
    Returns DataFrame with Actual Price, Predicted Price, Absolute Error, and Percentage Error.
    """
    y_pred = model.predict(X_test)
    y_test_arr = np.array(y_test)
    
    residuals = y_test_arr - y_pred
    abs_errors = np.abs(residuals)
    pct_errors = (abs_errors / y_test_arr) * 100
    
    df_res = pd.DataFrame({
        'Actual_SalePrice': y_test_arr,
        'Predicted_SalePrice': np.round(y_pred, 2),
        'Residual_Error': np.round(residuals, 2),
        'Abs_Error': np.round(abs_errors, 2),
        'Pct_Error (%)': np.round(pct_errors, 2)
    })
    
    top_errors = df_res.sort_values(by='Abs_Error', ascending=False).head(n_samples)
    return top_errors


def save_model(pipeline, filename="linear_regression_pipeline.joblib", base_dir=None):
    """
    Saves a trained model pipeline to models/ directory using joblib.
    """
    if base_dir is None:
        base_dir = get_project_root()
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(pipeline, filepath)
    print(f"Trained model pipeline saved to: {filepath}")
    return filepath


def save_figure(fig, filename, base_dir=None):
    """
    Saves a Matplotlib figure to outputs/figures/ directory.
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
    Saves a summary table to outputs/tables/ directory.
    """
    if base_dir is None:
        base_dir = get_project_root()
    tab_dir = os.path.join(base_dir, "outputs", "tables")
    os.makedirs(tab_dir, exist_ok=True)
    filepath = os.path.join(tab_dir, filename)
    df_table.to_csv(filepath, index=False)
    print(f"Table saved to: {filepath}")
