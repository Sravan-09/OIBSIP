"""
model_utils.py - Classification evaluation, confusion matrix,
feature importance, error analysis, and output saving utilities.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def get_project_root():
    """Returns absolute path to project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def evaluate_classifier(model, X_test, y_test, model_name="Classifier", target_names=["Normal/Bad", "Good"]) -> dict:
    """
    Evaluates a trained classifier on test set.
    Returns metrics dict, confusion matrix array, classification report string.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    
    metrics = {
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision (Weighted)": round(float(prec), 4),
        "Recall (Weighted)": round(float(rec), 4),
        "F1-Score (Weighted)": round(float(f1), 4)
    }
    
    return metrics, cm, report_str


def extract_misclassifications(model, X_test, y_test, n_samples=5) -> pd.DataFrame:
    """
    Extracts n_samples misclassified test examples (where actual != predicted).
    Returns DataFrame with Actual class and Predicted class.
    """
    y_pred = model.predict(X_test)
    y_test_arr = np.array(y_test)
    
    mis_mask = (y_test_arr != y_pred)
    mis_indices = np.where(mis_mask)[0]
    
    sample_indices = mis_indices[:n_samples]
    
    if isinstance(X_test, pd.DataFrame):
        X_sub = X_test.iloc[sample_indices].copy()
    else:
        X_sub = pd.DataFrame(X_test[sample_indices])
        
    X_sub['Actual_Quality'] = y_test_arr[sample_indices]
    X_sub['Predicted_Quality'] = y_pred[sample_indices]
    
    return X_sub


def save_model(pipeline, filename="best_wine_quality_model.joblib", base_dir=None):
    """Saves a trained model pipeline to models/ directory."""
    if base_dir is None:
        base_dir = get_project_root()
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(pipeline, filepath)
    print(f"Trained model saved to: {filepath}")
    return filepath


def save_figure(fig, filename, base_dir=None):
    """Saves a Matplotlib figure to outputs/figures/ directory."""
    if base_dir is None:
        base_dir = get_project_root()
    fig_dir = os.path.join(base_dir, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {filepath}")


def save_table(df_table, filename, base_dir=None):
    """Saves a summary table to outputs/tables/ directory."""
    if base_dir is None:
        base_dir = get_project_root()
    tab_dir = os.path.join(base_dir, "outputs", "tables")
    os.makedirs(tab_dir, exist_ok=True)
    filepath = os.path.join(tab_dir, filename)
    df_table.to_csv(filepath, index=False)
    print(f"Table saved to: {filepath}")
