"""
model_utils.py - SMOTE resampling, evaluation metrics, curves,
error analysis, and output saving utilities for Fraud Detection.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve,
    classification_report,
    confusion_matrix
)


def get_project_root():
    """Returns absolute path to project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def apply_smote_resampling(X_train, y_train, random_state=42):
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE)
    EXCLUSIVELY to training data to prevent data leakage.
    Returns X_resampled, y_resampled, and class count stats dict.
    """
    orig_counts = pd.Series(y_train).value_counts().to_dict()
    
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    res_counts = pd.Series(y_res).value_counts().to_dict()
    
    stats = {
        'Original_Legit': orig_counts.get(0, 0),
        'Original_Fraud': orig_counts.get(1, 0),
        'Original_Fraud_Ratio (%)': round((orig_counts.get(1, 0) / len(y_train)) * 100, 4),
        'Resampled_Legit': res_counts.get(0, 0),
        'Resampled_Fraud': res_counts.get(1, 0),
        'Resampled_Fraud_Ratio (%)': round((res_counts.get(1, 0) / len(y_res)) * 100, 4)
    }
    
    return X_res, y_res, stats


def evaluate_fraud_classifier(model, X_test, y_test, model_name="Classifier") -> dict:
    """
    Evaluates a trained fraud detection classifier on the natural test set.
    Returns metrics dict, confusion matrix, ROC curve data, and PR curve data.
    """
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    roc_auc = roc_auc_score(y_test, y_prob)
    prec_array, rec_array, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_array, prec_array)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent'], zero_division=0)
    
    metrics = {
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1-Score": round(float(f1), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "PR-AUC": round(float(pr_auc), 4)
    }
    
    curves = {
        'fpr': fpr,
        'tpr': tpr,
        'prec_array': prec_array,
        'rec_array': rec_array
    }
    
    return metrics, cm, report_str, curves


def extract_misclassifications(model, X_test, y_test, n_samples=5) -> pd.DataFrame:
    """
    Extracts top false negative (missed fraud) and false positive test samples.
    """
    y_pred = model.predict(X_test)
    y_test_arr = np.array(y_test)
    
    # False Negatives (Actual Fraud=1, Predicted Legit=0)
    fn_mask = (y_test_arr == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0][:n_samples]
    
    # False Positives (Actual Legit=0, Predicted Fraud=1)
    fp_mask = (y_test_arr == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0][:n_samples]
    
    records = []
    for idx in fn_indices:
        records.append({
            'Error_Type': 'False Negative (Missed Fraud)',
            'Actual_Class': 1,
            'Predicted_Class': 0,
            'Amount': X_test.iloc[idx]['Amount'] if isinstance(X_test, pd.DataFrame) and 'Amount' in X_test.columns else 'N/A'
        })
    for idx in fp_indices:
        records.append({
            'Error_Type': 'False Positive (False Alarm)',
            'Actual_Class': 0,
            'Predicted_Class': 1,
            'Amount': X_test.iloc[idx]['Amount'] if isinstance(X_test, pd.DataFrame) and 'Amount' in X_test.columns else 'N/A'
        })
        
    return pd.DataFrame(records)


def save_model(pipeline, filename="best_fraud_detection_model.joblib", base_dir=None):
    """Saves trained model pipeline to models/ directory using joblib."""
    if base_dir is None:
        base_dir = get_project_root()
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(pipeline, filepath)
    print(f"Trained model pipeline saved to: {filepath}")
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
