"""
model_utils.py - Model building, evaluation, WordCloud generation,
error analysis, and output saving utilities for Sentiment Analysis.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_project_root():
    """Returns absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_tfidf_pipeline(classifier, max_features=5000, ngram_range=(1, 2)):
    """
    Constructs a Scikit-Learn Pipeline combining TF-IDF vectorization and a classifier.
    Fitting the pipeline on training data prevents data leakage.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ("clf", classifier)
    ])
    return pipeline

def evaluate_model(pipeline, X_test, y_test, model_name="Classifier", labels=["Positive", "Negative", "Neutral"]):
    """
    Evaluates a trained classifier pipeline on test data.
    Returns evaluation metrics dictionary, confusion matrix array, and predictions.
    """
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    metrics_dict = {
        "Model_Name": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision_Weighted": round(float(prec_weighted), 4),
        "Recall_Weighted": round(float(rec_weighted), 4),
        "F1_Weighted": round(float(f1_weighted), 4),
        "Precision_Macro": round(float(prec_macro), 4),
        "Recall_Macro": round(float(rec_macro), 4),
        "F1_Macro": round(float(f1_macro), 4)
    }
    
    return metrics_dict, cm, y_pred

def generate_wordcloud(text_series, title="WordCloud", color_map="viridis"):
    """
    Generates a WordCloud figure for a pandas text Series.
    Returns the Matplotlib figure object.
    """
    combined_text = " ".join(text_series.dropna().astype(str))
    
    if not combined_text.strip():
        combined_text = "NoWords Available"
        
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=color_map,
        max_words=120,
        random_state=42
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontweight="bold", fontsize=14, pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig

def extract_misclassifications(pipeline, X_test, y_test, orig_text_series, n_samples=5):
    """
    Extracts n_samples misclassified test examples (where actual != predicted).
    Returns a pandas DataFrame with original text, actual sentiment, and predicted sentiment.
    """
    y_pred = pipeline.predict(X_test)
    
    # Identify indices where prediction failed
    mis_mask = (np.array(y_test) != np.array(y_pred))
    mis_indices = np.where(mis_mask)[0]
    
    sample_indices = mis_indices[:n_samples]
    
    mis_list = []
    for idx in sample_indices:
        mis_list.append({
            "Original_Text": orig_text_series.iloc[idx],
            "Actual_Sentiment": y_test.iloc[idx],
            "Predicted_Sentiment": y_pred[idx]
        })
        
    return pd.DataFrame(mis_list)

def save_model(pipeline, filename="best_sentiment_model.joblib", base_dir=None):
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
