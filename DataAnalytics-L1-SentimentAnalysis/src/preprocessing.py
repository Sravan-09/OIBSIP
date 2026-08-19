"""
preprocessing.py - Text preprocessing pipeline for Sentiment Analysis
Includes dataset acquisition, text cleaning, URL/mention stripping,
punctuation removal, tokenization, stopword removal, lemmatization,
and dataset saving.
"""

import os
import re
import urllib.request
import pandas as pd
import numpy as np
import nltk

# Ensure NLTK data resources are downloaded silently
for resource in ["stopwords", "wordnet", "punkt", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

DATASET_URL = "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv"

def get_project_root():
    """Returns absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def download_and_load_raw_data(base_dir=None, filename="twitter_sentiment_raw.csv", url=DATASET_URL):
    """
    Downloads the raw sentiment dataset if not locally present in data/raw/
    and returns a pandas DataFrame.
    """
    if base_dir is None:
        base_dir = get_project_root()
        
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, filename)
    
    if not os.path.exists(raw_filepath):
        print(f"Dataset not found locally. Downloading from {url} ...")
        urllib.request.urlretrieve(url, raw_filepath)
        print(f"Raw dataset saved to {raw_filepath}")
    else:
        print(f"Loading raw dataset from local cache: {raw_filepath}")
        
    df = pd.read_csv(raw_filepath)
    return df

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_single_text(text):
    """
    Cleans a single text string:
    1. Lowercases text
    2. Strips URLs and hyper links
    3. Strips Twitter handles (@username)
    4. Removes non-alphabetical punctuation and special characters
    5. Tokenizes string into words
    6. Removes English stopwords
    7. Applies WordNet Lemmatization
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # 3. Remove user mentions (@user)
    text = re.sub(r"@\w+", "", text)
    
    # 4. Remove punctuation & special characters (keep only letters and spaces)
    text = re.sub(r"[^a-z\s]", "", text)
    
    # 5. Tokenize & remove extra whitespace
    tokens = text.split()
    
    # 6. Stopword removal & 7. Lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]
    
    return " ".join(cleaned_tokens)

def preprocess_dataset(df, base_dir=None):
    """
    Applies the full preprocessing pipeline across the dataset:
    - Filters to 3 target sentiment classes: Positive, Negative, Neutral (removes 'irrelevant')
    - Maps casing to standard Title case ('Positive', 'Negative', 'Neutral')
    - Removes empty/null text records and duplicate rows
    - Applies clean_single_text() to generate 'clean_text' column
    - Saves processed DataFrame to data/processed/sentiment_cleaned.csv
    """
    cleaned_df = df.copy()
    
    # Check column names (Topic, Sentiment, TweetId, TweetDate, TweetText)
    text_col = "TweetText" if "TweetText" in cleaned_df.columns else "text"
    sentiment_col = "Sentiment" if "Sentiment" in cleaned_df.columns else "sentiment"
    
    # Standardize label values & filter out 'irrelevant'
    cleaned_df[sentiment_col] = cleaned_df[sentiment_col].astype(str).str.lower().str.strip()
    valid_mask = cleaned_df[sentiment_col].isin(["positive", "negative", "neutral"])
    cleaned_df = cleaned_df[valid_mask].reset_index(drop=True)
    
    # Capitalize sentiment labels
    label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    cleaned_df["sentiment_label"] = cleaned_df[sentiment_col].map(label_map)
    
    # Drop rows missing text
    cleaned_df = cleaned_df.dropna(subset=[text_col]).reset_index(drop=True)
    
    # Drop exact duplicate text rows
    cleaned_df = cleaned_df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    
    # Apply text cleaning pipeline
    print("Applying text preprocessing pipeline (lowercasing, URL/handle stripping, stopword removal, lemmatization) ...")
    cleaned_df["clean_text"] = cleaned_df[text_col].apply(clean_single_text)
    
    # Filter out empty clean_text records
    cleaned_df = cleaned_df[cleaned_df["clean_text"].str.strip() != ""].reset_index(drop=True)
    
    # Save cleaned dataset
    if base_dir is None:
        base_dir = get_project_root()
    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    proc_filepath = os.path.join(proc_dir, "sentiment_cleaned.csv")
    cleaned_df.to_csv(proc_filepath, index=False)
    print(f"Cleaned dataset saved to {proc_filepath} (Total Records: {len(cleaned_df)})")
    
    return cleaned_df
