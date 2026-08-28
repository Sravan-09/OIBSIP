"""
sentiment.py - TextBlob and VADER sentiment analysis, join validation,
and category-level sentiment aggregation utilities.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER lexicon quietly
try:
    nltk.download('vader_lexicon', quiet=True)
    _sia = SentimentIntensityAnalyzer()
except Exception:
    _sia = None


def compute_textblob_sentiment(text: str):
    """Returns (polarity, subjectivity, sentiment_label) using TextBlob."""
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0, 'Neutral'
    blob = TextBlob(text)
    pol = float(blob.sentiment.polarity)
    subj = float(blob.sentiment.subjectivity)
    
    if pol > 0.05:
        label = 'Positive'
    elif pol < -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
        
    return round(pol, 4), round(subj, 4), label


def analyze_sentiment_textblob(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """Computes TextBlob polarity, subjectivity, and sentiment label."""
    df_out = df_reviews.copy()
    if 'Translated_Review' not in df_out.columns or df_out.empty:
        return df_out.assign(TextBlob_Polarity=0.0, TextBlob_Subjectivity=0.0, TextBlob_Sentiment='Neutral')
        
    tb_results = df_out['Translated_Review'].apply(compute_textblob_sentiment)
    
    df_out['TextBlob_Polarity'] = [r[0] for r in tb_results]
    df_out['TextBlob_Subjectivity'] = [r[1] for r in tb_results]
    df_out['TextBlob_Sentiment'] = [r[2] for r in tb_results]
    
    return df_out


def compute_vader_sentiment(text: str):
    """Returns (compound_score, sentiment_label) using NLTK VADER."""
    if _sia is None or not isinstance(text, str) or not text.strip():
        return 0.0, 'Neutral'
    scores = _sia.polarity_scores(text)
    compound = float(scores['compound'])
    
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
        
    return round(compound, 4), label


def analyze_sentiment_vader(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """Computes NLTK VADER compound score and sentiment label."""
    df_out = df_reviews.copy()
    if 'Translated_Review' not in df_out.columns or df_out.empty:
        return df_out.assign(VADER_Compound=0.0, VADER_Sentiment='Neutral')
        
    vader_results = df_out['Translated_Review'].apply(compute_vader_sentiment)
    
    df_out['VADER_Compound'] = [r[0] for r in vader_results]
    df_out['VADER_Sentiment'] = [r[1] for r in vader_results]
    
    return df_out


def merge_apps_and_reviews(df_apps_clean: pd.DataFrame, df_reviews_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a validated inner join between cleaned apps and cleaned reviews
    on the 'App' column. Reports match counts and coverage ratios.
    """
    unique_apps = set(df_apps_clean['App']) if 'App' in df_apps_clean.columns else set()
    unique_review_apps = set(df_reviews_clean['App']) if 'App' in df_reviews_clean.columns else set()
    matched_apps = unique_apps.intersection(unique_review_apps)
    
    print(f"Total Unique Apps in Apps Dataset    : {len(unique_apps):,}")
    print(f"Total Unique Apps in Reviews Dataset : {len(unique_review_apps):,}")
    print(f"Matched Apps Count                   : {len(matched_apps):,}")
    
    cols_to_use = [c for c in ['App', 'Category', 'Rating', 'Installs', 'Price', 'Type', 'Estimated_Revenue'] if c in df_apps_clean.columns]
    merged_df = pd.merge(
        df_reviews_clean,
        df_apps_clean[cols_to_use],
        on='App',
        how='inner'
    )
    
    print(f"Merged Dataset Total Review Records  : {merged_df.shape[0]:,}")
    return merged_df


def aggregate_sentiment_by_category(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sentiment distribution (% Positive, % Negative, % Neutral)
    and average sentiment polarity by Category.
    """
    sentiment_col = 'TextBlob_Sentiment' if 'TextBlob_Sentiment' in merged_df.columns else ('Sentiment' if 'Sentiment' in merged_df.columns else None)
    polarity_col = 'TextBlob_Polarity' if 'TextBlob_Polarity' in merged_df.columns else ('Sentiment_Polarity' if 'Sentiment_Polarity' in merged_df.columns else None)
    
    if merged_df.empty or 'Category' not in merged_df.columns or sentiment_col is None:
        return pd.DataFrame(columns=['Category', 'Total_Reviews', 'Positive_Reviews', 'Negative_Reviews', 'Neutral_Reviews', 'Positive_Ratio (%)', 'Negative_Ratio (%)', 'Neutral_Ratio (%)', 'Avg_Polarity'])
        
    cat_group = merged_df.groupby('Category')
    
    records = []
    for cat, group in cat_group:
        total_revs = len(group)
        pos_count = (group[sentiment_col] == 'Positive').sum()
        neg_count = (group[sentiment_col] == 'Negative').sum()
        neu_count = (group[sentiment_col] == 'Neutral').sum()
        
        avg_pol = group[polarity_col].mean() if polarity_col in group.columns else 0.0
        
        records.append({
            'Category': cat,
            'Total_Reviews': total_revs,
            'Positive_Reviews': pos_count,
            'Negative_Reviews': neg_count,
            'Neutral_Reviews': neu_count,
            'Positive_Ratio (%)': round((pos_count / total_revs) * 100, 2),
            'Negative_Ratio (%)': round((neg_count / total_revs) * 100, 2),
            'Neutral_Ratio (%)': round((neu_count / total_revs) * 100, 2),
            'Avg_Polarity': round(avg_pol, 4)
        })
        
    res_df = pd.DataFrame(records).sort_values(by='Positive_Ratio (%)', ascending=False).reset_index(drop=True)
    return res_df
