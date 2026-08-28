# OASIS INFOBYTE Data Analytics Internship
## Level 1 Task 4: Sentiment Analysis (Machine Learning & NLP Workflow)

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green.svg)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## 📌 Project Overview & Objective
This repository contains the complete implementation for **OASIS INFOBYTE Data Analytics Level 1 Task 4: Sentiment Analysis**. The objective of this project is to build an end-to-end Machine Learning Sentiment Classification System that automatically categorizes text into three sentiment classes:
- **Positive**
- **Negative**
- **Neutral**

---

## 📊 Dataset Details
- **Dataset Name**: Public Twitter Sentiment Corpus
- **Source URL**: `https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv`
- **Volume**: 3,421 Processed Labelled Records (5,113 Raw Records)
- **Target Classes**: `Positive` (518 tweets), `Negative` (572 tweets), `Neutral` (2,331 tweets).

---

## 🛠️ Text Preprocessing Pipeline
1. **Lowercasing**: Unifies vocabulary across letter casing.
2. **URL & Mention Removal**: Strips web links (`http://...`) and Twitter handles (`@username`).
3. **Special Character Stripping**: Removes non-alphabetic punctuation.
4. **Tokenization & Stopword Removal**: Splits text into words and removes high-frequency English stop words (`nltk.corpus.stopwords`).
5. **Lemmatization**: Applies NLTK `WordNetLemmatizer` to reduce tokens to dictionary root forms.

---

## 🧮 TF-IDF Vectorization
**Term Frequency-Inverse Document Frequency (TF-IDF)** converts clean text strings into sparse numerical vectors:
$$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$
- **TF**: Measures word frequency in a specific document.
- **IDF**: Suppresses corpus-wide generic terms while inflating rare, sentiment-bearing keywords (`'amazing'`, `'terrible'`, `'love'`).
- **Data Leakage Prevention**: Integrated into a Scikit-Learn `Pipeline`, fitting TF-IDF vocabulary and weights **strictly on 80% training data**.

---

## 📈 Model Performance & Evaluation Results

| Model Classifier | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Best Model Selection |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 75.91% | 0.7712 | 0.7591 | 0.6974 | Baseline Model |
| **Logistic Regression** | **80.29%** | **0.7854** | **80.29%** | **0.7725** | **BEST MODEL (Selected)** |

---

## 🏆 Best Performing Model
- **Selected Model**: **Logistic Regression** (`max_iter=1000`, `random_state=42`)
- **Key Advantages**: Outperformed Naive Bayes across all metrics (+4.38% Accuracy boost, +7.51% Weighted F1-Score improvement).
- **Model Binary Location**: Persisted to [`models/best_sentiment_model.joblib`](file:///C:/Users/srava/.gemini/antigravity/scratch/OIBSIP/DataAnalytics-L1-SentimentAnalysis/models/best_sentiment_model.joblib).

---

## 🔍 Error Analysis Highlights
Analyzing 5 misclassified test examples revealed key systemic challenges in classical NLP:
1. **Sarcasm & Irony**: Sarcastic positive words (`"great"`, `"wonderful"`) used in negative contexts confuse bag-of-words models.
2. **Negation Context**: Unigrams fail to connect distant negation modifiers (`"not"`) with sentiment words (`"bad"`).
3. **Short Tweets**: 2-word tweets lack sufficient TF-IDF context, defaulting predictions to the majority `Neutral` class.

---

## 🌐 Real-World Applications
1. **Social Media Listening**: Monitor brand mentions on Twitter/X to detect negative publicity spikes in real time.
2. **Support Ticket Routing**: Route high-priority negative customer complaints to human escalation teams automatically.
3. **Market Intelligence**: Analyze e-commerce reviews to identify product defects or customer satisfaction trends.

---

## 📂 Project Directory Structure

```
DataAnalytics-L1-SentimentAnalysis/
│
├── README.md                                     # Project documentation & summary report
├── requirements.txt                              # Python package dependencies
├── .gitignore                                    # Git tracking rules
│
├── data/
│   ├── raw/
│   │   ├── README.md                             # Raw dataset schema & download notes
│   │   └── twitter_sentiment_raw.csv             # Raw dataset (Downloaded automatically)
│   └── processed/
│       └── sentiment_cleaned.csv                 # Cleaned 3-class sentiment dataset
│
├── notebooks/
│   └── Sentiment_Analysis_ML.ipynb               # Fully executed 15-section Jupyter Notebook
│
├── src/
│   ├── __init__.py                               # Package initializer
│   ├── preprocessing.py                          # Text preprocessing helper module
│   └── model_utils.py                            # Model pipeline, evaluation & viz utilities
│
├── models/
│   └── best_sentiment_model.joblib               # Persisted trained model pipeline
│
└── outputs/
    ├── figures/                                  # Visual outputs & WordClouds
    │   ├── confusion_matrices_comparison.png
    │   ├── sentiment_distribution.png
    │   ├── wordcloud_negative.png
    │   ├── wordcloud_neutral.png
    │   └── wordcloud_positive.png
    └── tables/                                   # Tabular outputs
        ├── misclassified_examples.csv
        ├── model_performance_comparison.csv
        └── sentiment_class_distribution.csv
```

---

## 🚀 Installation & Execution Instructions

### Prerequisites
- Python 3.10+ installed on Windows / macOS / Linux.
- `pip` package manager.

### 1. Environment Setup
Navigate to the task folder in PowerShell or Terminal:
```bash
cd OIBSIP/DataAnalytics-L1-SentimentAnalysis
python -m venv venv
```
Activate virtual environment:
- **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
- **Linux/macOS**: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pipeline Execution
Run the automated preprocessing and model training pipeline:
```bash
python -c "from src.preprocessing import download_and_load_raw_data, preprocess_dataset; raw = download_and_load_raw_data(); preprocess_dataset(raw)"
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/Sentiment_Analysis_ML.ipynb
```
Select `Run All Cells` to view the full sentiment classification notebook.

---

## ⚠️ Project Limitations
- **Domain Focus**: Tweets contain informal slangs and abbreviations; performance may vary when transferred to formal medical or financial domains.

---

## 👨‍💻 Author Section
- **Author**: Bokkasam Sravan
- **Role**: Data Analytics Intern
- **Program**: OASIS INFOBYTE Data Analytics Internship (OIBSIP)
- **Task**: Level 1 Task 4 - Sentiment Analysis
- **Repository**: `OIBSIP`
- **Folder**: `DataAnalytics-L1-SentimentAnalysis/`
