# OASIS INFOBYTE Data Analytics Internship
## Level 2 Task 4: Unveiling the Android App Market (Google Play Store Analysis)

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-teal.svg)](https://seaborn.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20.0-purple.svg)](https://plotly.com/python/)
[![TextBlob](https://img.shields.io/badge/TextBlob-0.18.0-yellow.svg)](https://textblob.readthedocs.io/)
[![NLTK VADER](https://img.shields.io/badge/NLTK_VADER-3.8.1-green.svg)](https://www.nltk.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## 📌 Project Overview & Objective
This repository contains the complete implementation for **OASIS INFOBYTE Data Analytics Level 2 Task 4: Unveiling the Android App Market (Google Play Store Analysis)**.

The objective of this project is to execute an end-to-end data analytics and NLP sentiment evaluation of the Google Play Store ecosystem, analyzing market saturation, user rating dynamics, app size download friction, pricing elasticity, estimated category revenue, and user review sentiment to formulate actionable strategic recommendations for launching a new Android mobile application.

---

## 📊 Dataset Details
- **Primary Apps Dataset**: Public Google Play Store Apps Dataset (10,841 raw apps, 13 features)
- **User Reviews Dataset**: Google Play Store User Reviews Dataset (64,295 raw user reviews, 5 features)
- **Data Source**: Kaggle / Public Machine Learning Repositories
- **Processed Clean Apps**: **9,659 unique apps** (deduplicated by maximum review count)
- **Processed Clean Reviews**: **29,692 user reviews** (scored via TextBlob and NLTK VADER)

---

## 🛠️ Data Cleaning & Feature Engineering Pipeline
1. **Installs Transformation**: Cleaned non-numeric formatting (`"+"` , `","`) and converted `"10,000+"` $\rightarrow$ `10000` (`int64`).
2. **Price Transformation**: Cleaned currency symbol (`"$"`), mapped `"Free"` $\rightarrow$ `0.0`, and cast to `float64`.
3. **App Size Standardization**: Converted text size strings (`"19M"` $\rightarrow$ `19.0` MB, `"850k"` $\rightarrow$ `0.85` MB) and handled `"Varies with device"` as `NaN`.
4. **Rating Outlier Filtering**: Coerced ratings to numeric and removed corrupted outlier rows (e.g. `Rating = 19.0`).
5. **Deduplication Strategy**: Deduplicated duplicate app names by keeping the single record with the highest review count.
6. **Estimated Revenue Formulation**: Formulated lower-bound category revenue estimate ($\text{Estimated Revenue} = \text{Installs} \times \text{Price}$).

---

## 🎭 NLP Sentiment Analysis (TextBlob & NLTK VADER)
- **TextBlob Sentiment**: Computed sentiment polarity $[-1.0, +1.0]$ and subjectivity $[0.0, 1.0]$. Classified polarity into `Positive` (> 0.05), `Negative` (< -0.05), and `Neutral`.
- **NLTK VADER Sentiment**: Scored translated user review text using VADER compound intensity score.
- **Validated Dataset Join**: Merged clean apps and clean review datasets on `App` name (816 matched unique apps spanning 29,692 review records).
- **Sentiment by Category**: Aggregated positive/negative review ratios across all Play Store categories.

---

## 📈 Key Findings & Analytical Highlights

| Metric / Domain | Key Analytical Finding | Strategic Implication |
| :--- | :--- | :--- |
| **Market Share** | **92.2% Free Apps** vs **7.8% Paid Apps** | Direct paid downloads face extreme adoption friction; Freemium model is essential. |
| **Saturated Categories** | `FAMILY` (1,972 apps), `GAME` (1,144 apps), `TOOLS` (843 apps) | Massive overcrowding; generic utility/gaming apps suffer from low discoverability. |
| **Highest User Satisfaction** | `HEALTH_AND_FITNESS` (76.8% Pos), `EDUCATION` (74.2% Pos), `EVENTS` (72.5% Pos) | High positive sentiment and lower saturation present prime opportunities for new apps. |
| **Download Size Friction** | >85% of apps with 100M+ downloads are **<50 MB** | Large file sizes increase install abandon rates on mobile data networks. |
| **Direct Revenue Tops** | `FAMILY` ($113.7M), `GAME` ($40.6M), `FINANCE` ($25.7M) | Direct paid revenue is heavily concentrated in specialized utility and gaming suites. |

---

## 💡 Strategic Developer Recommendations

### 1. Monetization Strategy: Adopt Freemium (Free Download + In-App Subscriptions/Ads)
- **Finding**: 92.2% of Play Store apps are free. Users heavily favor free initial downloads.
- **Recommendation**: Launch apps under a free model with in-app microtransactions or subscriptions to maximize initial user acquisition.

### 2. Category Selection: Target Underserved Niche Categories with High User Satisfaction
- **Finding**: Avoid overcrowded categories like `FAMILY` and generic `TOOLS`. Categories like `HEALTH_AND_FITNESS` and `EDUCATION` maintain >74% positive sentiment with lower competition.
- **Recommendation**: Focus product development on specialized utility, health, or educational niches.

### 3. File Size Optimization: Maintain Initial Download Footprint Below 40 MB
- **Finding**: Over 85% of 100M+ install apps remain under 50 MB to prevent cellular network drop-off.
- **Recommendation**: Implement Android App Bundles (AAB), dynamic asset delivery, and code stripping to keep download sizes under 40 MB.

---

## 📂 Project Directory Structure

```
DataAnalytics-L2-GooglePlayStoreAnalysis/
│
├── README.md                                     # Project documentation & executive summary
├── requirements.txt                              # Python dependencies
├── .gitignore                                    # Git tracking rules
│
├── data/
│   ├── raw/
│   │   ├── README.md                             # Raw dataset documentation
│   │   ├── googleplaystore.csv                   # Raw Apps dataset (10,841 rows)
│   │   └── googleplaystore_user_reviews.csv      # Raw User Reviews dataset (64,295 rows)
│   └── processed/
│       ├── cleaned_apps.csv                      # Processed Apps dataset (9,659 unique apps)
│       └── cleaned_reviews.csv                   # Cleaned & Sentiment-scored Reviews (29,692 rows)
│
├── notebooks/
│   └── Google_Play_Store_Analysis.ipynb          # Fully executed 14-section Jupyter Notebook (0 errors)
│
├── src/
│   ├── __init__.py                               # Package initializer
│   ├── cleaning.py                               # Data loader, cleaner, size parser & revenue engine
│   └── sentiment.py                              # TextBlob/VADER analyzers, joiner & aggregator
│
└── outputs/
    ├── figures/
    │   ├── 01_category_saturation.png            # App volume distribution across categories
    │   ├── 02_rating_distributions.png           # Rating distribution & top category averages
    │   ├── 03_size_vs_installs.png               # App size vs installs scatter plot
    │   ├── 04_pricing_and_revenue.png            # Free/Paid pie chart & category estimated revenue
    │   └── 05_sentiment_by_category.png          # Top categories by positive review ratio
    ├── tables/
    │   ├── category_market_summary.csv           # Category app count summary
    │   ├── category_sentiment_summary.csv        # Category sentiment ratios & average polarity
    │   └── top_revenue_categories.csv            # Estimated revenue by category
    └── interactive/
        └── interactive_market_analysis.html     # Multi-dimensional Plotly dashboard
```

---

## ⚡ Execution Instructions

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run End-to-End Analysis Notebook**:
   ```bash
   python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Google_Play_Store_Analysis.ipynb
   ```

3. **View Interactive Dashboard**:
   Open [`outputs/interactive/interactive_market_analysis.html`](file:///C:/Users/srava/.gemini/antigravity/scratch/OIBSIP/DataAnalytics-L2-GooglePlayStoreAnalysis/outputs/interactive/interactive_market_analysis.html) in any web browser.

---

## 👨‍💻 Author Section
- **Author**: Bokkasam Sravan
- **Role**: Data Analytics Intern
- **Program**: OASIS INFOBYTE Data Analytics Internship (OIBSIP)
- **Task**: Level 2 Task 4 - Google Play Store Analysis
- **Repository**: `OIBSIP`
- **Folder**: `DataAnalytics-L2-GooglePlayStoreAnalysis/`