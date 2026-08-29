# Raw Dataset Information: Twitter Sentiment Analysis

## Dataset Overview
- **Dataset Name**: Public Twitter Sentiment Corpus
- **Primary Source**: GitHub / Public Sentiment Repository (`zfz/twitter_corpus`)
- **Direct Raw Download URL**: `https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv`
- **Expected Filename**: `twitter_sentiment_raw.csv`
- **Target Location**: `data/raw/twitter_sentiment_raw.csv`

## Dataset Description
This dataset contains 5,113 raw tweets annotated across commercial topics (`apple`, `google`, `microsoft`, `twitter`) and sentiment classifications (`positive`, `negative`, `neutral`, `irrelevant`). It serves as a benchmark text corpus for natural language processing, text cleaning, TF-IDF feature extraction, and multi-class sentiment classification algorithms.

## Data Schema & Attributes
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Topic` | Object (String) | Target commercial topic entity (e.g. `apple`, `google`, `microsoft`, `twitter`). |
| `Sentiment` | Object (String) | Raw sentiment annotation (`positive`, `negative`, `neutral`, `irrelevant`). |
| `TweetId` | Integer / Object | Unique tweet identification key string. |
| `TweetDate` | Object (String) | Timestamp of post creation (e.g. `Sat Sep 24 22:42:33 +0000 2011`). |
| `TweetText` | Object (String) | Raw textual content of the tweet message. |

## Automated Acquisition
The text preprocessing utility in `src/preprocessing.py` automatically downloads `twitter_sentiment_raw.csv` into this directory when `download_and_load_raw_data()` is executed. Manual download is only required if running offline.
