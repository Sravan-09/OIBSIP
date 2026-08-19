# Raw Dataset Documentation: Twitter Sentiment Analysis

## Dataset Details
- **Name**: Public Twitter Sentiment Corpus
- **Source URL**: `https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv`
- **Volume**: 5,113 Raw Records
- **Raw Schema**:
  - `Topic`: Target topic entity (e.g. `apple`, `google`, `microsoft`, `twitter`)
  - `Sentiment`: Label attribute (`positive`, `negative`, `neutral`, `irrelevant`)
  - `TweetId`: Unique tweet identifier
  - `TweetDate`: Timestamp of post creation
  - `TweetText`: Raw textual content of tweet

## Target Filtering Strategy
To build a 3-class sentiment classifier, records with label `irrelevant` are filtered out, leaving 3,424 annotated tweets categorized into **`positive`**, **`negative`**, and **`neutral`**.
