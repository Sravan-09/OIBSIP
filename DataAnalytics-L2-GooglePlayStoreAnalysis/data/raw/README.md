# Raw Google Play Store Ecosystem Datasets Information

## Dataset Overview
- **Dataset Name**: Google Play Store Apps & User Reviews Datasets
- **Primary Source**: Kaggle / Public Machine Learning Repositories
- **Direct Raw Download URL**: `https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv`
- **Expected Filenames**: `googleplaystore.csv` & `googleplaystore_user_reviews.csv`
- **Target Location**: `data/raw/googleplaystore.csv` & `data/raw/googleplaystore_user_reviews.csv`

## Dataset Description
This project utilizes two complementary datasets capturing the Android app market ecosystem on Google Play. The primary dataset (`googleplaystore.csv`) covers web-scraped attributes for 10,841 mobile applications, including category classification, user rating scores, review counts, file sizes, install tiers, pricing models, content ratings, and genres. The secondary dataset (`googleplaystore_user_reviews.csv`) contains 64,295 customer review records with pre-translated English text and sentiment polarity scores.

## Data Schema & Attributes

### 1. Apps Dataset (`googleplaystore.csv`)
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `App` | Object (String) | Name of the application. |
| `Category` | Object (String) | Market category grouping (e.g. `FAMILY`, `GAME`, `TOOLS`, `FINANCE`). |
| `Rating` | Float64 | Overall average user rating score (1.0 to 5.0). |
| `Reviews` | Object / Integer | Total number of user reviews submitted. |
| `Size` | Object (String) | App file payload size (e.g. `"19M"`, `"850k"`, `"Varies with device"`). |
| `Installs` | Object (String) | Total download count tier (e.g. `"10,000+"`, `"1,000,000+"`). |
| `Type` | Object (String) | App pricing tier (`Free` / `Paid`). |
| `Price` | Object / Float64 | Upfront purchase price in USD ($). |
| `Content Rating` | Object (String) | Target age demographic group (`Everyone`, `Teen`, `Mature 17+`). |
| `Genres` | Object (String) | Sub-category classification tags. |
| `Last Updated` | Object / Date | Date of the latest application release update. |
| `Current Ver` | Object (String) | Current software version build string. |
| `Android Ver` | Object (String) | Minimum required Android OS version. |

### 2. User Reviews Dataset (`googleplaystore_user_reviews.csv`)
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `App` | Object (String) | Name of the application associated with the review. |
| `Translated_Review` | Object (String) | Textual customer feedback translated to English. |
| `Sentiment` | Object (String) | Categorical sentiment label (`Positive`, `Negative`, `Neutral`). |
| `Sentiment_Polarity` | Float64 | Numerical sentiment score ranging from -1.0 to +1.0. |
| `Sentiment_Subjectivity` | Float64 | Subjectivity score ranging from 0.0 (objective) to 1.0 (subjective). |

## Automated Acquisition
The data cleaning module in `src/cleaning.py` automatically verifies and downloads both raw CSV files into this directory when `fetch_and_save_raw_data()` is executed. Manual download is only required if running offline.
