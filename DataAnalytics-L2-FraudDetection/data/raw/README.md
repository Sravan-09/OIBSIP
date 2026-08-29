# Raw Credit Card Fraud Dataset Information

## Dataset Overview
- **Dataset Name**: Credit Card Fraud Detection Dataset
- **Primary Source**: OpenML (Data ID 1597) / Kaggle Repository
- **Direct Raw Download URL**: `https://www.openml.org/d/1597`
- **Expected Filename**: `creditcard_raw.csv`
- **Target Location**: `data/raw/creditcard_raw.csv`

## Dataset Description
This benchmark dataset contains transaction records made by credit cards in September 2013 by European cardholders. The dataset contains 284,807 total transactions occurring over two days, with 492 fraudulent transactions (0.172% extreme class imbalance). Features `V1` through `V28` are numerical attributes resulting from a Principal Component Analysis (PCA) transformation performed for confidentiality.

## Data Schema & Attributes
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `V1` to `V28` | Float64 | Anonymized numerical features resulting from PCA transformation to protect privacy. |
| `Amount` | Float64 | Transaction amount in currency units. |
| `Class` | Integer | Binary fraud classification target (0 = Legitimate, 1 = Fraudulent). |

## Automated Acquisition
The data fetching utility in `src/preprocessing.py` automatically fetches the raw dataset from OpenML (Data ID 1597) and saves `creditcard_raw.csv` into this directory when `fetch_and_save_raw_data()` is executed. Manual download is only required if running offline.
