# Raw House Prices Dataset Information: Ames Housing Dataset

## Dataset Overview
- **Dataset Name**: Ames Housing Dataset (House Prices Dataset)
- **Primary Source**: OpenML (Data ID 42165) / Kaggle Repository
- **Direct Raw Download URL**: `https://www.openml.org/d/42165`
- **Expected Filename**: `house_prices_raw.csv`
- **Target Location**: `data/raw/house_prices_raw.csv`

## Dataset Description
The Ames Housing dataset describes the sale of individual residential properties in Ames, Iowa, from 2006 to 2010. It contains 1,460 observations across 81 explanatory attributes (37 numerical features, 43 categorical features, and 1 target `SalePrice`). The dataset presents real-world housing complexities including missing values, skewed distributions, and multi-category property attributes.

## Data Schema & Attributes
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Id` | Integer / Object | Property identification number. |
| `MSSubClass` | Integer | Type of dwelling involved in the sale. |
| `MSZoning` | Object (String) | General zoning classification of the property (`RL`, `RM`, `FV`, `RH`, `C`). |
| `LotArea` | Integer / Float64 | Lot size in square feet. |
| `OverallQual` | Integer | Rates the overall material and finish of the house (1 to 10 scale). |
| `OverallCond` | Integer | Rates the overall condition of the house (1 to 10 scale). |
| `YearBuilt` | Integer | Original construction date. |
| `YearRemodAdd` | Integer | Remodel date (same as construction date if no remodeling). |
| `TotalBsmtSF` | Float64 | Total square feet of basement area. |
| `1stFlrSF` | Float64 | First floor square feet. |
| `2ndFlrSF` | Float64 | Second floor square feet. |
| `GrLivArea` | Float64 | Above grade (ground) living area square feet. |
| `FullBath` | Integer | Full bathrooms above grade. |
| `BedroomAbvGr` | Integer | Bedrooms above grade. |
| `TotRmsAbvGrd` | Integer | Total rooms above grade (excluding bathrooms). |
| `GarageCars` | Float64 | Size of garage in car capacity. |
| `GarageArea` | Float64 | Size of garage in square feet. |
| `WoodDeckSF` | Float64 | Wood deck area in square feet. |
| `OpenPorchSF` | Float64 | Open porch area in square feet. |
| `YrSold` | Integer | Year Sold (YYYY). |
| `SaleCondition` | Object (String) | Condition of sale (`Normal`, `Abnorml`, `Partial`, etc.). |
| `SalePrice` | Float64 | Property sale price in USD ($) — Target Variable. |

## Automated Acquisition
The data pipeline utility in `src/preprocessing.py` will automatically fetch the raw dataset from OpenML and save `house_prices_raw.csv` into this directory when `fetch_and_save_raw_data()` is executed. Manual download is only required if running offline.
