# Raw Dataset Information: Titanic Machine Learning Dataset

## Dataset Overview
- **Dataset Name**: Titanic Machine Learning Dataset
- **Primary Source**: DataScienceDojo / Kaggle / UCI Mirror
- **Direct Raw Download URL**: `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
- **Expected Filename**: `raw_titanic.csv`
- **Target Location**: `data/raw/raw_titanic.csv`

## Dataset Description
The Titanic dataset contains demographic, ticket, cabin, and survival details for 891 passengers on board the RMS Titanic. It presents classic real-world data quality issues including missing values (`Age`, `Cabin`, `Embarked`), unstandardized strings, duplicate rows, and extreme numerical outliers in `Fare`.

## Data Schema & Attributes
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `PassengerId` | Integer / String | Unique passenger identification key. |
| `Survived` | Integer | Binary survival indicator (0 = No, 1 = Yes). |
| `Pclass` | Integer | Ticket class (1 = 1st/Upper, 2 = 2nd/Middle, 3 = 3rd/Lower). |
| `Name` | Object (String) | Full passenger name including title (Mr., Mrs., Miss, Master, etc.). |
| `Sex` | Object (String) | Passenger gender (male / female). |
| `Age` | Float64 | Passenger age in years. |
| `SibSp` | Integer | Number of siblings / spouses aboard the Titanic. |
| `Parch` | Integer | Number of parents / children aboard the Titanic. |
| `Ticket` | Object (String) | Ticket number string. |
| `Fare` | Float64 | Passenger fare in British pounds (£). |
| `Cabin` | Object (String) | Cabin number string. |
| `Embarked` | Object (String) | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). |

## Automated Acquisition
The data cleaning utility in `src/cleaning_utils.py` automatically downloads `raw_titanic.csv` into this folder when `download_and_load_data()` is executed.
