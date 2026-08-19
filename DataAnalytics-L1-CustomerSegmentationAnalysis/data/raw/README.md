# Raw Dataset Information: UCI Online Retail

## Dataset Overview
- **Dataset Name**: UCI Online Retail Dataset
- **Primary Source**: UCI Machine Learning Repository
- **Direct Mirror URL**: `https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv`
- **Expected Filename**: `Online_Retail.csv`
- **Target Location**: `data/raw/Online_Retail.csv`

## Dataset Description
This dataset contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based non-store online retail organization specializing in unique all-occasion gifts. Many customers are wholesalers.

## Data Schema & Attributes
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | Object (String) | 6-digit integral number uniquely assigned to each transaction. If code starts with 'C', it indicates a cancellation. |
| `StockCode` | Object (String) | 5-digit integral number uniquely assigned to each distinct product. |
| `Description` | Object (String) | Product (item) name. |
| `Quantity` | Integer | The quantities of each product per transaction. |
| `InvoiceDate` | Object / Date | Invoice Day and time when transaction was generated (MM/DD/YYYY HH:MM). |
| `UnitPrice` | Float64 | Product price per unit in sterling (£). |
| `CustomerID` | Float64 / Object | 5-digit integral number uniquely assigned to each customer. |
| `Country` | Object (String) | Name of the country where customer resides. |

## Automated Acquisition
The data pipeline utility in `src/segmentation_utils.py` will automatically download and verify `Online_Retail.csv` when `download_and_load_data()` is executed. Manual download is only required if running offline.
