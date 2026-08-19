# Raw Retail Sales Dataset Information

## Dataset Source
- **Dataset Name**: Retail Sales Dataset
- **Primary Source**: Kaggle / Public Data Repository
- **Direct Raw Download URL**: `https://raw.githubusercontent.com/hatimh53/Retail-Sales-Analysis/main/retail_sales_dataset.csv`
- **Expected Filename**: `retail_sales_dataset.csv`
- **Destination Path**: `data/raw/retail_sales_dataset.csv`

## Dataset Description
This dataset contains transaction-level sales data for a retail organization spanning multiple years (2020–2024). It includes customer demographic information, product category breakdown, pricing, discounts, shipping attributes, customer satisfaction metrics, and order status flags.

## Data Schema & Attributes
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `order_id` | Object (String) | Unique alphanumeric order identification code |
| `order_date` | Object / Date | Timestamp of transaction creation (YYYY-MM-DD) |
| `customer_id` | Object (String) | Unique customer identifier |
| `customer_name` | Object (String) | Full name of the purchasing customer |
| `age` | Float64 / Integer | Age of customer in years |
| `gender` | Object (String) | Gender of customer (Male / Female / Other) |
| `region` | Object (String) | Geographical region of transaction |
| `city` | Object (String) | City location of customer |
| `product_category` | Object (String) | Product grouping (Electronics, Clothing, Home & Kitchen, Books, etc.) |
| `product_name` | Object (String) | Specific product title |
| `quantity` | Float64 / Integer | Number of units purchased |
| `unit_price` | Float64 | Unit price per item in USD ($) |
| `discount_pct` | Float64 | Discount rate applied (0.0 to 1.0) |
| `sales_amount` | Float64 | Net revenue generated ($) |
| `profit` | Float64 | Gross profit generated ($) |
| `shipping_cost` | Float64 | Shipping cost incurred ($) |
| `payment_method` | Object (String) | Payment mode (Credit Card, PayPal, Debit Card, Cash) |
| `customer_satisfaction` | Float64 | Customer review rating (1.0 to 5.0 scale) |
| `return_flag` | Object / Boolean | Indicates whether item was returned (True / False) |
| `order_status` | Object (String) | Fulfillment state (Delivered, Pending, Returned, Cancelled) |
| `days_to_ship` | Float64 / Integer | Delivery duration in days |

## Automated Acquisition
The data fetching utility in `src/data_utils.py` will automatically download and verify `retail_sales_dataset.csv` into this directory when `load_retail_data()` is invoked. Manual downloads are not strictly necessary unless running offline.
