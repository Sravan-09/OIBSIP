# Raw Wine Quality Dataset Information

## Dataset Overview
- **Dataset Name**: UCI Wine Quality Dataset (Red Wine Variant)
- **Primary Source**: UCI Machine Learning Repository
- **Direct Raw Download URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv`
- **Expected Filename**: `wine_quality_raw.csv`
- **Target Location**: `data/raw/wine_quality_raw.csv`

## Dataset Description
This benchmark dataset contains physicochemical laboratory test measurements for 1,599 red wine variants of the Portuguese "Vinho Verde" wine. The dataset contains 11 numerical input attributes measuring acidity, sugar, chlorides, sulfur dioxide, density, pH, sulphates, and alcohol content, alongside a sensory quality rating score (0 to 10 scale).

## Data Schema & Attributes
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `fixed_acidity` | Float64 | Non-volatile tartaric acid concentration (g/dm³). |
| `volatile_acidity` | Float64 | Acetic acid concentration in wine (g/dm³); high levels cause unpleasant vinegar taste. |
| `citric_acid` | Float64 | Citric acid concentration (g/dm³); adds freshness and flavor. |
| `residual_sugar` | Float64 | Amount of sugar remaining after fermentation stops (g/dm³). |
| `chlorides` | Float64 | Amount of salt in the wine (g/dm³). |
| `free_sulfur_dioxide` | Float64 | Free form of SO₂ preventing microbial growth and oxidation (mg/dm³). |
| `total_sulfur_dioxide` | Float64 | Bound + free forms of SO₂ (mg/dm³). |
| `density` | Float64 | Density of wine relative to water (g/cm³). |
| `pH` | Float64 | Describes acidity/alkalinity scale (0 = very acidic, 14 = very basic). |
| `sulphates` | Float64 | Wine additive contributing to SO₂ levels (g/dm³); acts as antimicrobial & antioxidant. |
| `alcohol` | Float64 | Percent alcohol content by volume (% vol). |
| `quality` | Integer | Sensory rating score assigned by wine experts (scale from 0 to 10). |

## Automated Acquisition
The preprocessing utility in `src/preprocessing.py` automatically downloads `winequality-red.csv` from the UCI ML Repository and saves standardized `wine_quality_raw.csv` into this directory when `fetch_and_save_raw_data()` is executed. Manual download is only required if running offline.
