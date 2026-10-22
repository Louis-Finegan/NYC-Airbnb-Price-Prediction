# Airbnb price prediction

Predict nightly New York City Airbnb listing prices using data from [Inside Airbnb](https://insideairbnb.com/). The project downloads data, prepares listing features, compares regression models and evaluates predictions on hosts excluded from training.

The supplied data is the June 2026 snapshot. Running `download.py` retrieves the latest available snapshot, so its results may differ from the included analysis.

## Setup

Use Python 3.11 or later. From the project directory:

```bash
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Scikit-learn is pinned to the version used for the supplied model. Retrain the model after changing that version. `requirements-tested.txt` records the exact core package versions used for the included results.

## Run the notebook

```bash
python -m jupyterlab analyse.ipynb
```

Run the cells in order. The notebook inspects missing data, demonstrates the engineered features, trains the candidate models, compares validation errors and plots predictions/residuals for the reserved test set. Training can take several minutes.

## Run the scripts

The supplied archive already contains the data. To prepare, train and predict:

```bash
python clean.py
python train.py
python predict.py
```

To download a fresh snapshot and replace the extracted data first:

```bash
python download.py
python read.py --overwrite
python clean.py
python train.py
python predict.py
```

`read.py` extracts the downloaded CSV files into `data/extracted/`.

The downloader uses functions, with no custom scraper class or network activity on import:

```python
from download import download_city_data

download_city_data(
    city_name="New York City, New York, United States",
    city_prefix="New_York",
)
```

It streams large files, checks HTTP errors, uses timeouts and replaces the destination only after a successful transfer. The old underscore form of the city name also works. Downloads are saved in `data/raw/`.

## Files

| File | Purpose |
| --- | --- |
| `download.py` | Download the selected city's compressed CSVs and neighbourhood GeoJSON |
| `read.py` | Extract `.csv.gz` files without loading them into memory |
| `clean.py` | Select allowed raw inputs and positive finite prices; preserve missing features |
| `features.py` | Shared feature functions used by both training and prediction |
| `train.py` | Compare candidates, evaluate the selected model and save its full pipeline |
| `predict.py` | Predict nightly dollar prices from raw listing rows |
| `analyse.ipynb` | Feature engineering, model comparison and held-out diagnostics |
| `models/airbnb_price_model.pkl` | Fitted feature/encoding/model pipeline |
| `reports/` | Metrics, split assignments, held-out predictions and figures |
| `tests/test_workflow.py` | Parsing, download, leakage, split and prediction regression checks |

## Features and evaluation

The engineered features include coordinates and distances to NYC landmarks, property type, bathroom counts recovered from text, capacity ratios, corrected amenity flags/counts, review history, host experience, availability and requested-stay timing. Date ages use the listing's scrape date, rather than today's date.

Listing/host IDs are used only to identify rows and separate hosts. Monetary quote fields and estimated revenue are excluded. Requested check-in/check-out dates are included because they are known inputs to a stay; quote prices are not.

Two fixed grouped splits allocate about 60% of hosts to training, 20% to validation and 20% to testing. Listing proportions vary with host size. No host appears in multiple partitions. Models are selected by **validation mean absolute error in USD**, then refitted on training + validation hosts. The reserved test set compares the selected model and original-feature baseline on the same listings.

The fixed candidates are the original 300-tree random forest, the same forest with engineered features, two histogram gradient-boosting models trained on log prices, and boosting on dollar prices with a Poisson loss and positive log-link predictions. The baseline preserves the original feature set and amenity parser; its imputation is now fitted only on training data.

The model reports dollar MAE/RMSE/R² and log-price MAE/RMSE/R². Expensive listings remain in every metric: no valid positive prices are removed or capped as outliers. The saved model is fitted on development hosts only, so the test results apply to that exact model.

The snapshot evaluation does not establish performance on future dates or completely new properties. Review/availability features describe existing listings. Prices without observed targets are excluded, and their missingness may be systematic. Large misses for expensive listings remain visible in the notebook.

## Predict your own listings

Use a CSV containing raw listing fields, including a JSON `amenities` array. Additional observed inputs generally help; missing optional fields receive the model's fitted missing-value handling.

```bash
python predict.py --input my_listings.csv --output predictions.csv
```

Or use Python:

```python
import pandas as pd
from predict import predict_prices

listings = pd.DataFrame([{
    "neighbourhood_group_cleansed": "Manhattan",
    "neighbourhood_cleansed": "Upper East Side",
    "room_type": "Entire home/apt",
    "property_type": "Entire rental unit",
    "accommodates": 2,
    "bedrooms": 1,
    "beds": 1,
    "bathrooms": 1,
    "amenities": '["Wifi", "Kitchen", "Air conditioning", "Elevator"]',
}])

print(predict_prices(listings)["predicted_price_USD"])
```

**`model.predict()` already returns dollars per night.** Log target inversion is included in the saved pipeline; do not apply `np.expm1()` a second time. The default `python predict.py` shows five real held-out listings.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Data attribution: Inside Airbnb. These are educational estimates of listing prices, rather than booking outcomes or property valuations.
