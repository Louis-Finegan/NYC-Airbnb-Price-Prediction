# NYC Airbnb Price Prediction

A Python project using scikit-learn to predict nightly Airbnb listing prices in New York City. The project uses data from [Inside Airbnb](https://insideairbnb.com/) to explore listing characteristics, engineer features and compare random forest and gradient boosting regression models.

The workflow covers downloading and preparing data, training models, evaluating predictions and visualising the results.

## Setup

Requires Python 3.11 or later. From the project directory, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The exact core package versions used for the included analysis are recorded in `requirements-tested.txt`. Retrain the model after changing the scikit-learn version.

## Run the project

From a fresh copy of the repository, download the data, extract the files, prepare the listings and train the models:

```bash
python download.py
python read.py --overwrite
python clean.py
python train.py
python predict.py
```

The downloader retrieves the available NYC snapshot from Inside Airbnb. Downloaded files are saved in `data/raw/`, and extracted CSV files are stored in `data/extracted/`.

Training compares the candidate models, saves the selected pipeline in `models/` and writes evaluation outputs to `reports/`. The prediction script then uses the saved pipeline to estimate nightly prices.

If the data has already been downloaded and extracted, run:

```bash
python clean.py
python train.py
python predict.py
```

Datasets and trained models are generated locally and are excluded from version control. Results may vary with the downloaded snapshot, features and model settings.

## Explore the notebook

Open the analysis notebook with:

```bash
python -m jupyterlab analyse.ipynb
```

Run the cells in order to explore missing data, examine engineered features, compare models and view prediction and residual plots.

## Project files

| File | Purpose |
| --- | --- |
| `download.py` | Downloads listing data and supporting files from Inside Airbnb |
| `read.py` | Extracts the compressed CSV files |
| `clean.py` | Prepares listing data for modelling |
| `features.py` | Defines shared feature engineering functions |
| `train.py` | Trains and compares models, evaluates predictions and saves the selected pipeline |
| `predict.py` | Generates price predictions using the saved pipeline |
| `analyse.ipynb` | Explores the data and visualises model performance |
| `reports/` | Stores evaluation metrics, predictions, split assignments and figures |
| `tests/test_workflow.py` | Checks data handling, dataset separation and prediction behaviour |

## Features and models

The models use listing characteristics such as location, property and room type, guest capacity, amenities, reviews and host information.

Feature engineering adds quantities such as distances to NYC landmarks, capacity ratios, amenity indicators and bathroom counts recovered from text. Date-based features use the listing's scrape date.

Random forests and histogram gradient boosting models are compared using different feature sets and target transformations. Preprocessing and prediction are combined in a pipeline so the same transformations are applied during training and inference.

Listing and host identifiers are excluded from the model inputs. Monetary quote fields and estimated revenue are also excluded.

## Evaluation

Listings are separated into training, validation and test sets by host. This prevents listings belonging to the same host from appearing in different partitions.

Models are selected using validation mean absolute error in dollars. The selected model is then refitted on the training and validation data and compared with the baseline on the same reserved test listings.

Performance is reported using:

- **MAE:** average absolute prediction error.
- **RMSE:** prediction error with greater weight given to large misses.
- **R²:** performance relative to a constant prediction based on the evaluation set's mean price.

Metrics are calculated in both dollar and log-price space. Numerical results, test predictions and diagnostic figures are available in [`reports/`](reports/) and the notebook.

All valid positive prices are retained, including expensive listings. Large errors can occur for unusual or high-priced properties. Evaluation within a single snapshot does not establish accuracy for future prices.

## Predict your own listings

After training, supply a CSV containing listing fields in the format used by the source data:

```bash
python predict.py --input my_listings.csv --output predictions.csv
```

Amenity values should use the JSON array format provided by Inside Airbnb.

Predictions are returned in USD per night. Any log-price transformation is reversed within the saved pipeline.

## Run the checks

Install the development dependencies and run the tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The checks cover data parsing, download handling, feature exclusions, separation of hosts between datasets and prediction outputs.

## Data source

Data is provided by [Inside Airbnb](https://insideairbnb.com/).

This project estimates advertised listing prices for educational analysis. Predictions do not represent realised booking revenue or property valuations.
