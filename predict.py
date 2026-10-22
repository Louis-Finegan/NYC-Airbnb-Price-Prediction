"""Predict dollars per night from raw listing rows using the saved pipeline."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train import MODEL_PATH, REPORT_DIR

ROOT = Path(__file__).resolve().parent


def predict_prices(listings, model_path=MODEL_PATH):
    """Return dollar predictions; optional missing fields use fitted handling."""
    model = joblib.load(model_path)
    prices = model.predict(listings)
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("Model returned invalid prices.")
    result = listings.copy()
    result["predicted_price_USD"] = np.round(prices, 2)
    return result


def example_listings():
    """Use complete unseen examples from the saved test set, rather than inventing inputs."""
    holdout_file = REPORT_DIR / "holdout_predictions.csv"
    data_file = ROOT / "data/processed/clean_New_York_listings.csv"
    if not holdout_file.exists() or not data_file.exists():
        raise FileNotFoundError("Run clean.py and train.py, or supply --input listings.csv.")
    ids = pd.read_csv(holdout_file, usecols=["id"]).head(5)["id"]
    data = pd.read_csv(data_file, low_memory=False).set_index("id")
    return data.loc[ids].reset_index().drop(columns="price")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV of raw listing features")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    listings = pd.read_csv(args.input, low_memory=False) if args.input else example_listings()
    result = predict_prices(listings, args.model)
    columns = [name for name in ["id", "neighbourhood_group_cleansed", "neighbourhood_cleansed",
                               "room_type", "accommodates", "predicted_price_USD"] if name in result]
    print(result[columns].to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
