"""Select safe model inputs and valid prices; leave imputation to training."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import RAW_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data/extracted/New_York_listings.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/clean_New_York_listings.csv"


def clean_listings(frame):
    """Keep positive finite targets and preserve IDs only for grouped splitting."""
    missing = {"id", "host_id", "price"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    price = pd.to_numeric(
        frame["price"].astype("string").str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )
    valid = price.notna() & np.isfinite(price) & price.gt(0)
    columns = ["id", "host_id"] + [c for c in RAW_FEATURE_COLUMNS if c in frame]
    cleaned = frame.loc[valid, columns].copy()
    if cleaned["id"].isna().any() or cleaned["id"].duplicated().any():
        raise ValueError("Each listing must have a unique, nonmissing id.")
    if cleaned["host_id"].isna().any():
        raise ValueError("host_id is needed to keep hosts out of multiple splits.")
    cleaned["price"] = price.loc[valid].astype(float)
    return cleaned.reset_index(drop=True)


def prepare_data(input_file=DEFAULT_INPUT, output_file=DEFAULT_OUTPUT):
    frame = pd.read_csv(input_file, low_memory=False)
    cleaned = clean_listings(frame)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_file, index=False)
    print(f"Kept {len(cleaned):,} of {len(frame):,} listings; saved {output_file}")
    return cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare_data(args.input, args.output)


if __name__ == "__main__":
    main()
