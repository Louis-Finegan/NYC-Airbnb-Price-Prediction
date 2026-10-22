"""Deterministic, target-free features shared by training and prediction."""

import csv
import json

import numpy as np
import pandas as pd

BASE_CATEGORICAL = [
    "neighbourhood_group_cleansed", "neighbourhood_cleansed", "room_type",
]
CATEGORICAL = BASE_CATEGORICAL + ["property_type"]
BASE_NUMERIC = [
    "accommodates", "bedrooms", "beds", "bathrooms", "number_of_reviews",
    "review_scores_rating",
]
EXTRA_NUMERIC = [
    "latitude", "longitude", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews_ltm", "number_of_reviews_l30d", "reviews_per_month",
    "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location", "review_scores_value",
    "host_listings_count", "calculated_host_listings_count",
]
BOOLEAN = ["host_is_superhost", "instant_bookable"]
BASE_AMENITIES = [
    "Wifi", "Kitchen", "TV", "Air conditioning", "Heating", "Washer", "Dryer",
    "Pool", "Free parking", "Elevator", "Gym",
]
AMENITY_PATTERNS = {
    "Wifi": r"\bwi[ -]?fi\b",
    "Kitchen": r"\bkitchen\b",
    "TV": r"\btv\b|\btelevision\b|hdtv",
    "Air_conditioning": r"air conditioning|central air|\bac\b",
    "Heating": r"heating",
    "Washer": r"(?m)^(?:(?:free|paid|shared|private) )*(?:washer|washing machine)\b",
    "Dryer": r"(?m)^(?:(?:free|paid|shared|private) )*dryer\b",
    "Pool": r"(?m)^(?:(?:free|paid|shared|private|indoor|outdoor|heated) )*pool\b(?! table)",
    "Free_parking": r"free.*parking",
    "Elevator": r"elevator",
    "Gym": r"\bgym\b|fitness",
    "Workspace": r"dedicated workspace",
    "Self_check_in": r"self check[ -]?in|lockbox|keypad",
    "Outdoor_space": r"balcony|patio|backyard|garden",
    "Hot_tub": r"hot tub",
    "Dishwasher": r"dishwasher",
    "Pets_allowed": r"pets allowed",
    "Breakfast": r"breakfast",
}
# IDs, monetary quote fields and revenue/occupancy estimates are excluded.
RAW_FEATURE_COLUMNS = list(dict.fromkeys(
    CATEGORICAL + BASE_NUMERIC + EXTRA_NUMERIC + BOOLEAN + [
        "amenities", "bathrooms_text", "last_scraped", "first_review", "last_review",
        "host_since", "hosts_time_as_host_years", "hosts_time_as_host_months",
        "host_identity_verified", "host_has_profile_pic", "has_availability",
        "price_quote_checkin_date", "price_quote_checkout_date",
    ]
))


def column(frame, name):
    """Return a missing series when an optional source field is unavailable."""
    if name in frame:
        return frame[name]
    return pd.Series(np.nan, index=frame.index)


def numeric(frame, name):
    return pd.to_numeric(column(frame, name), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def boolean(frame, name):
    values = column(frame, name).astype("string").str.lower().str.strip()
    return values.map({"t": 1.0, "true": 1.0, "1": 1.0, "1.0": 1.0,
                       "f": 0.0, "false": 0.0, "0": 0.0, "0.0": 0.0}).astype(float)


def parse_amenities(value):
    """Read current JSON arrays and older quoted, brace-delimited lists."""
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        result = json.loads(value)
        if isinstance(result, list):
            return [str(item).strip() for item in result]
    except (ValueError, TypeError):
        pass
    return [item.strip() for item in next(csv.reader([value.strip("{}[]")]), [])
            if item.strip()]


def distance_km(latitude, longitude, target_latitude, target_longitude):
    """Great-circle distance in kilometres (mean Earth radius: 6,371 km)."""
    lat, lon = np.radians(latitude), np.radians(longitude)
    target_lat, target_lon = np.radians([target_latitude, target_longitude])
    a = (np.sin((lat - target_lat) / 2) ** 2
         + np.cos(lat) * np.cos(target_lat) * np.sin((lon - target_lon) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))


def make_features(frame, feature_set="engineered"):
    """Build features without learning statistics or reading the target.

    ``baseline`` reproduces the original feature set and amenity parsing. The
    engineered version repairs that parsing and adds the features below. Missing
    values remain missing until the model's training-fitted preprocessing step.
    """
    if feature_set not in {"baseline", "engineered"}:
        raise ValueError("feature_set must be 'baseline' or 'engineered'.")
    features = pd.DataFrame(index=frame.index)
    categorical = BASE_CATEGORICAL if feature_set == "baseline" else CATEGORICAL
    for name in categorical:
        features[name] = column(frame, name).fillna("Unknown").astype(str)
    for name in BASE_NUMERIC:
        features[name] = numeric(frame, name)
    for name in BOOLEAN:
        features[name] = boolean(frame, name)

    if feature_set == "baseline":
        features[BOOLEAN] = features[BOOLEAN].fillna(0)
        # Preserve the old parser only in the benchmark for a direct comparison.
        legacy = column(frame, "amenities").map(
            lambda value: [item.strip('" ') for item in value.strip("{}").split(",")]
            if isinstance(value, str) else []
        )
        for name in BASE_AMENITIES:
            features[f"amenity_{name.replace(' ', '_')}"] = legacy.map(
                lambda values: int(name in values)
            )
        return features

    for name in EXTRA_NUMERIC:
        features[name] = numeric(frame, name)
    for name in ["host_identity_verified", "host_has_profile_pic", "has_availability"]:
        features[name] = boolean(frame, name)

    bath_text = column(frame, "bathrooms_text").fillna("").astype(str).str.lower()
    bath_count = pd.to_numeric(bath_text.str.extract(r"(\d+(?:\.\d+)?)", expand=False),
                               errors="coerce")
    bath_count = bath_count.mask(bath_text.str.contains("half"), 0.5)
    features["bathrooms"] = features["bathrooms"].fillna(bath_count)
    features["shared_bathroom"] = bath_text.str.contains("shared").astype(int)
    for name in ["accommodates", "bedrooms", "beds", "bathrooms"]:
        features[f"{name}_missing"] = features[name].isna().astype(int)
    guests = features["accommodates"].where(features["accommodates"] > 0)
    bedrooms = features["bedrooms"].where(features["bedrooms"] > 0)
    features["guests_per_bedroom"] = guests / bedrooms
    features["beds_per_guest"] = features["beds"] / guests
    features["bathrooms_per_guest"] = features["bathrooms"] / guests
    features["is_entire_home"] = features["room_type"].eq("Entire home/apt").astype(int)
    features["long_stay_minimum"] = features["minimum_nights"].ge(28).astype(int)

    amenities = column(frame, "amenities").map(parse_amenities)
    features["amenity_count"] = amenities.map(len)
    text = amenities.map(lambda values: "\n".join(values)).str.lower()
    for name, pattern in AMENITY_PATTERNS.items():
        features[f"amenity_{name}"] = text.str.contains(pattern, regex=True).astype(int)

    for name, coordinates in {
        "times_square": (40.7580, -73.9855),
        "financial_district": (40.7075, -74.0113),
        "central_park": (40.7829, -73.9654),
    }.items():
        features[f"distance_{name}_km"] = distance_km(
            features["latitude"], features["longitude"], *coordinates
        )

    snapshot = pd.to_datetime(column(frame, "last_scraped"), errors="coerce")
    for name in ["first_review", "last_review", "host_since"]:
        date = pd.to_datetime(column(frame, name), errors="coerce")
        age = (snapshot - date).dt.days
        features[f"days_since_{name}"] = age.where(age >= 0)
    features["host_experience_months"] = (
        12 * numeric(frame, "hosts_time_as_host_years")
        + numeric(frame, "hosts_time_as_host_months")
    )
    features["has_reviews"] = features["number_of_reviews"].gt(0).astype(int)
    for name in ["number_of_reviews", "host_listings_count", "minimum_nights"]:
        features[f"log1p_{name}"] = np.log1p(features[name].clip(lower=0))
    features["unavailable_fraction_30"] = 1 - features["availability_30"] / 30
    features["availability_days_31_to_60"] = (
        features["availability_60"] - features["availability_30"]
    )

    # Requested dates are known before quoting a stay; no quote prices are used.
    checkin = pd.to_datetime(column(frame, "price_quote_checkin_date"), errors="coerce")
    checkout = pd.to_datetime(column(frame, "price_quote_checkout_date"), errors="coerce")
    stay = (checkout - checkin).dt.days
    lead = (checkin - snapshot).dt.days
    features["quoted_stay_nights"] = stay.where(stay > 0)
    features["checkin_lead_days"] = lead.where(lead >= 0)
    features["checkin_month_sin"] = np.sin(2 * np.pi * checkin.dt.month / 12)
    features["checkin_month_cos"] = np.cos(2 * np.pi * checkin.dt.month / 12)
    features["checkin_weekday"] = checkin.dt.dayofweek
    return features.replace([np.inf, -np.inf], np.nan)
