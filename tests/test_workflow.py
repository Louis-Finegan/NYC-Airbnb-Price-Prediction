"""Regression checks for parsing, leakage, held-out hosts and inference."""

import gzip
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
import requests
from bs4 import BeautifulSoup
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import download
from clean import clean_listings
from features import distance_km, make_features, parse_amenities
from read import extract_data
from train import build_model, split_data


def test_json_and_old_amenities_and_aliases():
    assert parse_amenities('["Wifi", "Kitchen", "Gym"]') == ["Wifi", "Kitchen", "Gym"]
    assert parse_amenities('{"Wifi","Kitchen"}') == ["Wifi", "Kitchen"]
    assert parse_amenities(None) == []
    frame = pd.DataFrame({"amenities": [json.dumps([
        "Fast wifi – 330 Mbps", "Free street parking", "Hair dryer", "Pool table",
    ]), json.dumps(["Shared outdoor pool", "Free dryer – In unit"])]})
    engineered = make_features(frame)
    assert engineered.loc[0, "amenity_Wifi"] == 1
    assert engineered.loc[0, "amenity_Free_parking"] == 1
    assert engineered.loc[0, "amenity_Dryer"] == 0
    assert engineered.loc[0, "amenity_Pool"] == 0
    assert engineered.loc[1, "amenity_Dryer"] == 1
    assert engineered.loc[1, "amenity_Pool"] == 1


def test_bathroom_recovery_and_zero_denominators():
    frame = pd.DataFrame({"bathrooms_text": ["Shared half-bath", "1.5 baths"],
                          "accommodates": [0, 2], "bedrooms": [0, 0], "beds": [0, 1]})
    features = make_features(frame)
    assert features["bathrooms"].tolist() == [0.5, 1.5]
    assert features["shared_bathroom"].tolist() == [1, 0]
    assert np.isnan(features.loc[0, "beds_per_guest"])
    assert not np.isinf(features.select_dtypes("number").to_numpy()).any()
    assert np.isclose(distance_km(pd.Series([40.758]), pd.Series([-73.9855]),
                                  40.758, -73.9855).iloc[0], 0)


def test_price_derived_inputs_cannot_change_features():
    frame = pd.DataFrame({"accommodates": [2, 4], "price": [100, 500],
                          "price_quote_price_per_night": [100, 500],
                          "estimated_revenue_l365d": [5000, 40000],
                          "id": [1, 2], "host_id": [1, 1]})
    changed = frame.copy()
    changed[["price", "price_quote_price_per_night", "estimated_revenue_l365d",
             "id", "host_id"]] = -999
    assert_frame_equal(make_features(frame), make_features(changed))


def test_cleaning_keeps_expensive_prices_and_does_not_impute_globally():
    frame = pd.DataFrame({"id": range(6), "host_id": range(6),
                          "price": ["$100.00", "$30,000.00", None, "-1", "0", "inf"],
                          "beds": [np.nan, 10, 1, 1, 1, 1],
                          "estimated_revenue_l365d": [999] * 6})
    cleaned = clean_listings(frame)
    assert cleaned["price"].tolist() == [100, 30000]
    assert pd.isna(cleaned.loc[0, "beds"])
    assert "estimated_revenue_l365d" not in cleaned


def test_grouped_split_is_reproducible_and_has_no_overlapping_hosts():
    frame = pd.DataFrame({"host_id": np.repeat(np.arange(50), 3)})
    first, second = split_data(frame), split_data(frame)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    groups = [set(frame.iloc[index].host_id) for index in first.values()]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert sorted(np.concatenate(list(first.values()))) == list(range(len(frame)))


def test_saved_pipeline_predicts_dollars_for_unseen_categories_and_missing_fields(tmp_path):
    frame = pd.DataFrame({"accommodates": [1, 2, 4, 6] * 20,
                          "room_type": ["Private room", "Entire home/apt"] * 40,
                          "amenities": ['["Wifi", "Kitchen"]'] * 80})
    prices = np.array([70, 100, 180, 260] * 20)
    model = build_model(max_iter=5, min_samples_leaf=4)
    model.fit(frame, prices)
    path = tmp_path / "model.pkl"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    unseen = pd.DataFrame({"accommodates": [2], "room_type": ["Unseen category"]})
    predicted = reloaded.predict(unseen)
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    np.testing.assert_allclose(predicted, np.expm1(reloaded.regressor_.predict(unseen)))
    np.testing.assert_allclose(model.predict(unseen), predicted)


def test_import_does_not_download_and_city_match_stops_at_next_heading():
    with patch("requests.Session") as session:
        importlib.reload(download)
        session.assert_not_called()
    soup = BeautifulSoup('<h3>Some City</h3><h3>New York City, New York, United States</h3>'
                         '<h4>A date</h4><table id="nyc"><a>Missing href</a></table>'
                         '<h3>Other City</h3><table id="other"></table>', "html.parser")
    assert download.find_city_table(soup, "New_York_City,_New_York,_United_States")["id"] == "nyc"
    with pytest.raises(ValueError):
        download.find_city_table(soup, "Some City")


def test_download_resolves_links_and_streams_files(tmp_path):
    page = MagicMock()
    page.content = b'<h3>Test City</h3><table><a>No URL</a><a href="/listings.csv.gz?q=1">Data</a><a href="/listings.csv.gz?q=1">Duplicate</a><a href="/summary.csv">Skip</a></table>'
    page.__enter__.return_value = page
    streamed = MagicMock()
    streamed.__enter__.return_value = streamed
    streamed.iter_content.return_value = [b"abc", b"", b"def"]
    session = MagicMock()
    session.get.side_effect = [page, streamed]
    session.__enter__.return_value = session
    with patch("download.requests.Session", return_value=session):
        paths = download.download_city_data("Test City", "Test", tmp_path,
                                            "https://example.org/get-data/")
    assert paths == [tmp_path / "Test_listings.csv.gz"]
    assert paths[0].read_bytes() == b"abcdef"
    assert session.get.call_args_list[1].args[0] == "https://example.org/listings.csv.gz?q=1"


def test_failed_download_preserves_previous_complete_file(tmp_path):
    destination = tmp_path / "data.csv.gz"
    destination.write_bytes(b"previous")
    response = MagicMock()
    response.__enter__.return_value = response
    response.raise_for_status.side_effect = requests.HTTPError("404")
    session = MagicMock()
    session.get.return_value = response
    with pytest.raises(requests.HTTPError):
        download.download_file(session, "https://example.org/missing", destination)
    assert destination.read_bytes() == b"previous"
    assert not destination.with_name(destination.name + ".part").exists()


def test_extraction_can_skip_or_overwrite(tmp_path):
    source, target = tmp_path / "raw", tmp_path / "extracted"
    source.mkdir()
    with gzip.open(source / "listings.csv.gz", "wb") as handle:
        handle.write(b"id,price\n1,100\n")
    extracted = extract_data(source, target)[0]
    extracted.write_text("old")
    extract_data(source, target)
    assert extracted.read_text() == "old"
    extract_data(source, target, overwrite=True)
    assert extracted.read_bytes() == b"id,price\n1,100\n"
