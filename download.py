"""Download a city's detailed Inside Airbnb datasets using ordinary functions."""

import argparse
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://insideairbnb.com/get-the-data/"
DEFAULT_CITY = "New York City, New York, United States"


def find_city_table(soup, city_name):
    """Find the latest table for a city; accept spaces or the old underscores."""
    wanted = " ".join(city_name.replace("_", " ").split()).casefold()
    for heading in soup.find_all("h3"):
        city = " ".join(heading.get_text(" ", strip=True).split()).casefold()
        if city == wanted:
            for element in heading.find_all_next(["h3", "table"]):
                if element.name == "h3":
                    break
                return element
    raise ValueError(f"No download table found for {city_name!r}.")


def download_file(session, url, destination):
    """Stream a file to disk, publishing it only after the download completes."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with session.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def download_city_data(city_name=DEFAULT_CITY, city_prefix="New_York",
                       download_dir=ROOT / "data/raw", base_url=DATA_URL):
    """Download .csv.gz and .geojson files; return the paths that were saved."""
    if not city_prefix or any(c in city_prefix for c in "/\\") or city_prefix in {".", ".."}:
        raise ValueError("city_prefix must be a filename prefix without slashes.")
    with requests.Session() as session:
        session.headers["User-Agent"] = "Airbnb-price-analysis/1.0"
        with session.get(base_url, timeout=(15, 60)) as response:
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
        table = find_city_table(soup, city_name)
        saved = []
        seen = set()
        for link in table.find_all("a", href=True):
            url = urljoin(base_url, link["href"])
            parsed = urlparse(url)
            name = Path(unquote(parsed.path)).name
            if parsed.scheme not in {"http", "https"}:
                continue
            if not name.endswith((".csv.gz", ".geojson")) or name in seen:
                continue
            seen.add(name)
            destination = Path(download_dir) / f"{city_prefix}_{name}"
            saved.append(download_file(session, url, destination))
            print(f"Downloaded {destination}")
    if not saved:
        raise ValueError(f"No detailed data links found for {city_name!r}.")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--prefix", default="New_York")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    download_city_data(args.city, args.prefix, args.output)


if __name__ == "__main__":
    main()
