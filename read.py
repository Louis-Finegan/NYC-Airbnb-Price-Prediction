"""Extract downloaded .csv.gz files without loading them into memory."""

import argparse
import gzip
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def extract_data(source_dir=ROOT / "data/raw", output_dir=ROOT / "data/extracted",
                 overwrite=False):
    output_dir = Path(output_dir)
    files = sorted(Path(source_dir).glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No .csv.gz files found in {source_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for compressed in files:
        output = output_dir / compressed.with_suffix("").name
        if output.exists() and not overwrite:
            print(f"Already extracted: {output.name}")
        else:
            temporary = output.with_name(output.name + ".part")
            try:
                with gzip.open(compressed, "rb") as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Extracted {output.name}")
        paths.append(output)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/extracted")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_data(args.source, args.output, args.overwrite)


if __name__ == "__main__":
    main()
