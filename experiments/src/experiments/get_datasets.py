from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from experiments.scalability import DATASET, ZONE_LOOKUP

logger = logging.getLogger("experiments.get_datasets")

SYNTABFALL_URL = "https://zenodo.org/records/20427694/files/SynTabFall.csv?download=1"
TRIP_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_LOOKUP_SHA256 = "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed"
NYC_TAXI_SHA256 = {
    "yellow_tripdata_2024-01.parquet": "c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510",
    "yellow_tripdata_2024-02.parquet": "c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20",
    "yellow_tripdata_2024-03.parquet": "2d4cdc8fb96726cdd3803b13b02d2e61e71d45720aff0ebc693a8bdd1f249823",
    "yellow_tripdata_2024-04.parquet": "ce37f736c7cf9e92164c850ef2b502a8f7bb590fc1804b6edbcb80293da01b89",
    "yellow_tripdata_2024-05.parquet": "1974230be2b5a3ea92e47b5eb77f32dbf1de896d81ad8df4ab6faba52749ad23",
    "yellow_tripdata_2024-06.parquet": "677cf14c8347f745b583f012fbdba072334c6c4efa17bfd6a64369f2ba30329c",
    "yellow_tripdata_2024-07.parquet": "af12d6b04daeb78550799ae99933d495b27d715b6dbb0fad9bb750a4537d3070",
    "yellow_tripdata_2024-08.parquet": "7315643dd16bafc03e27d9903edbfd35404c2550a695d0da0a3e73852d940cfb",
    "yellow_tripdata_2024-09.parquet": "2f42e4a383a6635de803001462848d0aee2b47f6ce899126805d2d05361fb843",
    "yellow_tripdata_2024-10.parquet": "d19fe05aa0b259af24eb47051735293287f1e6dc263ba04030ec1ab4c6ec2650",
    "yellow_tripdata_2024-11.parquet": "5ef321876de5007a7c147a347389133fc94fd7ac28f82eb1d57d8f2cf05cfd3b",
    "yellow_tripdata_2024-12.parquet": "41ebf7db80bebde60c58e5143c14cdf38ad04a0f3e3ff44215b3e240d55f6c78",
}


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fetch(url: str, path: Path, expected_sha256: str | None = None) -> None:
    """Download `url` to `path` unless it is there, then check it against `expected_sha256`."""
    if not path.exists():
        logger.info("downloading %s", url)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response, path.open("wb") as out:
            shutil.copyfileobj(response, out)
    # TLC may replace a file; the paper's runtimes hold only for these.
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f"{path} differs from the file the paper measured")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the NYC taxi trips and SynTabFall.")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "experiments/data")))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fetch(SYNTABFALL_URL, args.data_dir / "syntabfall" / "SynTabFall.csv")
    taxi = args.data_dir / DATASET
    fetch(ZONE_LOOKUP_URL, taxi / ZONE_LOOKUP, ZONE_LOOKUP_SHA256)
    for month, expected in NYC_TAXI_SHA256.items():
        fetch(f"{TRIP_DATA_URL}/{month}", taxi / month, expected)


if __name__ == "__main__":
    main()
