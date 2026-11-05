#!/usr/bin/env python3
"""Download and extract the SpamAssassin public corpus into ``data/raw/``.

Source: https://spamassassin.apache.org/old/publiccorpus/
License/provenance: donated to the Apache SpamAssassin project; distributed
publicly for research/educational anti-spam use. See the README for details.

Usage:
    python scripts/fetch_data.py
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from email_spam_classifier.config import (  # noqa: E402
    RAW_DATA_DIR,
    SPAMASSASSIN_ARCHIVES,
    SPAMASSASSIN_BASE_URL,
)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already downloaded: {dest.name}")
        return
    print(f"  downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed, known-good HTTPS source


def extract(archive_path: Path, dest_dir: Path) -> None:
    print(f"  extracting {archive_path.name}")
    with tarfile.open(archive_path, "r:bz2") as tf:
        tf.extractall(dest_dir)  # noqa: S202 - trusted, hard-coded corpus archives


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for archive_name in SPAMASSASSIN_ARCHIVES:
        url = SPAMASSASSIN_BASE_URL + archive_name
        archive_path = RAW_DATA_DIR / archive_name
        download(url, archive_path)
        extract(archive_path, RAW_DATA_DIR)
    print("Done. Raw corpus is under", RAW_DATA_DIR)


if __name__ == "__main__":
    main()
