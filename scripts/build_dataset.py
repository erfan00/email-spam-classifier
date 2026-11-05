#!/usr/bin/env python3
"""Parse the raw SpamAssassin corpus (``data/raw/``) into ``data/processed/dataset.csv``.

Usage:
    python scripts/build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from email_spam_classifier.config import PROCESSED_DATASET_CSV  # noqa: E402
from email_spam_classifier.data.loader import build_dataset  # noqa: E402


def main() -> None:
    df = build_dataset()
    PROCESSED_DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATASET_CSV, index=False)
    print(f"Wrote {len(df)} rows to {PROCESSED_DATASET_CSV}")
    print(df["label"].value_counts(normalize=True).rename({0: "ham", 1: "spam"}))


if __name__ == "__main__":
    main()
