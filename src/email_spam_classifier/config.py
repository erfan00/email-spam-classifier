"""Central configuration: filesystem paths and constants shared across the package."""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_project_root() -> Path:
    """Locate the project root (the directory containing ``pyproject.toml``).

    In an editable install (``pip install -e .``, used in local dev and CI),
    ``__file__`` still points at ``src/email_spam_classifier/config.py`` inside
    the repo checkout, so walking two levels up lands on the repo root.

    In a regular, non-editable install (e.g. ``pip install .`` in the
    Dockerfile), the package is copied into ``site-packages`` and that trick
    no longer works -- so we fall back to the current working directory,
    which is where a deployment is expected to run from (the Dockerfile sets
    ``WORKDIR /app`` and copies ``pyproject.toml``/``static/``/``artifacts/``
    there). This can be overridden explicitly via the
    ``EMAIL_SPAM_CLASSIFIER_HOME`` environment variable.
    """
    env_override = os.environ.get("EMAIL_SPAM_CLASSIFIER_HOME")
    if env_override:
        return Path(env_override)

    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd()


PROJECT_ROOT = _resolve_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATASET_CSV = PROCESSED_DATA_DIR / "dataset.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BASELINE_MODEL_PATH = ARTIFACTS_DIR / "baseline_tfidf_logreg.joblib"
EMBEDDING_MODEL_PATH = ARTIFACTS_DIR / "embedding_classifier.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# The SpamAssassin public corpus archives used to build data/processed/dataset.csv.
# See https://spamassassin.apache.org/old/publiccorpus/ for provenance and licensing notes
# (Apache Software Foundation, donated corpus, free for research/educational reuse).
SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_ARCHIVES = [
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
]

LABEL_HAM = 0
LABEL_SPAM = 1
LABEL_NAMES = {LABEL_HAM: "ham", LABEL_SPAM: "spam"}

RANDOM_STATE = 42

# sentence-transformers model used for the embedding-based classifier.
# Small + fast (384-dim) general purpose sentence embedding model.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
