"""Train and evaluate the spam classifiers, saving fitted models + metrics to disk.

Usage:
    python -m email_spam_classifier.train --model baseline
    python -m email_spam_classifier.train --model embedding
    python -m email_spam_classifier.train --model both
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from email_spam_classifier.config import (
    ARTIFACTS_DIR,
    BASELINE_MODEL_PATH,
    EMBEDDING_MODEL_PATH,
    METRICS_PATH,
    PROCESSED_DATASET_CSV,
    RANDOM_STATE,
)
from email_spam_classifier.data.loader import build_dataset, load_dataset
from email_spam_classifier.models.baseline import build_baseline_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_or_build_dataset(csv_path: Path = PROCESSED_DATASET_CSV) -> pd.DataFrame:
    """Load the processed dataset CSV, building it from the raw corpus if absent."""
    if csv_path.exists():
        return load_dataset(csv_path)
    logger.info("Processed dataset not found at %s; building from raw corpus...", csv_path)
    df = build_dataset()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), csv_path)
    return df


def evaluate_predictions(
    y_true: pd.Series | Any, y_pred: Any, y_proba: Any
) -> dict[str, float | list[list[int]]]:
    """Compute the metrics we report in the README: not just accuracy.

    Precision/recall/F1 are reported for the *spam* (positive) class, since on
    an imbalanced dataset (this corpus is roughly 70% ham / 30% spam) accuracy
    alone is misleading -- a classifier that always predicts "ham" would still
    score ~70% accuracy while being useless.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall_spam": float(recall_score(y_true, y_pred, pos_label=1)),
        "f1_spam": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_baseline(df: pd.DataFrame, test_size: float = 0.2) -> dict[str, Any]:
    """Train + evaluate the TF-IDF/Logistic Regression baseline. Returns a results dict."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].to_numpy(),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    pipeline = build_baseline_pipeline("logreg")
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - start

    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    metrics["train_seconds"] = train_seconds
    metrics["predict_ms_per_email"] = predict_seconds / max(len(X_test), 1) * 1000
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, BASELINE_MODEL_PATH)
    logger.info("Saved baseline model to %s", BASELINE_MODEL_PATH)
    loggable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    logger.info("Baseline metrics: %s", loggable)
    return {"model": "baseline_tfidf_logreg", **metrics}


def train_embedding(df: pd.DataFrame, test_size: float = 0.2) -> dict[str, Any]:
    """Train + evaluate the sentence-embedding + Logistic Regression model."""
    from email_spam_classifier.models.embedding import build_embedding_pipeline

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].to_numpy(),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    model = build_embedding_pipeline()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - start

    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    metrics["train_seconds"] = train_seconds
    metrics["predict_ms_per_email"] = predict_seconds / max(len(X_test), 1) * 1000
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, EMBEDDING_MODEL_PATH)
    logger.info("Saved embedding model to %s", EMBEDDING_MODEL_PATH)
    loggable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    logger.info("Embedding metrics: %s", loggable)
    return {"model": "embedding_minilm_logreg", **metrics}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["baseline", "embedding", "both"],
        default="baseline",
        help="Which model(s) to train. 'embedding' requires the optional "
        "'embedding' extra (pip install -e '.[embedding]').",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args(argv)

    df = load_or_build_dataset()
    logger.info(
        "Dataset: %d messages (%d spam, %d ham)",
        len(df),
        int((df["label"] == 1).sum()),
        int((df["label"] == 0).sum()),
    )

    all_results: dict[str, Any] = {}
    if args.model in ("baseline", "both"):
        all_results["baseline"] = train_baseline(df, test_size=args.test_size)
    if args.model in ("embedding", "both"):
        all_results["embedding"] = train_embedding(df, test_size=args.test_size)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if METRICS_PATH.exists():
        existing = json.loads(METRICS_PATH.read_text())
    existing.update(all_results)
    METRICS_PATH.write_text(json.dumps(existing, indent=2))
    logger.info("Wrote metrics to %s", METRICS_PATH)


if __name__ == "__main__":
    main()
