"""Shared pytest fixtures.

Tests never depend on the full downloaded SpamAssassin corpus (which is not
committed to the repo and is gitignored) -- fixtures below synthesize tiny
corpora and datasets so the test suite is fast and fully self-contained.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

import email_spam_classifier.predict as predict_module
from email_spam_classifier.models.baseline import build_baseline_pipeline

# A small, clearly-separable synthetic dataset: enough rows per class for a
# stratified train/test split, with vocabulary a linear TF-IDF model can
# actually separate (so tests of "does training work end-to-end" are meaningful).
SPAM_SAMPLES = [
    "WINNER!! Claim your FREE prize now, click here urgent limited offer",
    "Free money viagra discount buy now click link winner cash prize",
    "URGENT: You have won a lottery, send your bank details to claim cash",
    "Cheap pills online pharmacy discount viagra cialis buy now free shipping",
    "Congratulations you are selected for a free gift card click now",
    "Make money fast working from home guaranteed income click here now",
    "Limited time offer act now free trial credit card required urgent",
    "Hot singles in your area click here free chat now limited offer",
    "You have been selected winner claim prize now free cash bonus",
    "Lowest price guaranteed buy cheap meds online no prescription needed",
]
HAM_SAMPLES = [
    "Hey, are we still meeting for lunch tomorrow at noon?",
    "Please find attached the quarterly report for review before Friday.",
    "Thanks for your help yesterday, the project deployment went smoothly.",
    "Can you send me the slides from this morning's team meeting?",
    "Reminder: the dentist appointment is scheduled for next Tuesday at 3pm.",
    "I reviewed the pull request and left a few comments, looks good overall.",
    "Let's catch up over coffee next week, I have some ideas to discuss.",
    "The flight itinerary for the conference trip is attached, please confirm.",
    "Great seeing everyone at the family dinner last weekend, let's do it again.",
    "Here are the notes from today's sync, action items are listed at the bottom.",
]


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    """A small but class-separable spam/ham dataframe with ``text``/``label`` columns."""
    texts = SPAM_SAMPLES + HAM_SAMPLES
    labels = [1] * len(SPAM_SAMPLES) + [0] * len(HAM_SAMPLES)
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def synthetic_raw_corpus(tmp_path: Path) -> Path:
    """Write a tiny raw-email corpus laid out like the real SpamAssassin directories."""
    raw_dir = tmp_path / "raw"
    for subdir, texts, label_hint in (
        ("easy_ham", HAM_SAMPLES, "ham"),
        ("spam", SPAM_SAMPLES, "spam"),
    ):
        d = raw_dir / subdir
        d.mkdir(parents=True)
        # A non-email bookkeeping file, which the loader must skip.
        (d / "cmds").write_text("rm 0001\n")
        for i, body in enumerate(texts):
            msg = (
                f"From: sender{i}@example.com\n"
                f"To: recipient@example.com\n"
                f"Subject: Test {label_hint} message {i}\n"
                "Content-Type: text/plain; charset=us-ascii\n"
                "\n"
                f"{body}\n"
            )
            (d / f"{i:04d}.{label_hint}").write_text(msg)
    return raw_dir


@pytest.fixture
def trained_baseline_model_path(tmp_path: Path, synthetic_dataset: pd.DataFrame) -> Path:
    """Train the baseline pipeline on the synthetic dataset and persist it to a tmp path."""
    pipeline = build_baseline_pipeline("logreg")
    pipeline.fit(synthetic_dataset["text"].tolist(), synthetic_dataset["label"].to_numpy())
    model_path = tmp_path / "baseline_test_model.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture
def wired_baseline_model(
    monkeypatch: pytest.MonkeyPatch, trained_baseline_model_path: Path
) -> None:
    """Point the prediction module's baseline model path at a freshly trained test model."""
    predict_module.clear_model_cache()
    monkeypatch.setitem(predict_module._MODEL_PATHS, "baseline", trained_baseline_model_path)
    yield
    predict_module.clear_model_cache()
