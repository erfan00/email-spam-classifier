"""Inference wrapper: load a trained model artifact once, then classify text.

Used directly by the FastAPI app (see :mod:`email_spam_classifier.api.main`) and
by anything else (scripts, tests, notebooks) that wants predictions without
going through HTTP.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib

from email_spam_classifier.config import BASELINE_MODEL_PATH, EMBEDDING_MODEL_PATH
from email_spam_classifier.data.preprocess import clean_text
from email_spam_classifier.models.common import PredictionResult

ModelName = Literal["baseline", "embedding"]

_MODEL_PATHS: dict[ModelName, Path] = {
    "baseline": BASELINE_MODEL_PATH,
    "embedding": EMBEDDING_MODEL_PATH,
}


class ModelNotTrainedError(RuntimeError):
    """Raised when a prediction is requested for a model that hasn't been trained yet."""


@lru_cache(maxsize=4)
def _load_model(model_name: ModelName) -> object:
    path = _MODEL_PATHS[model_name]
    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained '{model_name}' model found at {path}. Train it first, e.g.: "
            f"python -m email_spam_classifier.train --model {model_name}"
        )
    return joblib.load(path)


def is_model_available(model_name: ModelName) -> bool:
    """Check whether a trained artifact exists for ``model_name`` without loading it."""
    return _MODEL_PATHS[model_name].exists()


def clear_model_cache() -> None:
    """Drop any cached (loaded) models -- primarily useful for tests."""
    _load_model.cache_clear()


def predict_text(text: str, model_name: ModelName = "baseline") -> PredictionResult:
    """Classify a single piece of raw text as spam or ham.

    Handles edge cases explicitly:
        * ``None`` / empty / whitespace-only text -> classified using the model
          as-is (an empty string), never raises.
        * Non-ASCII text -> handled transparently by unicode-aware cleaning.
        * Very long text -> truncated by :func:`clean_text` before modeling.
    """
    model = _load_model(model_name)
    cleaned = clean_text(text)
    # Even an empty string must flow through the real model so behavior (e.g. the
    # resulting probability) is governed by the trained model, not a special case.
    proba = model.predict_proba([cleaned])[0]  # type: ignore[attr-defined]
    spam_index = list(model.classes_).index(1)  # type: ignore[attr-defined]
    spam_probability = float(proba[spam_index])
    label_id = 1 if spam_probability >= 0.5 else 0
    return PredictionResult.from_label_id(label_id, spam_probability, model_name=model_name)
