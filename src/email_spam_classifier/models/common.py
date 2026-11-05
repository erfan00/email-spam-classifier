"""Shared types used by both model implementations and the API layer."""

from __future__ import annotations

from dataclasses import dataclass

from email_spam_classifier.config import LABEL_NAMES


@dataclass(frozen=True)
class PredictionResult:
    """A single classification result, independent of which model produced it."""

    label: str
    spam_probability: float
    model_name: str

    @classmethod
    def from_label_id(
        cls, label_id: int, spam_probability: float, model_name: str
    ) -> PredictionResult:
        return cls(
            label=LABEL_NAMES[label_id],
            spam_probability=spam_probability,
            model_name=model_name,
        )
