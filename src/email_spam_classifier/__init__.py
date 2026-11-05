"""email_spam_classifier.

A modern (2026) rebuild of a 2016 SVM-based spam classifier.

Provides:
    * Dataset loading/parsing for the SpamAssassin public corpus.
    * Text preprocessing utilities shared by both model families.
    * A fast classical baseline (TF-IDF + linear model).
    * An optional transformer-embedding model with a classifier head.
    * A FastAPI service exposing a ``/predict`` endpoint.
"""

from __future__ import annotations

__version__ = "0.1.0"
