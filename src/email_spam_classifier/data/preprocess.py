"""Text preprocessing/normalization shared by both model families.

Kept deliberately simple: the classical TF-IDF model benefits from light
normalization (lowercasing, URL/number/currency placeholders), while the
transformer embedding model works best on near-raw text. Both paths reuse
:func:`clean_text` as a first pass to strip control characters and cap length,
then :func:`normalize_for_tfidf` applies the extra, model-specific folding.
"""

from __future__ import annotations

import re
import unicodedata

# A generous but finite cap. Extremely long inputs (e.g. pasted raw MIME dumps)
# are truncated so downstream vectorizers/tokenizers have bounded latency/memory.
MAX_CHARS = 20_000

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_CURRENCY_RE = re.compile(r"[$£€]\s?\d[\d,.]*")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text: str | None) -> str:
    """Normalize unicode, strip control characters/extra whitespace, and cap length.

    Returns an empty string for ``None`` or non-string input, so callers never need
    to special-case missing text before calling downstream vectorizers.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Normalize unicode (NFKC) so visually-identical characters (e.g. full-width
    # letters, combining accents) collapse to a consistent form.
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    return text


def normalize_for_tfidf(text: str | None) -> str:
    """Additional folding tailored to the TF-IDF + linear-model baseline.

    Lowercases, and replaces URLs/emails/currency amounts/numbers with
    coarse placeholder tokens -- a classic bag-of-words trick that reduces
    vocabulary explosion from one-off tokens (random URLs, invoice numbers,
    prices) while preserving the signal that "this message contains a URL /
    a dollar amount" which is itself predictive of spam.
    """
    text = clean_text(text)
    if not text:
        return ""

    text = text.lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _CURRENCY_RE.sub(" currencytoken ", text)
    text = _NUMBER_RE.sub(" numbertoken ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_for_embedding(text: str | None) -> str:
    """Light-touch normalization for the transformer embedding model.

    Sentence-transformer models are trained on natural, cased text with
    punctuation intact, so we only apply the shared safety cleanup
    (:func:`clean_text`) and avoid the lowercasing/placeholder substitution
    used for TF-IDF.
    """
    return clean_text(text)
