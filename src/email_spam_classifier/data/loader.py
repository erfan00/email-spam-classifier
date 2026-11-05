"""Parse the raw SpamAssassin public corpus into a flat, labeled dataset.

The corpus ships as directories of individual raw RFC 822 email files, one message
per file (plus a non-email ``cmds`` bookkeeping file in each directory, which is
skipped). This module walks those directories, extracts a plain-text
representation (subject + body) from each message using the standard library's
:mod:`email` parser, and returns a tidy :class:`pandas.DataFrame` with one row per
message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from email import message_from_bytes
from email.message import Message
from pathlib import Path

import pandas as pd

from email_spam_classifier.config import LABEL_HAM, LABEL_SPAM, RAW_DATA_DIR

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# Directory name -> label, per the SpamAssassin public corpus README.
_SOURCE_LABELS: dict[str, int] = {
    "easy_ham": LABEL_HAM,
    "easy_ham_2": LABEL_HAM,
    "hard_ham": LABEL_HAM,
    "spam": LABEL_SPAM,
    "spam_2": LABEL_SPAM,
}


@dataclass(frozen=True)
class RawMessage:
    """A single parsed email with its label and provenance."""

    text: str
    label: int
    source: str
    filename: str


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace from a (possibly HTML) string."""
    no_tags = _HTML_TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", no_tags).strip()


def _decode_payload(payload: bytes, charset: str | None) -> str:
    """Decode a raw payload using its declared charset, falling back to UTF-8."""
    charset = charset or "utf-8"
    try:
        return payload.decode(charset, errors="ignore")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="ignore")


def _extract_part_text(part: Message) -> str | None:
    """Decode a single (non-multipart) message part, if it's text/plain or text/html."""
    content_type = part.get_content_type()
    if content_type not in ("text/plain", "text/html"):
        return None
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes) or not payload:
        return None
    decoded = _decode_payload(payload, part.get_content_charset())
    return _strip_html(decoded) if content_type == "text/html" else decoded


def _extract_body(msg: Message) -> str:
    """Extract a best-effort plain-text body from a (possibly multipart) email."""
    parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            text = _extract_part_text(part)
            if text:
                parts.append(text)
    else:
        text = _extract_part_text(msg)
        if text:
            parts.append(text)

    return _WHITESPACE_RE.sub(" ", " ".join(parts)).strip()


def parse_email_file(path: Path) -> str:
    """Parse a single raw email file into a plain-text "subject + body" string."""
    raw_bytes = path.read_bytes()
    msg = message_from_bytes(raw_bytes)
    subject = msg.get("Subject", "") or ""
    body = _extract_body(msg)
    return f"{subject.strip()} {body}".strip()


def iter_raw_messages(raw_dir: Path = RAW_DATA_DIR) -> list[RawMessage]:
    """Walk every known corpus subdirectory under ``raw_dir`` and parse each message.

    Non-message bookkeeping files (e.g. ``cmds``) and dotfiles are skipped.
    """
    messages: list[RawMessage] = []
    for source, label in _SOURCE_LABELS.items():
        source_dir = raw_dir / source
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.iterdir()):
            if not path.is_file() or path.name == "cmds" or path.name.startswith("."):
                continue
            try:
                text = parse_email_file(path)
            except Exception:  # noqa: BLE001 - corpus has a handful of malformed emails
                continue
            if not text:
                continue
            messages.append(RawMessage(text=text, label=label, source=source, filename=path.name))
    return messages


def build_dataset(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Build a tidy dataframe (``text``, ``label``, ``source``, ``filename``) from raw corpus."""
    messages = iter_raw_messages(raw_dir)
    if not messages:
        raise FileNotFoundError(
            f"No raw corpus files found under {raw_dir}. Run "
            "`python scripts/fetch_data.py` first to download the SpamAssassin "
            "public corpus."
        )
    df = pd.DataFrame(
        {
            "text": [m.text for m in messages],
            "label": [m.label for m in messages],
            "source": [m.source for m in messages],
            "filename": [m.filename for m in messages],
        }
    )
    # Drop exact duplicate messages (the corpus has some repeats across archives).
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a previously built dataset CSV from disk."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {csv_path}. Run "
            "`python scripts/build_dataset.py` to generate it from the raw corpus."
        )
    return pd.read_csv(csv_path)
