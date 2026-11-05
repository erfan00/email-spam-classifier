"""Tests for text preprocessing, including explicit edge cases."""

from __future__ import annotations

import pytest

from email_spam_classifier.data.preprocess import (
    MAX_CHARS,
    clean_text,
    normalize_for_embedding,
    normalize_for_tfidf,
)


class TestCleanText:
    def test_none_returns_empty_string(self) -> None:
        assert clean_text(None) == ""

    def test_empty_string_returns_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_whitespace_only_returns_empty_string(self) -> None:
        assert clean_text("   \n\t  ") == ""

    def test_non_ascii_text_is_preserved(self) -> None:
        text = "Bonjour, texte avec accents: café, naïve, 日本語, emoji 🎉"
        cleaned = clean_text(text)
        assert "café" in cleaned
        assert "日本語" in cleaned
        assert "🎉" in cleaned

    def test_control_characters_are_stripped(self) -> None:
        text = "hello\x00world\x07test"
        cleaned = clean_text(text)
        assert "\x00" not in cleaned
        assert "\x07" not in cleaned

    def test_collapses_whitespace(self) -> None:
        assert clean_text("hello    \n\n   world") == "hello world"

    def test_very_long_text_is_truncated(self) -> None:
        long_text = "a" * (MAX_CHARS * 3)
        cleaned = clean_text(long_text)
        assert len(cleaned) <= MAX_CHARS

    def test_non_string_input_is_coerced(self) -> None:
        assert clean_text(12345) == "12345"  # type: ignore[arg-type]


class TestNormalizeForTfidf:
    def test_empty_input_returns_empty_string(self) -> None:
        assert normalize_for_tfidf("") == ""
        assert normalize_for_tfidf(None) == ""

    def test_lowercases(self) -> None:
        assert "free" in normalize_for_tfidf("FREE MONEY NOW")

    def test_url_is_replaced_with_placeholder(self) -> None:
        result = normalize_for_tfidf("Click here: https://scam.example.com/win now")
        assert "urltoken" in result
        assert "scam.example.com" not in result

    def test_email_address_is_replaced_with_placeholder(self) -> None:
        result = normalize_for_tfidf("Contact me at someone@example.com for details")
        assert "emailtoken" in result

    def test_currency_is_replaced_with_placeholder(self) -> None:
        result = normalize_for_tfidf("You won $1,000.00 today")
        assert "currencytoken" in result

    def test_numbers_are_replaced_with_placeholder(self) -> None:
        result = normalize_for_tfidf("Call 1234567890 now")
        assert "numbertoken" in result

    def test_punctuation_is_removed(self) -> None:
        result = normalize_for_tfidf("Hello!!! Are you there??")
        assert "!" not in result
        assert "?" not in result

    def test_non_ascii_text_does_not_crash(self) -> None:
        result = normalize_for_tfidf("日本語のテキスト café naïve")
        assert isinstance(result, str)

    def test_very_long_text_does_not_crash(self) -> None:
        result = normalize_for_tfidf("spam " * 50_000)
        assert isinstance(result, str)
        assert len(result) > 0


class TestNormalizeForEmbedding:
    def test_empty_input_returns_empty_string(self) -> None:
        assert normalize_for_embedding("") == ""
        assert normalize_for_embedding(None) == ""

    def test_preserves_casing_and_punctuation(self) -> None:
        text = "Hello World! Is this Spam?"
        result = normalize_for_embedding(text)
        assert result == text

    @pytest.mark.parametrize(
        "text",
        [
            "日本語のメール本文です",
            "Café société naïve",
            "😀 emoji only message 🚀",
        ],
    )
    def test_non_ascii_round_trips_safely(self, text: str) -> None:
        result = normalize_for_embedding(text)
        assert isinstance(result, str)
        assert len(result) > 0
