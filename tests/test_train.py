"""Tests for the data loader and training/evaluation pipeline.

Uses small synthetic fixtures (see conftest.py) instead of the real,
multi-megabyte SpamAssassin corpus so the suite stays fast and does not
require any downloaded data to run in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from email_spam_classifier.data.loader import build_dataset
from email_spam_classifier.models.baseline import build_baseline_pipeline
from email_spam_classifier.train import evaluate_predictions, train_baseline


class TestBuildDataset:
    def test_parses_labeled_messages(self, synthetic_raw_corpus: Path) -> None:
        df = build_dataset(synthetic_raw_corpus)
        assert set(df.columns) >= {"text", "label", "source", "filename"}
        assert len(df) == 20
        assert set(df["label"].unique()) == {0, 1}

    def test_skips_cmds_bookkeeping_file(self, synthetic_raw_corpus: Path) -> None:
        df = build_dataset(synthetic_raw_corpus)
        assert "cmds" not in set(df["filename"])

    def test_raises_if_raw_dir_missing(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            build_dataset(tmp_path / "does_not_exist")

    def test_subject_is_included_in_text(self, synthetic_raw_corpus: Path) -> None:
        df = build_dataset(synthetic_raw_corpus)
        assert df["text"].str.contains("Test").any()


class TestBaselinePipeline:
    def test_build_returns_unfitted_pipeline_with_expected_steps(self) -> None:
        pipeline = build_baseline_pipeline("logreg")
        assert [name for name, _ in pipeline.steps] == ["tfidf", "clf"]

    def test_fit_predict_round_trip(self, synthetic_dataset: pd.DataFrame) -> None:
        pipeline = build_baseline_pipeline("logreg")
        pipeline.fit(synthetic_dataset["text"].tolist(), synthetic_dataset["label"].to_numpy())
        preds = pipeline.predict(synthetic_dataset["text"].tolist())
        # The synthetic classes are clearly separable; the model should fit them well.
        accuracy = (preds == synthetic_dataset["label"].to_numpy()).mean()
        assert accuracy >= 0.9

    def test_predict_proba_sums_to_one(self, synthetic_dataset: pd.DataFrame) -> None:
        pipeline = build_baseline_pipeline("logreg")
        pipeline.fit(synthetic_dataset["text"].tolist(), synthetic_dataset["label"].to_numpy())
        proba = pipeline.predict_proba(["free money winner click now"])
        assert proba.shape == (1, 2)
        assert np.isclose(proba.sum(), 1.0)

    def test_linear_svm_variant_fits(self, synthetic_dataset: pd.DataFrame) -> None:
        pipeline = build_baseline_pipeline("linear_svm")
        pipeline.fit(synthetic_dataset["text"].tolist(), synthetic_dataset["label"].to_numpy())
        proba = pipeline.predict_proba(synthetic_dataset["text"].tolist())
        assert proba.shape == (len(synthetic_dataset), 2)


class TestEvaluatePredictions:
    def test_reports_expected_keys(self) -> None:
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 1, 1, 1, 0]
        y_proba = [0.1, 0.6, 0.9, 0.8, 0.3]
        metrics = evaluate_predictions(y_true, y_pred, y_proba)
        assert set(metrics) == {
            "accuracy",
            "precision_spam",
            "recall_spam",
            "f1_spam",
            "roc_auc",
            "confusion_matrix",
        }
        assert 0.0 <= metrics["accuracy"] <= 1.0  # type: ignore[operator]


class TestTrainBaselineEndToEnd:
    def test_train_baseline_produces_sane_metrics(
        self, synthetic_dataset: pd.DataFrame, tmp_path: Path, monkeypatch
    ) -> None:
        import email_spam_classifier.train as train_module

        monkeypatch.setattr(train_module, "ARTIFACTS_DIR", tmp_path)
        monkeypatch.setattr(train_module, "BASELINE_MODEL_PATH", tmp_path / "baseline.joblib")

        results = train_baseline(synthetic_dataset, test_size=0.3)

        assert results["model"] == "baseline_tfidf_logreg"
        assert 0.0 <= results["accuracy"] <= 1.0
        assert (tmp_path / "baseline.joblib").exists()
