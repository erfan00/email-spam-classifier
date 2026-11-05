"""Classical baseline: TF-IDF features + a linear classifier (Logistic Regression).

This is the direct modern analogue of the original 2016 project's hand-built
feature vectors + SVM: TF-IDF is still a bag-of-words feature representation,
and Logistic Regression (or a linear SVM) is a linear classifier in the same
family as the original SVM. The upgrade is almost entirely in tooling and
reproducibility, not model sophistication -- which is the point of this
baseline: it is fast, cheap, interpretable, and a strong reference point.

``LogisticRegression`` is used by default because it gives calibrated-ish
probabilities out of the box via ``predict_proba``, which the API needs. A
``LinearSVC`` variant is also provided for comparison; it has no native
``predict_proba`` (a ``CalibratedClassifierCV`` wrapper is used to add one).
"""

from __future__ import annotations

from typing import Literal

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from email_spam_classifier.config import RANDOM_STATE
from email_spam_classifier.data.preprocess import normalize_for_tfidf

ClassifierKind = Literal["logreg", "linear_svm"]


def build_baseline_pipeline(classifier: ClassifierKind = "logreg") -> Pipeline:
    """Construct the TF-IDF + linear-model scikit-learn pipeline.

    Args:
        classifier: ``"logreg"`` for Logistic Regression (default, has native
            ``predict_proba``) or ``"linear_svm"`` for a calibrated Linear SVM.

    Returns:
        An unfitted scikit-learn :class:`~sklearn.pipeline.Pipeline` with a
        ``"tfidf"`` step and a ``"clf"`` step.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_for_tfidf,
        tokenizer=str.split,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    clf: LogisticRegression | CalibratedClassifierCV
    if classifier == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=5.0,
            random_state=RANDOM_STATE,
        )
    elif classifier == "linear_svm":
        base_svm = LinearSVC(class_weight="balanced", C=1.0, random_state=RANDOM_STATE)
        # LinearSVC has no predict_proba; calibrate with Platt scaling so the API
        # can still return a probability alongside the label.
        clf = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
    else:  # pragma: no cover - exhaustive Literal, defensive branch
        raise ValueError(f"Unknown classifier kind: {classifier!r}")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])
