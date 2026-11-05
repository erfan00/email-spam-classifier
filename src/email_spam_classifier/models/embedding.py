"""Modern alternative: small transformer sentence embeddings + a linear classifier head.

Instead of hand-built or bag-of-words features, this model encodes each email
with a small pretrained sentence-transformer (``all-MiniLM-L6-v2``, 384-dim,
~80MB) and fits a cheap Logistic Regression head on top of the resulting dense
vectors. This captures semantic/contextual signal (paraphrased spam, synonyms,
word order) that TF-IDF cannot, at the cost of needing to download model
weights once and run a forward pass per prediction.

``sentence-transformers`` (and its ``torch`` dependency) are an optional extra
(``pip install -e ".[embedding]"``) because they are heavy and not required for
the fast TF-IDF baseline or for running the API with the baseline model. The
import is therefore deferred to first use, with a clear error message if the
extra is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from email_spam_classifier.config import EMBEDDING_MODEL_NAME, RANDOM_STATE
from email_spam_classifier.data.preprocess import normalize_for_embedding

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

_EMBEDDER_CACHE: dict[str, SentenceTransformer] = {}


def _load_sentence_transformers() -> Any:
    try:
        import sentence_transformers
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The embedding model requires the optional 'embedding' extra. "
            'Install it with: pip install -e ".[embedding]"'
        ) from exc
    return sentence_transformers


def get_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Return a process-wide cached :class:`SentenceTransformer` instance.

    Loading the model (downloading weights on first use, then reading them from
    the local HuggingFace cache) is the expensive part of using this model
    family, so we only do it once per process.
    """
    if model_name not in _EMBEDDER_CACHE:
        st_module = _load_sentence_transformers()
        # Force CPU explicitly. Some PyTorch/macOS MPS (Apple GPU) backend
        # combinations crash on certain linear-layer shapes used by this
        # model; CPU is also plenty fast for a model this small (384-dim,
        # 6 layers) at the batch sizes used here.
        _EMBEDDER_CACHE[model_name] = st_module.SentenceTransformer(model_name, device="cpu")
    return _EMBEDDER_CACHE[model_name]


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> NDArray[np.float32]:
    """Encode a batch of raw texts into dense sentence-embedding vectors."""
    cleaned = [normalize_for_embedding(t) or " " for t in texts]
    embedder = get_embedder(model_name)
    vectors = embedder.encode(
        cleaned,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


class EmbeddingClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn-compatible classifier: sentence embeddings + Logistic Regression.

    Deliberately mirrors the ``fit(X, y)`` / ``predict(X)`` / ``predict_proba(X)``
    interface of the baseline pipeline, where ``X`` is a list/array of raw
    strings, so both models are interchangeable from the training script's and
    the API's point of view. Only the lightweight classifier head (not the
    transformer weights) is persisted by :mod:`joblib` -- the embedder is
    re-loaded (and cached) by name at inference time.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        C: float = 1.0,
        max_iter: int = 2000,
    ) -> None:
        self.model_name = model_name
        self.C = C
        self.max_iter = max_iter

    def fit(self, X: list[str], y: NDArray[np.int_]) -> EmbeddingClassifier:
        embeddings = embed_texts(list(X), self.model_name)
        self.classifier_ = LogisticRegression(
            max_iter=self.max_iter,
            class_weight="balanced",
            C=self.C,
            random_state=RANDOM_STATE,
        )
        self.classifier_.fit(embeddings, y)
        self.classes_ = self.classifier_.classes_
        return self

    def predict(self, X: list[str]) -> NDArray[np.int_]:
        embeddings = embed_texts(list(X), self.model_name)
        result: NDArray[np.int_] = self.classifier_.predict(embeddings)
        return result

    def predict_proba(self, X: list[str]) -> NDArray[np.float64]:
        embeddings = embed_texts(list(X), self.model_name)
        result: NDArray[np.float64] = self.classifier_.predict_proba(embeddings)
        return result


def build_embedding_pipeline() -> EmbeddingClassifier:
    """Construct the (unfitted) embedding + Logistic Regression classifier."""
    return EmbeddingClassifier()
