"""FastAPI application exposing the spam classifier.

Run locally with:
    uvicorn email_spam_classifier.api.main:app --reload

Or via the installed console script:
    espam-serve
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from email_spam_classifier import __version__
from email_spam_classifier.api.schemas import HealthResponse, PredictRequest, PredictResponse
from email_spam_classifier.config import PROJECT_ROOT
from email_spam_classifier.predict import ModelNotTrainedError, is_model_available, predict_text

app = FastAPI(
    title="Email Spam Classifier",
    description="A 2026 rebuild of a 2016 SVM spam classifier: TF-IDF/linear-model "
    "baseline plus an optional transformer-embedding model, served over HTTP.",
    version=__version__,
)

_STATIC_DIR = PROJECT_ROOT / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_demo() -> FileResponse:
    """Serve the minimal static HTML demo front-end at the site root."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Demo front-end not found.")
    return FileResponse(index_path)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe that also reports which models are trained."""
    return HealthResponse(
        status="ok",
        baseline_available=is_model_available("baseline"),
        embedding_available=is_model_available("embedding"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify a piece of email text as spam or ham.

    Accepts empty strings, non-ASCII text, and very long text (internally
    truncated) -- see :mod:`email_spam_classifier.data.preprocess`.
    """
    try:
        result = predict_text(request.text, model_name=request.model)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(
        label=result.label,  # type: ignore[arg-type]
        spam_probability=result.spam_probability,
        model=result.model_name,
    )


def run() -> None:
    """Entry point for the ``espam-serve`` console script."""
    import uvicorn

    uvicorn.run(
        "email_spam_classifier.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
