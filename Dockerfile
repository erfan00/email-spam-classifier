# Serves the FastAPI spam-classifier API (+ static demo UI) with the TF-IDF
# baseline model. Build an image that includes a trained model artifact by
# training before `docker build` (see README) -- the artifacts/ directory is
# copied into the image as-is.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only the base (non-"embedding") dependencies: fast, small image,
# matching the baseline model that ships by default. See README for how to
# build an "embedding"-extra image if you need the transformer model served too.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY static ./static
COPY artifacts ./artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "email_spam_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
