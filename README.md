# Email Spam Classifier (2026 rebuild)

A from-scratch, 2026 rebuild of a 2016 university project that used an SVM over
hand-built feature vectors to classify spam email. This version keeps the same
problem (binary spam/ham classification) and the same family of linear model
for the baseline, but modernizes the tooling: a typed Python package, a
scikit-learn pipeline instead of hand-rolled features, an optional transformer-
embedding model, a FastAPI service, tests, CI, and a Dockerfile.

## Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Training](#training)
- [Running the API + demo](#running-the-api--demo)
- [Example request](#example-request)
- [Docker](#docker)
- [Tests, lint, type-checking](#tests-lint-type-checking)
- [CI](#ci)
- [Limitations](#limitations)
- [License](#license)

## Overview

Given raw email text (subject + body), predict whether it is **spam** or
**ham** (not spam), and return a calibrated-ish spam probability. The project
provides two interchangeable models behind the same interface:

1. **Baseline**: TF-IDF features + a linear classifier (Logistic Regression,
   with a Linear SVM variant available). This is the direct descendant of the
   2016 project's approach -- still a bag-of-words representation and a linear
   decision boundary -- but built on a reproducible, well-tested scikit-learn
   pipeline instead of hand-rolled feature extraction.
2. **Embedding**: a small pretrained sentence-transformer
   (`all-MiniLM-L6-v2`, 384-dim) encodes each email into a dense vector, and a
   Logistic Regression head is trained on top. This captures semantic
   similarity (paraphrased spam, synonyms) that a bag-of-words model misses,
   at the cost of needing to run a transformer forward pass per prediction.

Both models share the same training entry point, the same evaluation metrics,
and the same `/predict` API contract -- you can switch between them with a
single request field.

## Dataset

**Primary dataset: the [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/)**
(real, raw emails from the early-2000s SpamAssassin project).

- **Provenance**: donated message corpus hosted by the Apache Software
  Foundation (SpamAssassin project) at
  `https://spamassassin.apache.org/old/publiccorpus/`.
- **License**: the corpus page states the messages are made available "for
  use in testing spam filtering systems" and are donated by individuals; it is
  widely used in academic/open-source anti-spam research (this is, in fact,
  close in spirit to what the original 2016 project likely used). It is **not
  committed to this repository** -- `data/raw/` and `data/processed/` are
  gitignored. Run `scripts/fetch_data.py` to download it yourself.
- **Archives used**: `20030228_easy_ham(.tar.bz2)`, `20030228_easy_ham_2`,
  `20030228_hard_ham`, `20030228_spam`, `20030228_spam_2`.
- **Size after parsing/dedup**: ~5,800 emails, roughly **69% ham / 31% spam**
  (see [Results](#results) for the exact split used in the run below) --
  i.e. a **moderately imbalanced** binary classification problem. This is why
  the README (and the training script) reports precision/recall/F1 for the
  spam class and ROC-AUC, not just accuracy: a model that always predicts
  "ham" would score ~69% accuracy while catching zero spam.
- **Documented fallback**: if the SpamAssassin archives are ever unreachable
  in your environment, the
  [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
  (UCI Machine Learning Repository, CC-BY license, ~5,500 SMS messages
  labeled spam/ham) is a documented, equally imbalanced short-text fallback
  that drops into the same `text,label` CSV schema (`data/processed/dataset.csv`)
  with minimal changes to `scripts/build_dataset.py`. We used the full email
  corpus successfully in this environment, so the fallback was not needed.

### Rebuilding the dataset

```bash
python scripts/fetch_data.py     # downloads + extracts the raw corpus into data/raw/
python scripts/build_dataset.py  # parses raw emails into data/processed/dataset.csv
```

`email_spam_classifier.train` will also build the dataset automatically on
first run if `data/processed/dataset.csv` doesn't exist yet (as long as
`data/raw/` has already been fetched).

## Models

| | Baseline (TF-IDF + Logistic Regression) | Embedding (MiniLM + Logistic Regression) |
|---|---|---|
| Representation | Bag-of-words TF-IDF (1-2 grams), URL/number/currency folded to placeholders | 384-dim sentence-transformer embedding |
| Classifier | Logistic Regression (`class_weight="balanced"`) | Logistic Regression (`class_weight="balanced"`) on top of embeddings |
| Training time (this corpus, 4,708 emails) | ~1.9 seconds | ~42 seconds (CPU; dominated by encoding all training emails) |
| Inference latency (measured) | ~0.7 ms/email | ~17 ms/email (CPU; transformer forward pass dominates) |
| Install footprint | scikit-learn + pandas (small) | + `sentence-transformers` + `torch` (large, optional extra) |
| Captures semantics/paraphrase | No -- relies on shared vocabulary | Yes -- nearby meaning -> nearby vector |
| Interpretability | High (inspect TF-IDF weights per token) | Low (opaque dense vector) |
| When to prefer | Default choice: fast, cheap, easy to deploy anywhere, strong accuracy on this corpus | When vocabulary-based features are known to be brittle (e.g. heavily obfuscated/paraphrased spam) and you can afford the extra latency/dependencies |

The embedding model is an **optional extra** (`pip install -e ".[embedding]"`)
so the default install stays lightweight; the API and tests run fine with
only the baseline model trained.

## Results

Metrics from a real run of `python -m email_spam_classifier.train --model both`
against the full SpamAssassin corpus parsed via `scripts/build_dataset.py`
(5,885 emails: 4,104 ham / 1,781 spam; 80/20 stratified train/test split, seed
42). Raw machine-generated output lives in `artifacts/metrics.json`.

| Metric | Baseline (TF-IDF + LogReg) | Embedding (MiniLM + LogReg) |
|---|---|---|
| Accuracy | 0.983 | 0.956 |
| Precision (spam) | 0.975 | 0.915 |
| Recall (spam) | 0.969 | 0.941 |
| F1 (spam) | 0.972 | 0.928 |
| ROC-AUC | 0.996 | 0.990 |
| Confusion matrix (ham/spam rows) | `[[812, 9], [11, 345]]` | `[[790, 31], [21, 335]]` |
| Train time (4,708 emails) | 1.9 s | 42.2 s |
| Inference latency | 0.7 ms/email | 16.8 ms/email |

**Why not just accuracy?** The dataset is imbalanced (~70% ham / 30% spam).
Accuracy alone rewards a trivial "always predict ham" classifier with ~70%
accuracy while it misses 100% of spam (recall = 0). Precision/recall/F1 on the
spam (positive) class make the actual filtering behavior visible: precision
tells you how often a "spam" flag is correct (false positives block real
mail, which is costly), and recall tells you what fraction of actual spam is
caught.

**An honest, slightly surprising result**: on *this* corpus, the simple
TF-IDF baseline actually beats the transformer-embedding model on every
metric, while also being ~20x faster to train and ~24x faster at inference.
This is a real, reproducible finding here, not a universal law -- early-2000s
SpamAssassin spam is heavily lexical (dollar signs, "FREE", "WINNER", ALL
CAPS, excessive punctuation), which is exactly the kind of signal a bag-of-
words TF-IDF model is built to exploit. A general-purpose sentence embedding
has no special incentive to weight those surface cues, and `all-MiniLM-L6-v2`
also truncates input to 256 word-piece tokens, discarding the tail of longer
emails that TF-IDF still sees in full. Embedding-based approaches tend to pull
ahead on messages where the *spamminess* is semantic/contextual rather than
lexical (paraphrased scams, subtle social-engineering text) -- which is why
both models are kept here, with the API able to switch between them per
request, rather than presenting only the model that "wins" on this one
dataset.

## Project layout

```
email-spam-classifier/
├── src/email_spam_classifier/
│   ├── config.py              # paths, constants, dataset source URLs
│   ├── data/
│   │   ├── loader.py          # parses raw SpamAssassin corpus -> DataFrame
│   │   └── preprocess.py      # shared text cleaning + model-specific normalization
│   ├── models/
│   │   ├── baseline.py        # TF-IDF + Logistic Regression / Linear SVM pipeline
│   │   ├── embedding.py       # sentence-transformer + Logistic Regression head
│   │   └── common.py          # shared PredictionResult type
│   ├── train.py                # training/evaluation CLI, writes artifacts/*.joblib + metrics.json
│   ├── predict.py              # inference wrapper (model loading/caching, edge cases)
│   └── api/
│       ├── main.py             # FastAPI app (/predict, /health, demo front-end)
│       └── schemas.py          # pydantic request/response models
├── static/index.html           # minimal JS front-end for the /predict endpoint
├── scripts/
│   ├── fetch_data.py           # downloads + extracts the SpamAssassin corpus
│   └── build_dataset.py        # raw corpus -> data/processed/dataset.csv
├── tests/                      # pytest: preprocessing, training, inference, API, edge cases
├── artifacts/                  # trained model + metrics.json (gitignored, regenerate via train)
├── data/                       # raw + processed dataset (gitignored, regenerate via scripts/)
├── .github/workflows/ci.yml    # lint + type-check + test on push/PR
├── Dockerfile
├── pyproject.toml              # deps, ruff/black/mypy/pytest config
└── LICENSE                     # MIT
```

## Setup

Requires Python 3.10+.

```bash
cd email-spam-classifier
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # baseline model + dev tooling (pytest/ruff/black/mypy)
# Optional, for the transformer-embedding model:
pip install -e ".[embedding]"
```

## Training

```bash
# 1. Get the data (skip if data/processed/dataset.csv already exists)
python scripts/fetch_data.py
python scripts/build_dataset.py

# 2. Train the baseline (fast, seconds)
python -m email_spam_classifier.train --model baseline

# 3. (Optional) train the embedding model too -- requires the `embedding` extra
python -m email_spam_classifier.train --model embedding

# Or train both in one go:
python -m email_spam_classifier.train --model both
```

This writes `artifacts/baseline_tfidf_logreg.joblib` (and/or
`artifacts/embedding_classifier.joblib`) plus `artifacts/metrics.json`.

## Running the API + demo

```bash
# After training at least the baseline model:
uvicorn email_spam_classifier.api.main:app --reload
# or, via the installed console script:
espam-serve
```

Then open **http://localhost:8000/** for the minimal HTML demo (paste an
email, pick a model, click "Classify"), or **http://localhost:8000/docs** for
the interactive FastAPI/Swagger docs.

## Example request

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER!! Claim your FREE prize now, click here urgent limited offer", "model": "baseline"}'
```

Example response:

```json
{
  "label": "spam",
  "spam_probability": 0.97,
  "model": "baseline"
}
```

A ham example:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hey, are we still meeting for lunch tomorrow at noon?", "model": "baseline"}'
```

```json
{
  "label": "ham",
  "spam_probability": 0.03,
  "model": "baseline"
}
```

`GET /health` reports which trained models are available:

```bash
curl -s http://localhost:8000/health
# {"status": "ok", "baseline_available": true, "embedding_available": false}
```

## Docker

The image bundles the baseline model (train it first so `artifacts/` is
populated before building):

```bash
python -m email_spam_classifier.train --model baseline   # make sure artifacts/ exists
docker build -t email-spam-classifier .
docker run --rm -p 8000:8000 email-spam-classifier
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Free cash now, click here!!!", "model": "baseline"}'
```

The image installs only the base dependencies (no `torch`/`sentence-transformers`),
so it only serves the baseline model unless you extend the Dockerfile with the
`embedding` extra and bake in a trained embedding artifact.

## Tests, lint, type-checking

```bash
pytest                       # unit tests: preprocessing, training, inference, API, edge cases
ruff check .                 # lint
black --check .              # format check
mypy src                     # static types
```

Tests do not depend on the real downloaded corpus -- they synthesize tiny,
clearly-separable spam/ham fixtures so the suite is fast and fully
self-contained (see `tests/conftest.py`).

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`: `ruff check`,
`black --check`, `mypy`, and `pytest` (with coverage) on Python 3.11 and 3.12.

## Limitations

- The SpamAssassin corpus is from the early 2000s; spam tactics have evolved
  substantially since then (the model will not have seen, e.g., modern
  phishing templates, SMS-style spam, or multilingual spam campaigns). Treat
  this as a demonstration of the pipeline, not a production-ready filter.
- The embedding model's Logistic Regression head is trained on a fixed,
  frozen embedding model; no fine-tuning of the transformer itself is done.
- `predict_proba` on the Logistic Regression baseline is a genuine model
  probability; the Linear SVM variant's probability is a Platt-scaled
  approximation (`CalibratedClassifierCV`), not a native one.
- No persistence/auth/rate-limiting on the API -- it's a demo service, not a
  hardened production deployment.
- Email parsing (`src/email_spam_classifier/data/loader.py`) extracts a
  best-effort plain-text view (subject + text/plain or stripped text/html
  parts) and silently skips a handful of malformed messages in the corpus;
  it does not attempt full MIME-structure-aware feature extraction (e.g.
  attachment analysis, header-based features) the way a production spam
  filter would.

## License

[MIT](LICENSE).
