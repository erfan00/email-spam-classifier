"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    text: str = Field(
        default="",
        description="Raw email text (subject + body, or just body) to classify.",
        examples=["Subject: WIN A FREE IPHONE NOW!!! Click here to claim your prize..."],
    )
    model: Literal["baseline", "embedding"] = Field(
        default="baseline",
        description="Which trained model to use: the fast TF-IDF baseline, or "
        "the (optional, heavier) transformer-embedding model.",
    )


class PredictResponse(BaseModel):
    """Response body for ``POST /predict``."""

    label: Literal["spam", "ham"]
    spam_probability: float = Field(..., ge=0.0, le=1.0)
    model: str = Field(..., description="Name of the model that produced this prediction.")


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: Literal["ok"]
    baseline_available: bool
    embedding_available: bool
