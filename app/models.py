"""
The sidecar's REST contract. `state` and `questions` are kept as free-form
dicts (passthrough) because the shape of typed questions (choice/score/noul)
is defined by Laya, not by us — so the sidecar doesn't need updating every
time the laya package changes its question schema.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

ModelName = Literal["laya", "laya-multilingual", "typed-decisions"]


class PredictRequest(BaseModel):
    state: dict[str, Any] = Field(
        ..., description="The text/state to evaluate, e.g. {'body': '...'}"
    )
    questions: dict[str, Any] = Field(
        ..., description="Typed questions (choice/score/noul) in Laya's format"
    )
    model: Optional[ModelName] = Field(
        default=None,
        description=(
            "Checkpoint to use: laya (english), laya-multilingual, "
            "typed-decisions. Omit to use the first checkpoint configured "
            "via LAYA_MODELS (defaults to laya/english)."
        ),
    )


class PredictResponse(BaseModel):
    answers: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    router_loaded: bool
