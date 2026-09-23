"""
The sidecar's REST contract. `state` and `questions` are kept as
free-form passthrough (no fixed schema) because their shape (typed
questions like choice/score/noul, and what `state` can be) is defined by
Laya, not by us — so the sidecar doesn't need updating every time the
laya package changes its question schema.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from app.config import jev_compat_enabled

# The checkpoint names the router actually knows about.
ModelName = Literal["english", "multilingual", "typed-decisions"]

# Display form used in the response body's `model` field, e.g. "laya/english".
ResponseModelName = Literal["laya/english", "laya/multilingual", "laya/typed-decisions"]

# Prefix of jev's own model ids (jev-1.13.0, jev-latest, jev-preview, and
# any future version). Laya has no equivalent checkpoints for these -- when
# jev compat is on they just mean "use the default checkpoint", same as
# omitting `model` entirely.
JEV_MODEL_PREFIX = "jev-"


def to_response_model_name(name: ModelName) -> ResponseModelName:
    return f"laya/{name}"  # type: ignore[return-value]


def normalize_model_name(raw: str) -> ModelName:
    """Strip an optional "laya/" prefix and validate against ModelName.

    Accepts both "english" and "laya/english" so a caller can pass back
    either the bare checkpoint name or the response's own "laya/..." form.
    """
    name = raw[len("laya/"):] if raw.startswith("laya/") else raw
    if name not in ("english", "multilingual", "typed-decisions"):
        raise ValueError(
            f"invalid model {raw!r}; expected one of english, multilingual, "
            f"typed-decisions (with or without a 'laya/' prefix)"
        )
    return name  # type: ignore[return-value]


class PredictRequest(BaseModel):
    state: str | dict[str, Any] | list[Any] = Field(
        ...,
        description=(
            "The text/state to evaluate. A dict (e.g. {'body': '...'}), a "
            "plain string, or a list -- matches what laya.Router.predict "
            "accepts."
        ),
    )
    questions: dict[str, Any] = Field(
        ..., description="Typed questions (choice/score/noul) in Laya's format"
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Checkpoint to use, e.g. 'english' or 'laya/english' (the "
            "'laya/' prefix is optional). Omit to use the first checkpoint "
            "configured via LAYA_MODELS. When jev compat is on, any jev "
            "model id (jev-*, e.g. jev-latest or jev-1.13.0) is also "
            "accepted and resolves to that same default."
        ),
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value.startswith(JEV_MODEL_PREFIX):
            if jev_compat_enabled():
                return None  # defer to the router's default, like omitting `model`
            raise ValueError(
                f"invalid model {value!r}; jev compatibility is disabled "
                f"(LAYA_JEV_COMPAT=false), so jev model ids aren't accepted"
            )
        return normalize_model_name(value)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class PredictResponse(BaseModel):
    model: ResponseModelName = Field(
        ..., description="Checkpoint that actually served this request."
    )
    answers: dict[str, Any]
    usage: Usage


# /v1/models, in the shape jev's SDKs read: {"models": [{"name", "description",
# "release_date"}]}.
class ModelMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    release_date: Optional[str] = None


class ListModelsResponse(BaseModel):
    models: list[ModelMetadata]


class HealthResponse(BaseModel):
    status: str
    router_loaded: bool
    models_loaded: list[ResponseModelName] = Field(default_factory=list)


# jev's error envelope, as its SDKs parse it. Two shapes under "detail":
#   {"detail": {"error_type": ..., "message": ..., "field_path": ...}}
#     for errors the service raises. `error_type` mirrors jev's exception
#     naming (invalid_request_error, authentication_error, ...);
#     `field_path` is our addition, the dotted path to the offending field,
#     omitted when the error isn't field-specific. Clients ignore it.
#   {"detail": [{"type": ..., "loc": [...], "msg": ..., "input": ...}]}
#     for request-body validation errors (FastAPI's own format, which jev
#     returns too).
class ErrorDetail(BaseModel):
    error_type: str
    message: str
    field_path: Optional[str] = None


class ValidationErrorItem(BaseModel):
    type: str
    loc: list[str | int]
    msg: str
    input: Any = None


class ErrorResponse(BaseModel):
    detail: ErrorDetail | list[ValidationErrorItem]
