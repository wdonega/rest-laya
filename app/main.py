import hmac
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import (
    api_token,
    configured_models,
    get_router,
    jev_compat_enabled,
    max_concurrent_predictions,
)
from app.models import (
    ErrorResponse,
    ListModelsResponse,
    ModelMetadata,
    PredictRequest,
    PredictResponse,
    HealthResponse,
    Usage,
    to_response_model_name,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rest-laya")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force the preload on container startup. If loading fails, we want the
    # process to die here rather than silently failing on the first request —
    # that way the orchestrator (k8s/etc) won't mark the pod as ready with
    # a broken model.
    logger.info("Loading Laya Router, preloading %s...", configured_models())
    get_router()
    logger.info("Router loaded successfully.")
    yield


# With auth on, the interactive docs and the OpenAPI schema would be the one
# thing left open to anonymous callers -- so they're only served with auth off.
_docs_enabled = api_token() is None

app = FastAPI(
    title="rest-laya",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

# HTTP status -> jev exception-class name, used as the error envelope's "error_type".
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_denied_error",
    404: "not_found_error",
    422: "unprocessable_entity_error",
    429: "rate_limit_error",
}

_prediction_slots = threading.BoundedSemaphore(max_concurrent_predictions())

# Documents the error envelope in the OpenAPI schema for the predict routes.
_ERROR_RESPONSES = {status: {"model": ErrorResponse} for status in (400, 401, 422, 500)}


# Sent on every response. jev's SDKs read it into the response's and the
# exception's requestId(); it's also logged with unhandled errors, so a
# client report can be matched to the server log.
REQUEST_ID_HEADER = "x-typesafe-request-id"


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request.state.request_id
    return response


def _error_body(status_code: int, message: str, field_path: str | None = None) -> dict:
    error_type = _ERROR_TYPES.get(status_code, "internal_server_error" if status_code >= 500 else "api_error")
    detail = {"error_type": error_type, "message": message}
    if field_path is not None:
        detail["field_path"] = field_path
    return {"detail": detail}


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Covers, among other things, an invalid `model` value in the request
    # body. Same list FastAPI returns by default (which jev returns too),
    # trimmed to the fields jev's SDKs read, and without the "Value error, "
    # prefix pydantic puts on messages from our own validators.
    errors = [
        {
            "type": error["type"],
            "loc": list(error["loc"]),
            "msg": error["msg"].removeprefix("Value error, "),
            "input": error.get("input"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": errors}))


# Registered on Starlette's base class, not FastAPI's subclass: routing
# errors (404 unknown path, 405 wrong method) raise the base class, and
# they'd otherwise come back as a bare {"detail": "Not Found"}.
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # `detail` is either a plain message or {"message": ..., "field_path": ...}
    # when the raise site wants to name the offending field.
    if isinstance(exc.detail, dict):
        message, field_path = exc.detail["message"], exc.detail.get("field_path")
    else:
        message, field_path = str(exc.detail), None
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.status_code, message, field_path),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Last resort, so even unexpected failures keep the JSON error envelope
    # instead of Starlette's plain-text "Internal Server Error".
    # This handler runs outside the request-id middleware, so it sets the
    # header itself.
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    logger.exception("Unhandled error on %s %s (request id %s)", request.method, request.url.path, request_id)
    return JSONResponse(
        status_code=500,
        content=_error_body(500, "internal server error"),
        headers={REQUEST_ID_HEADER: request_id},
    )


def require_auth(authorization: Annotated[Optional[str], Header()] = None) -> None:
    """Validate `Authorization: Bearer <token>` against LAYA_API_TOKEN.

    LAYA_API_TOKEN unset/empty -- auth is disabled, every request passes.
    Used as a route dependency so it runs before body validation: an
    unauthenticated caller gets a 401, never schema details in a 422.
    """
    expected = api_token()
    if expected is None:
        return
    provided = authorization[len("Bearer "):] if authorization and authorization.startswith("Bearer ") else ""
    # Constant-time compare so the token can't be timed out of the service.
    # Compare bytes: compare_digest raises TypeError on non-ASCII str.
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    try:
        get_router()
        models_loaded = [to_response_model_name(name) for name in configured_models()]
        return HealthResponse(status="ok", router_loaded=True, models_loaded=models_loaded)
    except Exception:
        logger.exception("Health check failed")
        # Non-2xx so the container healthcheck (and any load balancer) fails.
        response.status_code = 503
        return HealthResponse(status="error", router_loaded=False)


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses=_ERROR_RESPONSES,
    dependencies=[Depends(require_auth)],
)
def predict(req: PredictRequest) -> PredictResponse:
    router = get_router()
    available = configured_models()

    if req.model is None:
        model = router.default
    elif req.model not in available:
        # Valid name (english/multilingual/typed-decisions, with or without
        # "laya/") but not one of the checkpoints this instance actually
        # preloaded via LAYA_MODELS.
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"model '{req.model}' is not configured on this "
                f"instance; available models: {available}",
                "field_path": "model",
            },
        )
    else:
        model = req.model

    try:
        # Requests beyond the limit wait here (in FastAPI's threadpool)
        # instead of competing for the same CPU cores.
        with _prediction_slots:
            result = router.predict(req.state, req.questions, model=model)
    except (ValueError, KeyError, TypeError) as e:
        # laya signals a bad request (malformed question, too many options,
        # ...) with these -- the caller's fault, so report it back.
        logger.warning("Laya rejected the request: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    # Anything else (OOM, CUDA, bugs) falls through to the 500 handler, so
    # internal details aren't leaked as a client error.

    return PredictResponse(
        model=to_response_model_name(model),
        answers=result["answers"],
        usage=Usage(**result["usage"]),
    )


# Date of the first commit to convaiinnovations/laya on Hugging Face, as an
# ISO-8601 timestamp with offset (the format jev returns and its SDKs
# document). Laya ships no per-checkpoint release date, so every /v1/models
# entry reports this one.
_RELEASE_DATE = "2026-09-18T05:13:12+00:00"

# Checkpoint -> the Hugging Face repo it comes from, for /v1/models.
_MODEL_SOURCES = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya, multilingual",
    "typed-decisions": "convaiinnovations/laya, typed-decisions",
}


def list_models() -> ListModelsResponse:
    """The checkpoints this instance preloaded, default first, plus jev-latest.

    Names are the "laya/..." form the predict response reports, and each is
    accepted as `model`. jev-latest is listed because jev clients default to
    it; here it resolves to the default checkpoint.
    """
    names = configured_models()
    models = [
        ModelMetadata(
            name=to_response_model_name(name),
            description=f"Laya {name} checkpoint ({_MODEL_SOURCES[name]})",
            release_date=_RELEASE_DATE,
        )
        for name in names
    ]
    models.append(
        ModelMetadata(
            name="jev-latest",
            description=f"jev compatibility alias for the default checkpoint, {to_response_model_name(names[0])}",
            release_date=_RELEASE_DATE,
        )
    )
    return ListModelsResponse(models=models)


# jev's own endpoint paths only exist when jev compat is on: /v1/systemone
# (same handler as /predict, registered as a second route) and /v1/models.
# When compat is off, both 404 like any other unknown route.
if jev_compat_enabled():
    app.add_api_route(
        "/v1/systemone",
        predict,
        methods=["POST"],
        response_model=PredictResponse,
        responses=_ERROR_RESPONSES,
        dependencies=[Depends(require_auth)],
    )
    app.add_api_route(
        "/v1/models",
        list_models,
        methods=["GET"],
        response_model=ListModelsResponse,
        responses={401: {"model": ErrorResponse}},
        dependencies=[Depends(require_auth)],
    )
