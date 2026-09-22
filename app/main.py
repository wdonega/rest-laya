import hmac
import logging
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
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

# HTTP status -> jev exception-class name, used as the error envelope's "type".
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


def _error_body(status_code: int, message: str, field_path: str | None = None) -> dict:
    error_type = _ERROR_TYPES.get(status_code, "internal_server_error" if status_code >= 500 else "api_error")
    body = {"message": message, "type": error_type}
    if field_path is not None:
        body["field_path"] = field_path
    return {"error": body}


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Covers, among other things, an invalid `model` value in the request
    # body. jev's docs describe `field_path` as "a dotted path to the
    # offending field" -- pydantic's error `loc` is exactly that, minus the
    # leading "body".
    first = exc.errors()[0]
    if first["type"] == "json_invalid":
        # loc here is ("body", <char offset>) -- not a field, so no field_path.
        field_path = None
    else:
        field_path = ".".join(str(p) for p in first["loc"] if p != "body") or None
    # pydantic prefixes messages from our own validators with "Value error, ".
    message = first["msg"].removeprefix("Value error, ")
    return JSONResponse(status_code=422, content=_error_body(422, message, field_path))


# Registered on Starlette's base class, not FastAPI's subclass: routing
# errors (404 unknown path, 405 wrong method) raise the base class, and
# they'd otherwise skip the envelope and come back as {"detail": ...}.
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
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content=_error_body(500, "internal server error"))


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


# /v1/systemone (jev's own endpoint path) only exists when jev compat is
# on -- same handler as /predict, registered as a second route. When
# compat is off, that path 404s like any other unknown route.
if jev_compat_enabled():
    app.add_api_route(
        "/v1/systemone",
        predict,
        methods=["POST"],
        response_model=PredictResponse,
        responses=_ERROR_RESPONSES,
        dependencies=[Depends(require_auth)],
    )
