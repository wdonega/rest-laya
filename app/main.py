import logging

from fastapi import FastAPI, HTTPException

from app.config import get_router
from app.models import PredictRequest, PredictResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rest-laya")

app = FastAPI(title="rest-laya", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    # Force the preload on container startup. If loading fails, we want the
    # process to die here rather than silently failing on the first request —
    # that way the orchestrator (k8s/etc) won't mark the pod as ready with
    # a broken model.
    logger.info("Loading Laya Router (preload=True)...")
    get_router()
    logger.info("Router loaded successfully.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_router()
        return HealthResponse(status="ok", router_loaded=True)
    except Exception:
        return HealthResponse(status="error", router_loaded=False)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    router = get_router()
    try:
        result = router.predict(req.state, req.questions, model=req.model or router.default)
    except Exception as e:
        logger.exception("Failed to run Laya predict")
        raise HTTPException(status_code=422, detail=str(e)) from e

    return PredictResponse(answers=result["answers"])
