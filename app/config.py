"""
Loads the Laya Router once at process startup and keeps the checkpoints
in memory (Router.preload). This avoids the 7-10s reload cost every time
we switch languages/checkpoints.

Which checkpoints get loaded is controlled by the LAYA_MODELS env var
(comma-separated, e.g. "english" or "english,multilingual"). Defaults to
just "english". Router.__init__ always seeds self.models with all of
DEFAULT_MODELS (english/multilingual/typed-decisions) even when `models=`
is passed, since that param merges rather than replaces -- so we also
have to pass `names=` to preload() to avoid building checkpoints we
don't want.
"""

import os

from laya import Router

router: Router | None = None


def configured_models() -> list[str]:
    """Checkpoint names enabled via LAYA_MODELS, in configured order.

    Also used by main.py to validate the request's `model` field: a
    request can only select a checkpoint that was actually preloaded.
    """
    raw = os.environ.get("LAYA_MODELS", "english")
    return [name.strip() for name in raw.split(",") if name.strip()]


def api_token() -> str | None:
    """Bearer token required on requests, from LAYA_API_TOKEN.

    Empty or unset -- auth is disabled and any request is accepted.
    """
    return os.environ.get("LAYA_API_TOKEN") or None


def max_concurrent_predictions() -> int:
    """How many predictions may run at once, from LAYA_MAX_CONCURRENT_PREDICTIONS.

    Defaults to 1. On CPU a single forward pass already uses every core, so
    running more in parallel only makes them fight over the same cores.
    Raise it on GPU, where batching concurrent requests pays off.
    """
    return max(1, int(os.environ.get("LAYA_MAX_CONCURRENT_PREDICTIONS") or 1))


def jev_compat_enabled() -> bool:
    """Whether jev API compatibility is on, from LAYA_JEV_COMPAT.

    Defaults to on, including when set but empty -- only an explicit
    false/0/no/off turns it off. When on: the /v1/systemone endpoint
    responds (it's a 404 when off), and `model` accepts any jev model id
    (jev-*), which resolves to the default checkpoint.
    """
    return os.environ.get("LAYA_JEV_COMPAT", "").strip().lower() not in ("false", "0", "no", "off")


def get_router() -> Router:
    global router
    if router is None:
        names = configured_models()
        router = Router(default=names[0])
        router.preload(names=names)
    return router
