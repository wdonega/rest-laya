"""
Loads the Laya Router once at process startup and keeps the checkpoints
in memory (preload=True). This avoids the 7-10s reload cost every time
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


def _configured_models() -> list[str]:
    raw = os.environ.get("LAYA_MODELS", "english")
    return [name.strip() for name in raw.split(",") if name.strip()]


def get_router() -> Router:
    global router
    if router is None:
        names = _configured_models()
        router = Router(default=names[0])
        router.preload(names=names)
    return router
