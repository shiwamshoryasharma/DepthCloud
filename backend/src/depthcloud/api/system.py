import sys
import time

import psutil
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["system"])


def system_status(app):
    return {
        **app.state.models.status(),
        "uptime_seconds": round(time.monotonic() - app.state.started_at, 1),
        "python": sys.version.split()[0],
        "memory_mb": round(psutil.Process().memory_info().rss / 1024**2, 1),
        "stream_clients": app.state.stream_clients,
    }


@router.get("/health")
def health():
    return {"status": "ok", "application": "DepthCloud", "version": "0.1.0"}


@router.get("/system")
def system(request: Request):
    return system_status(request.app)

