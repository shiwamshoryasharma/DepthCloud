import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from depthcloud.api import camera, reconstruction, system
from depthcloud.camera.calibration import CalibrationService
from depthcloud.camera.service import CameraService
from depthcloud.config import BACKEND_ROOT, Settings
from depthcloud.logging_config import configure_logging
from depthcloud.models.manager import ModelManager
from depthcloud.streaming import stream_camera
from depthcloud.workflow.capture_scan import CaptureScanner
from depthcloud.workflow.live import LiveScanner
from depthcloud.workflow.service import ReconstructionService


def create_app(settings: Settings | None = None, camera_service: CameraService | None = None):
    settings = settings or Settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(app.state.models.initialize_device)
        yield
        await asyncio.to_thread(app.state.camera.stop)
        await asyncio.to_thread(app.state.reconstruction.close)

    app = FastAPI(title="DepthCloud", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.camera = camera_service or CameraService(settings)
    app.state.models = ModelManager(settings)
    app.state.calibration = CalibrationService()
    app.state.reconstruction = ReconstructionService(app.state.models, app.state.calibration, settings.max_saved_views, settings.session_memory_mb * 1024**2)
    app.state.live_scan = LiveScanner(app.state.reconstruction, app.state.camera)
    app.state.capture_scan = CaptureScanner(app.state.reconstruction, app.state.camera, BACKEND_ROOT / ".cache" / "capture-session")
    app.state.started_at = time.monotonic()
    app.state.stream_clients = 0
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"], allow_headers=["content-type"],
    )

    @app.middleware("http")
    async def reject_untrusted_mutations(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and origin and origin not in settings.allowed_origins:
            return JSONResponse({"detail": "This browser origin is not allowed to control the camera."}, status_code=403)
        return await call_next(request)

    app.include_router(camera.router)
    app.include_router(system.router)
    app.include_router(reconstruction.router)
    app.add_api_websocket_route("/ws/camera", stream_camera)
    frontend = BACKEND_ROOT.parent / "app" / "dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()

