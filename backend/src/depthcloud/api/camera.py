import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request

from depthcloud.camera.service import CameraBusy
from depthcloud.schemas import BufferSettings, CameraStart

router = APIRouter(prefix="/api/camera", tags=["camera"])


@router.get("/status")
def status(request: Request):
    return request.app.state.camera.status()


@router.post("/discover")
async def discover(request: Request, backend: Literal["auto", "dshow", "msmf", "any"] = "auto"):
    try:
        return await asyncio.to_thread(request.app.state.camera.discover, backend)
    except CameraBusy as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/start", status_code=202)
async def start(payload: CameraStart, request: Request):
    try:
        return await asyncio.to_thread(request.app.state.camera.start, payload)
    except CameraBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/stop")
async def stop(request: Request):
    return await asyncio.to_thread(request.app.state.camera.stop)


@router.patch("/buffer")
def configure_buffer(payload: BufferSettings, request: Request):
    return request.app.state.camera.configure_buffer(payload)


