import io
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from depthcloud.workflow.schemas import (
    CaptureOptions,
    CaptureView,
    Checkerboard,
    Intrinsics,
    LiveOptions,
    PipelineOptions,
    PreviewOptions,
    ViewUpdate,
)
from depthcloud.workflow.service import WorkflowBusy

router = APIRouter(prefix="/api", tags=["reconstruction"])


def invoke(function, *args):
    try:
        return function(*args)
    except WorkflowBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def active_frame(request):
    service = request.app.state.camera
    with service._lock:
        packet = service.buffer.latest()
        if service.state != "running" or packet is None:
            raise ValueError("Start the camera and wait for a live frame first.")
        return packet, {"index": service.request.index, **service.metadata, "generation": service.generation}


def png(image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(500, "Could not encode observation image.")
    return Response(data.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@router.patch("/camera/preview")
def preview(payload: PreviewOptions, request: Request):
    return request.app.state.camera.configure_preview(payload)


@router.get("/reconstruction/state")
def state(request: Request):
    return request.app.state.reconstruction.status()


@router.post("/reconstruction/freeze")
def freeze(request: Request):
    packet, metadata = invoke(active_frame, request)
    return invoke(request.app.state.reconstruction.freeze, packet, metadata)


@router.get("/reconstruction/snapshot/{snapshot_id}")
def snapshot(snapshot_id: str, request: Request):
    service = request.app.state.reconstruction
    with service.lock:
        frame = service.store.snapshot
        if frame is None or frame["id"] != snapshot_id:
            raise HTTPException(404, "Frozen frame expired.")
        image = frame["bgr"]
    return png(image)


@router.post("/reconstruction/views")
def capture(payload: CaptureView, request: Request):
    return invoke(request.app.state.reconstruction.capture, payload)


@router.patch("/reconstruction/views/{view_id}")
def select(view_id: str, payload: ViewUpdate, request: Request):
    return invoke(request.app.state.reconstruction.select, view_id, payload.selected)


@router.delete("/reconstruction/views/{view_id}")
def delete(view_id: str, request: Request):
    invoke(request.app.state.reconstruction.delete, view_id)
    return {"deleted": view_id}


@router.get("/reconstruction/views/{view_id}/image/{kind}")
def image(view_id: str, kind: Literal["original", "crop", "depth", "mask", "object"], request: Request,
          boxes: bool = False, mask_overlay: bool = False):
    from depthcloud.vision.masking import depth_preview
    service = request.app.state.reconstruction
    with service.lock:
        view = invoke(service.store.get, view_id)
        bgr, result, roi = view.bgr, dict(view.result), view.roi_pixels
    if kind in ("depth", "mask", "object") and "mask" not in result:
        raise HTTPException(422, "Process this observation before inspecting its depth or mask.")
    if kind == "depth" and "depth" not in result:
        raise HTTPException(422, "Detection reference has no depth; only incoming live frames are reconstructed.")
    if kind == "crop":
        x0, y0, x1, y1 = roi
        output = bgr[y0:y1, x0:x1]
    elif kind == "depth":
        output = depth_preview(result["depth"], result["mask"] if mask_overlay else None)
    elif kind == "mask":
        output = result["mask"]
    elif kind == "object":
        output = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        output[:, :, 3] = result["mask"]
    else:
        output = bgr.copy()
        if mask_overlay and "mask" in result:
            active = result["mask"] > 0
            output[active] = (0.65 * output[active] + 0.35 * np.array([100, 220, 160])).astype(np.uint8)
        if boxes:
            x0, y0, x1, y1 = roi
            cv2.rectangle(output, (x0, y0), (x1, y1), (90, 220, 190), 2)
            for detection in result.get("detections", []):
                a, b, c, d = [round(v) for v in detection["box"]]
                cv2.rectangle(output, (a, b), (c, d), (60, 160, 255), 2)
                cv2.putText(output, f'{detection["label"]} {detection["score"]:.2f}', (a, max(15, b - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 160, 255), 1, cv2.LINE_AA)
    return png(output)


@router.get("/reconstruction/views/{view_id}/depth.npy")
def raw_depth(view_id: str, request: Request):
    view = invoke(request.app.state.reconstruction.store.get, view_id)
    depth = view.result.get("depth")
    if depth is None:
        raise HTTPException(422, "No raw depth is available.")
    buffer = io.BytesIO()
    np.save(buffer, depth, allow_pickle=False)
    return Response(buffer.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="depth-{view.id}.npy"'})


@router.post("/reconstruction/start", status_code=202)
def start(payload: PipelineOptions, request: Request):
    return invoke(request.app.state.reconstruction.start, payload)


@router.post("/reconstruction/cancel")
def cancel(request: Request):
    return request.app.state.reconstruction.cancel()


@router.post("/reconstruction/reset")
def reset(request: Request):
    return invoke(request.app.state.reconstruction.reset)


@router.get("/reconstruction/geometry/{kind}")
def geometry(kind: Literal["cloud", "mesh"], revision: int, request: Request):
    data = invoke(request.app.state.reconstruction.geometry, kind, revision)
    return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@router.get("/reconstruction/export/{kind}")
def export(kind: Literal["ply", "obj"], request: Request):
    data = invoke(request.app.state.reconstruction.export, kind)
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="depthcloud.{kind}"'})


@router.get("/calibration")
def calibration(request: Request):
    return request.app.state.calibration.status()


@router.put("/calibration")
def set_calibration(payload: Intrinsics, request: Request):
    service = request.app.state.reconstruction
    with service.lock:
        invoke(service._require_idle)
        return request.app.state.calibration.set(payload.model_copy(update={"source": "manual", "rms": None}))


@router.delete("/calibration")
def clear_calibration(request: Request):
    service = request.app.state.reconstruction
    with service.lock:
        invoke(service._require_idle)
        return request.app.state.calibration.clear()


@router.post("/calibration/capture")
def calibration_capture(payload: Checkerboard, request: Request):
    service = request.app.state.reconstruction
    with service.lock:
        invoke(service._require_idle)
        packet, _ = invoke(active_frame, request)
        return invoke(request.app.state.calibration.capture, packet.bgr, payload)


@router.post("/calibration/solve")
def calibration_solve(request: Request):
    service = request.app.state.reconstruction
    with service.lock:
        invoke(service._require_idle)
        return invoke(request.app.state.calibration.solve)


@router.get("/integration/isaac")
def isaac(request: Request):
    from depthcloud.integrations.isaac import descriptor
    return descriptor(request.app.state.reconstruction.status())



@router.post("/reconstruction/live/start", status_code=202)
def start_live(payload: LiveOptions, request: Request):
    return invoke(request.app.state.live_scan.start, payload)


@router.post("/reconstruction/live/stop")
def stop_live(request: Request):
    return request.app.state.live_scan.stop()


@router.get("/reconstruction/live/image")
def live_image(request: Request):
    scanner = request.app.state.live_scan
    with scanner.service.lock:
        data = scanner.preview
    if data is None:
        raise HTTPException(404,"No processed live frame yet.")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control":"no-store"})


@router.post('/reconstruction/capture/start', status_code=202)
def start_capture(payload: CaptureOptions, request: Request):
    return invoke(request.app.state.capture_scan.start, payload)


@router.post('/reconstruction/capture/{action}')
def capture_action(action: Literal['pause', 'resume', 'build', 'discard'], request: Request):
    return invoke(getattr(request.app.state.capture_scan, action))


@router.get('/reconstruction/capture/image/{index}')
def captured_image(index: int, request: Request, kind: Literal['original', 'mask', 'filtered'] = 'original'):
    scanner = request.app.state.capture_scan
    with scanner.service.lock:
        return png(invoke(scanner.dataset.image, index, kind))


@router.get('/reconstruction/capture/report')
def capture_report(request: Request):
    scanner = request.app.state.capture_scan
    with scanner.service.lock:
        if not scanner.dataset.data or 'report' not in scanner.dataset.data:
            raise HTTPException(404, 'No reconstruction report is available yet.')
        return scanner.dataset.data['report']
