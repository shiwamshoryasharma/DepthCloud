import asyncio
import contextlib
import time

from fastapi import WebSocket, WebSocketDisconnect

from depthcloud.api.system import system_status


async def stream_camera(websocket: WebSocket):
    app = websocket.app
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in app.state.settings.allowed_origins:
        await websocket.close(code=1008)
        return
    if app.state.stream_clients >= app.state.settings.max_stream_clients:
        await websocket.close(code=1013)
        return
    await websocket.accept()
    app.state.stream_clients += 1
    acknowledged = asyncio.Event()
    acknowledged.set()

    async def receive_acknowledgements():
        while True:
            message = await websocket.receive_text()
            if message == "ack":
                acknowledged.set()

    async def send_updates():
        seen = None
        next_status = 0.0
        sent_at = 0.0
        while True:
            now = time.monotonic()
            if now >= next_status:
                await asyncio.wait_for(websocket.send_json({
                    "type": "status",
                    "camera": app.state.camera.status(),
                    "system": system_status(app),
                    "reconstruction": app.state.reconstruction.status(),
                }), timeout=2)
                next_status = now + 0.5
            frame = app.state.camera.latest_preview()
            if frame is not None and frame[0] != seen and acknowledged.is_set():
                acknowledged.clear()
                await asyncio.wait_for(websocket.send_bytes(frame[1]), timeout=2)
                seen = frame[0]
                sent_at = now
            if not acknowledged.is_set() and now - sent_at > 10:
                await websocket.close(code=1013, reason="Preview acknowledgement timeout")
                return
            await asyncio.sleep(1 / app.state.settings.preview_fps)

    tasks = [asyncio.create_task(receive_acknowledgements()), asyncio.create_task(send_updates())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, asyncio.TimeoutError):
                await task
        app.state.stream_clients -= 1

