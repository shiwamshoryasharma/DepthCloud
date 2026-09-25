import time

from fastapi.testclient import TestClient

from depthcloud.camera.service import CameraService
from depthcloud.config import Settings
from depthcloud.main import create_app
from depthcloud.schemas import CameraStart
from tests.workers import streaming_worker


def test_websocket_sends_one_image_until_client_acknowledges_and_closes_cleanly():
    settings = Settings(worker_join_timeout=0.2)
    service = CameraService(settings, worker_target=streaming_worker)
    with TestClient(create_app(settings, camera_service=service)) as client:
        service.start(CameraStart())
        deadline = time.monotonic() + 8
        while service.latest_preview() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert service.latest_preview() is not None
        with client.websocket_connect("/ws/camera") as ws:
            assert ws.receive_json()["type"] == "status"
            image = ws.receive_bytes()
            assert image[:2] == b"\xff\xd8"
            # With no ACK, the next message is telemetry, never another queued image.
            assert ws.receive_json()["type"] == "status"
            ws.send_text("ack")
            message = ws.receive()
            if "text" in message:
                message = ws.receive()
            assert message["bytes"][:2] == b"\xff\xd8"
        deadline = time.monotonic() + 2
        while client.get("/api/system").json()["stream_clients"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.get("/api/system").json()["stream_clients"] == 0
        service.stop()
        assert client.get("/api/camera/status").json()["buffer_fill"] == 0

