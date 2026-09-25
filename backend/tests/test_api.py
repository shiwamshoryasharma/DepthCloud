from fastapi.testclient import TestClient

from depthcloud.config import Settings
from depthcloud.main import create_app


def test_health_and_idle_camera_do_not_open_hardware():
    with TestClient(create_app(Settings())) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        state = client.get("/api/camera/status").json()
        assert state["state"] == "idle"
        assert state["buffer_fill"] == 0
        assert state["capture_fps"] == 0


def test_invalid_camera_request_is_rejected():
    with TestClient(create_app(Settings())) as client:
        response = client.post("/api/camera/start", json={"index": -1})
        assert response.status_code == 422
        assert client.get("/api/camera/status").json()["state"] == "idle"


def test_buffer_configuration_is_validated_and_persists():
    with TestClient(create_app(Settings())) as client:
        assert client.patch("/api/camera/buffer", json={"capacity": 0}).status_code == 422
        response = client.patch("/api/camera/buffer", json={"capacity": 7, "automatic": False})
        assert response.status_code == 200
        assert response.json()["current_buffer_size"] == 7
        assert response.json()["automatic_buffer"] is False


def test_stop_is_idempotent_and_flushed():
    with TestClient(create_app(Settings())) as client:
        for _ in range(2):
            state = client.post("/api/camera/stop").json()
            assert state["state"] == "idle"
            assert state["buffer_fill"] == 0


def test_untrusted_browser_cannot_start_camera():
    with TestClient(create_app(Settings())) as client:
        response = client.post("/api/camera/start", json={"index": 0}, headers={"origin": "https://untrusted.example"})
        assert response.status_code == 403


def test_websocket_reports_real_idle_state():
    with TestClient(create_app(Settings())) as client:
        with client.websocket_connect("/ws/camera", headers={"origin": "http://localhost:5173"}) as ws:
            status = ws.receive_json()
            assert status["type"] == "status"
            assert status["camera"]["state"] == "idle"
            assert status["camera"]["buffer_fill"] == 0



def test_discovery_honors_selected_backend(monkeypatch):
    calls = []

    def probe(max_index, timeout, backend="auto"):
        calls.append(backend)
        return {"devices": [], "timed_out_indices": [], "scanned_through": max_index, "probe_errors": []}

    monkeypatch.setattr("depthcloud.camera.service.discover_cameras", probe)
    with TestClient(create_app(Settings())) as client:
        assert client.post("/api/camera/discover?backend=msmf").status_code == 200
    assert calls == ["msmf"]
