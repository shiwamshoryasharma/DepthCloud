import time

import pytest

from depthcloud.camera.service import CameraBusy, CameraService
from depthcloud.config import Settings
from depthcloud.schemas import CameraStart
from tests.workers import blocked_worker, disconnected_worker, streaming_worker


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition did not become true before timeout")


def test_continuous_capture_stop_flush_and_restart():
    service = CameraService(Settings(worker_join_timeout=0.2), worker_target=streaming_worker)
    try:
        service.start(CameraStart(width=640, height=480))
        wait_until(lambda: service.status()["frames_received"] >= 5 and service.latest_preview() is not None)
        first_generation = service.status()["generation"]
        before = service.status()["frames_received"]
        wait_until(lambda: service.status()["frames_received"] > before + 3)
        assert service.status()["buffer_fill"] <= service.status()["current_buffer_size"]
        assert service.status()["capture_fps"] > 0
        with pytest.raises(CameraBusy):
            service.start(CameraStart())
        service.stop()
        assert service.status()["state"] == "idle"
        assert service.status()["buffer_fill"] == 0
        assert service.status()["latest_sequence"] is None
        assert service.latest_preview() is None
        service.start(CameraStart())
        wait_until(lambda: service.status()["frames_received"] > 2)
        assert service.status()["generation"] != first_generation
    finally:
        service.stop()


def test_disconnect_clears_buffer_and_preview_without_hiding_error():
    service = CameraService(Settings(worker_join_timeout=0.2), worker_target=disconnected_worker)
    try:
        service.start(CameraStart())
        wait_until(lambda: service.status()["state"] == "error")
        assert "disconnected" in service.status()["error"]
        assert service.status()["buffer_fill"] == 0
        assert service.latest_preview() is None
    finally:
        service.stop()


def test_stop_terminates_blocked_driver_process_within_bound():
    service = CameraService(Settings(worker_join_timeout=0.1), worker_target=blocked_worker)
    service.start(CameraStart())
    began = time.monotonic()
    service.stop()
    assert time.monotonic() - began < 4
    assert service.status()["state"] == "idle"
    assert service._process is None

