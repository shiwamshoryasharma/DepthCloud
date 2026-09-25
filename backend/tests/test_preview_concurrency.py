import threading
import time

import numpy as np

from depthcloud.camera.buffer import FramePacket
from depthcloud.camera.service import CameraService
from depthcloud.config import Settings
from depthcloud.workflow.schemas import PreviewOptions


def test_changing_representation_during_encoding_keeps_preview_worker_alive(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def transform(image, options):
        calls.append(options.mode)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return image
    monkeypatch.setattr('depthcloud.camera.service.transform_preview', transform)
    service = CameraService(Settings())
    service.buffer.push(FramePacket(1, time.time(), time.monotonic(), np.zeros((60,80,3), np.uint8)))
    service._received = 1
    worker = threading.Thread(target=service._encode_preview, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        service.configure_preview(PreviewOptions(mode='grayscale'))
        release.set()
        deadline = time.monotonic() + 2
        while service.latest_preview() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert service.latest_preview() is not None
        assert calls[-1] == 'grayscale'
        assert worker.is_alive()
    finally:
        service._stop.set()
        release.set()
        worker.join(3)
