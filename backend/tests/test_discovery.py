from queue import Queue

import numpy as np

from depthcloud.camera import worker


def test_discovery_retries_transient_initial_read_failure(monkeypatch):
    class WarmingCamera:
        def __init__(self):
            self.reads = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.reads += 1
            return (False, None) if self.reads == 1 else (True, np.zeros((4, 4, 3), dtype=np.uint8))

        def getBackendName(self):
            return "TEST"

        def set(self, *args):
            return True

        def release(self):
            self.released = True

    camera = WarmingCamera()
    monkeypatch.setattr(worker.cv2, "VideoCapture", lambda *args: camera)
    output = Queue()
    worker.probe_main(0, "auto", output)
    assert not output.empty(), "A transient first read must not hide an available camera"
    assert output.get()["index"] == 0
    assert camera.released
