"""Deterministic synthetic inputs strictly for lifecycle regression tests."""
import time

import numpy as np


def streaming_worker(config, output, stop):
    seq = 0
    while not stop.wait(0.02):
        seq += 1
        output.put({
            "type": "frame", "sequence": seq, "timestamp": time.time(),
            "monotonic": time.monotonic(), "bgr": np.full((48, 64, 3), seq % 255, dtype=np.uint8),
            "metadata": {"backend": "TEST ONLY"},
        })
    output.cancel_join_thread()


def disconnected_worker(config, output, stop):
    output.put({"type": "frame", "sequence": 1, "timestamp": time.time(), "monotonic": time.monotonic(), "bgr": np.zeros((48, 64, 3), dtype=np.uint8)})
    time.sleep(0.1)
    output.put({"type": "error", "message": "Test camera disconnected"})
    time.sleep(0.2)


def blocked_worker(config, output, stop):
    time.sleep(30)

