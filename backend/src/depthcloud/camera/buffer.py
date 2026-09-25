import math
from collections import deque
from dataclasses import dataclass
from threading import RLock

import numpy as np


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence_id: int
    timestamp: float
    captured_monotonic: float
    bgr: np.ndarray

    @property
    def nbytes(self):
        return self.bgr.nbytes


class FrameBuffer:
    """One owned BGR array per frame, bounded by both count and actual bytes."""

    def __init__(self, capacity: int, max_bytes: int):
        if capacity < 1 or max_bytes < 1:
            raise ValueError("Buffer capacity and memory budget must be positive.")
        self.capacity = capacity
        self.max_bytes = max_bytes
        self._frames = deque()
        self._bytes = 0
        self._evicted = 0
        self._lock = RLock()

    def _remove_oldest(self):
        self._bytes -= self._frames.popleft().nbytes
        self._evicted += 1

    def push(self, frame: FramePacket):
        if frame.nbytes > self.max_bytes:
            raise ValueError("One camera frame exceeds the buffer memory budget.")
        with self._lock:
            while self._frames and (
                len(self._frames) >= self.capacity or self._bytes + frame.nbytes > self.max_bytes
            ):
                self._remove_oldest()
            self._frames.append(frame)
            self._bytes += frame.nbytes

    def resize(self, capacity: int):
        if capacity < 1:
            raise ValueError("Buffer capacity must be positive.")
        with self._lock:
            self.capacity = capacity
            while len(self._frames) > capacity:
                self._remove_oldest()

    def latest(self):
        with self._lock:
            return self._frames[-1] if self._frames else None

    def snapshot(self):
        with self._lock:
            return list(self._frames)

    def clear(self):
        with self._lock:
            self._frames.clear()
            self._bytes = 0
            self._evicted = 0

    def stats(self):
        with self._lock:
            return {
                "fill": len(self._frames), "capacity": self.capacity,
                "bytes": self._bytes, "evicted": self._evicted,
                "sequence_ids": [f.sequence_id for f in self._frames][-16:],
            }


def recommend_capacity(fps, latency_ms, frame_bytes, memory_bytes, maximum):
    """Retain two measured processing intervals, subject to hard RAM/count caps."""
    memory_cap = max(1, memory_bytes // max(1, frame_bytes))
    target = max(2, math.ceil(max(1, fps) * max(0.001, latency_ms / 1000) * 2))
    return int(max(1, min(maximum, memory_cap, target)))

