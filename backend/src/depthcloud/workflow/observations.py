import threading
import uuid
from dataclasses import dataclass, field

import numpy as np

from depthcloud.workflow.schemas import CaptureView


@dataclass
class Observation:
    id: str
    label: str
    sequence_id: int
    timestamp: float
    bgr: np.ndarray
    roi: dict
    roi_pixels: tuple
    camera: dict
    selected: bool = True
    status: str = "captured"
    error: str | None = None
    result: dict = field(default_factory=dict)

    def metadata(self):
        return {"id": self.id, "label": self.label, "sequence_id": self.sequence_id,
                "timestamp": self.timestamp, "width": self.bgr.shape[1], "height": self.bgr.shape[0],
                "roi": self.roi, "roi_pixels": self.roi_pixels, "camera": self.camera,
                "selected": self.selected, "status": self.status, "error": self.error,
                "result": {k: v for k, v in self.result.items() if not isinstance(v, np.ndarray)}}

    def bytes_used(self):
        return self.bgr.nbytes + sum(value.nbytes for value in self.result.values() if isinstance(value, np.ndarray))


class ObservationStore:
    """One bounded in-memory session. Capture snapshots outlive camera stop."""
    def __init__(self, max_views=24, max_bytes=512 * 1024**2):
        self.max_views, self.max_bytes = max_views, max_bytes
        self.lock = threading.RLock()
        self.session_id = uuid.uuid4().hex
        self.views = {}
        self.snapshot = None

    def bytes_used(self):
        with self.lock:
            return sum(view.bytes_used() for view in self.views.values()) + (self.snapshot["bgr"].nbytes if self.snapshot else 0)

    def freeze(self, packet, camera):
        with self.lock:
            previous = self.snapshot["bgr"].nbytes if self.snapshot else 0
            if self.bytes_used() - previous + packet.nbytes > self.max_bytes:
                raise ValueError("Session memory limit reached. Delete observations or reset reconstruction.")
            bgr = packet.bgr.copy()
            bgr.setflags(write=False)
            self.snapshot = {"id": uuid.uuid4().hex, "sequence_id": packet.sequence_id, "timestamp": packet.timestamp,
                             "bgr": bgr, "camera": camera}
            return {k: v for k, v in self.snapshot.items() if k != "bgr"} | {"width": bgr.shape[1], "height": bgr.shape[0]}

    def capture(self, payload: CaptureView):
        with self.lock:
            snap = self.snapshot
            if snap is None or payload.snapshot_id != snap["id"]:
                raise ValueError("Frozen frame expired. Freeze the current frame again.")
            previous = self.get(payload.replace_id) if payload.replace_id else None
            if previous is None and len(self.views) >= self.max_views:
                raise ValueError(f"Session view limit is {self.max_views}. Delete or replace a view.")
            if self.bytes_used() + snap["bgr"].nbytes - (previous.bytes_used() if previous else 0) > self.max_bytes:
                raise ValueError("Session memory limit reached.")
            h, w = snap["bgr"].shape[:2]
            pixels = payload.roi.pixels(w, h)
            view = Observation(id=previous.id if previous else uuid.uuid4().hex,
                               label=payload.label.strip() or (previous.label if previous else f"Reference {len(self.views) + 1}"),
                               sequence_id=snap["sequence_id"], timestamp=snap["timestamp"],
                               bgr=snap["bgr"], roi=payload.roi.model_dump(), roi_pixels=pixels, camera=snap["camera"])
            self.views[view.id] = view
            return view.metadata()

    def get(self, view_id):
        with self.lock:
            if view_id not in self.views:
                raise ValueError("Observation does not exist.")
            return self.views[view_id]

    def list(self):
        with self.lock:
            return [view.metadata() for view in self.views.values()]

    def delete(self, view_id):
        with self.lock:
            self.get(view_id)
            del self.views[view_id]

    def reset(self):
        with self.lock:
            self.views.clear()
            self.snapshot = None
            self.session_id = uuid.uuid4().hex

