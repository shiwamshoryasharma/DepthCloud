import numpy as np
import pytest

from depthcloud.camera.buffer import FramePacket
from depthcloud.workflow.observations import ObservationStore
from depthcloud.workflow.schemas import ROI, CaptureView


def packet(sequence=1):
    return FramePacket(sequence, 123.5, 10, np.zeros((60, 80, 3), dtype=np.uint8))


def test_capture_uses_frozen_original_and_native_roi_not_later_camera_frame():
    store = ObservationStore(max_views=2, max_bytes=100_000)
    original = packet()
    frozen = store.freeze(original, {"index": 0})
    original.bgr[:] = 255
    view = store.capture(CaptureView(snapshot_id=frozen["id"], roi=ROI(x=.25, y=.25, width=.5, height=.5), label="front"))
    observation = store.get(view["id"])
    assert observation.sequence_id == 1 and observation.roi_pixels == (20, 15, 60, 45)
    assert not observation.bgr.any()
    assert observation.bgr.flags.writeable is False


def test_stale_freeze_id_cannot_capture_a_different_frame():
    store = ObservationStore(max_views=2, max_bytes=100_000)
    old = store.freeze(packet(), {})
    store.freeze(packet(2), {})
    with pytest.raises(ValueError, match="expired"):
        store.capture(CaptureView(snapshot_id=old["id"], roi=ROI(x=0, y=0, width=1, height=1)))


def test_view_count_budget_recapture_and_reset():
    store = ObservationStore(max_views=1, max_bytes=100_000)
    frozen = store.freeze(packet(), {})
    payload = CaptureView(snapshot_id=frozen["id"], roi=ROI(x=0, y=0, width=1, height=1))
    view = store.capture(payload)
    with pytest.raises(ValueError, match="limit"):
        store.capture(payload)
    replacement = store.capture(payload.model_copy(update={"replace_id": view["id"]}))
    assert replacement["id"] == view["id"] and len(store.list()) == 1
    store.reset()
    assert store.list() == [] and store.bytes_used() == 0


def test_freeze_rejects_an_image_exceeding_memory_budget():
    store = ObservationStore(max_views=2, max_bytes=100)
    with pytest.raises(ValueError, match="memory"):
        store.freeze(packet(), {})

