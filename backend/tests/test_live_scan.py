import threading
import time

import numpy as np
import pytest

from depthcloud.camera.buffer import FramePacket
from depthcloud.camera.calibration import CalibrationService
from depthcloud.reconstruction.live_geometry import SurfaceVolume, validate_incremental_pose
from depthcloud.vision.reference import ReferenceAppearance
from depthcloud.workflow.live import LiveScanner
from depthcloud.workflow.schemas import ROI, CaptureView, Intrinsics, LiveOptions, PipelineOptions
from depthcloud.workflow.service import ReconstructionService


def test_reference_color_mask_rejects_adjacent_background_and_keeps_object():
    reference = np.full((120, 120, 3), 210, np.uint8)
    reference[20:100, 45:75] = [200, 55, 20]
    mask = np.zeros((120, 120), np.uint8)
    mask[20:100, 45:75] = 255
    appearance = ReferenceAppearance(reference, mask)
    current = reference.copy()
    current[40:100, 75:100] = [60, 110, 180]
    result = appearance.segment(current, (20, 10, 110, 110))
    assert result[55, 60] == 255
    assert result[60, 90] == 0 and result[:10].sum() == 0


def test_incremental_pose_rejects_large_jumps_and_nonrigid_transforms():
    points = np.array([[-0.05, -0.1, 0.5], [0.05, 0.1, 0.5], [0, 0, 0.6]])
    bad = np.eye(4)
    bad[0, 3] = 0.4
    with pytest.raises(ValueError, match="motion"):
        validate_incremental_pose(bad, points, points)
    scale = np.eye(4)
    scale[0, 0] = 1.2
    with pytest.raises(ValueError, match="rigid"):
        validate_incremental_pose(scale, points, points)
    assert validate_incremental_pose(np.eye(4), points, points)["rotation_deg"] == 0


def test_tsdf_builds_surface_from_masked_metric_depth_and_bounds_allocations():
    options = LiveOptions(scan_mode="rotating_object", reference_ids=["ref"], volume_resolution=64)
    points = np.array([[-0.1, -0.1, 0.5], [0.1, 0.1, 0.5], [0, 0, 0.52]])
    volume = SurfaceVolume(points, options)
    image = np.full((80, 80, 3), 120, np.uint8)
    depth = np.full((80, 80), 0.5, np.float32)
    mask = np.zeros((80, 80), np.uint8)
    mask[15:65, 15:65] = 255
    intr = Intrinsics(fx=120, fy=120, cx=39.5, cy=39.5, width=80, height=80)
    volume.integrate(image, depth, mask, intr, np.eye(4))
    cloud, mesh = volume.extract()
    assert len(mesh.triangles) > 100 and len(cloud.points) > 100
    vertices = np.asarray(mesh.vertices)
    assert np.max(np.abs(vertices[:, 2] - 0.5)) < volume.voxel_size * 2
    assert volume.resolution == 64
    assert depth[0, 0] == 0.5


class TestModels:
    __test__ = False

    def __init__(self):
        self.depth_inputs = []

    def describe_objects(self, crops):
        # Deterministic unit-test seam; real local model covered by saved-frame replay.
        return np.ones((len(crops), 1), np.float32)

    def detect(self, image, threshold):
        return []

    def infer_depth(self, image, focal_px=None):
        self.depth_inputs.append(int(image[0, 0, 0]))
        return np.full(image.shape[:2], 0.5, np.float32), 120.0


class TestCamera:
    __test__ = False

    def __init__(self):
        self._lock = threading.RLock()
        self.state = "running"
        self.generation = "generation1"
        self.sequence = 10
        self.image = np.full((80, 80, 3), 210, np.uint8)
        self.image[15:65, 25:55] = [200, 55, 20]

    def latest_for_scan(self):
        with self._lock:
            self.sequence += 1
            return FramePacket(
                self.sequence, time.time(), time.monotonic(), self.image.copy()
            ), self.generation


def test_live_reference_is_detection_only_stop_preserves_incremental_result(monkeypatch):
    models = TestModels()
    service = ReconstructionService(models, CalibrationService())
    camera = TestCamera()
    reference = camera.image.copy()
    reference[0, 0] = [99, 99, 99]
    snap = service.freeze(FramePacket(1, 1, 1, reference), {})
    view = service.capture(CaptureView(snapshot_id=snap["id"], roi=ROI(x=0.15, y=0.1, width=0.7, height=0.8)))
    other = camera.image.copy()
    other[0, 0] = [98, 98, 98]
    other[15:65, 25:55] = [20, 65, 200]
    snap2 = service.freeze(FramePacket(2, 2, 2, other), {})
    view2 = service.capture(CaptureView(snapshot_id=snap2["id"], roi=ROI(x=.15, y=.1, width=.7, height=.8)))
    scanner = LiveScanner(service, camera)
    scanner.start(
        LiveOptions(scan_mode="rotating_object", 
            reference_ids=[view["id"], view2["id"]],
            volume_resolution=64,
            interval_seconds=0.1,
            pipeline=PipelineOptions(voxel_size=0.003, stride=1),
        )
    )
    deadline = time.monotonic() + 15
    while service.result is None and service.busy and time.monotonic() < deadline:
        time.sleep(0.02)
    scanner.stop()
    service.thread.join(8)
    assert not service.busy
    assert service.result is not None, service.status()
    assert service.result["triangles"] > 0
    assert 99 not in models.depth_inputs and 98 not in models.depth_inputs  # reference was never used for depth/fusion
    assert service.live_status["integrated"] >= 1
    assert len(service.store.views) == 2  # no endless saved-frame list
    assert service.live_status["prepared_references"] == 2
    assert service.live_status["matched_reference"]["reference_id"] == view["id"]
    assert all(v.result.get("reference_only") and "depth" not in v.result for v in service.store.views.values())
    assert service.state == "completed"


def test_uncertain_live_pose_never_updates_or_deforms_accepted_surface(monkeypatch):
    models = TestModels()
    service = ReconstructionService(models, CalibrationService())
    camera = TestCamera()
    snap = service.freeze(FramePacket(1, 1, 1, camera.image.copy()), {})
    view = service.capture(CaptureView(snapshot_id=snap["id"], roi=ROI(x=0.15, y=0.1, width=0.7, height=0.8)))
    monkeypatch.setattr("depthcloud.workflow.live.depth_feature_pose", lambda *args: (None, 0))
    scanner = LiveScanner(service, camera)
    scanner.start(
        LiveOptions(scan_mode="rotating_object", 
            reference_ids=[view["id"]],
            volume_resolution=64,
            interval_seconds=0.1,
            pipeline=PipelineOptions(voxel_size=0.003, stride=1),
        )
    )
    deadline = time.monotonic() + 10
    while service.busy and service.live_status["rejected"] < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    scanner.stop()
    service.thread.join(5)
    assert service.live_status["rejected"] >= 2
    assert service.live_status["integrated"] == 1
    assert service.live_status["surface_updates"] == 1
    assert "Pose uncertain" in service.live_status["last_error"]
    assert service.result["triangles"] > 0


def test_camera_stop_ends_live_generation_without_erasing_mesh():
    service = ReconstructionService(TestModels(), CalibrationService())

    class DisconnectingCamera(TestCamera):
        def latest_for_scan(self):
            if self.sequence >= 12:
                raise ValueError("Camera stopped")
            return super().latest_for_scan()

    camera = DisconnectingCamera()
    snap = service.freeze(FramePacket(1, 1, 1, camera.image.copy()), {})
    view = service.capture(CaptureView(snapshot_id=snap["id"], roi=ROI(x=0.15, y=0.1, width=0.7, height=0.8)))
    scanner = LiveScanner(service, camera)
    scanner.start(LiveOptions(scan_mode="rotating_object", reference_ids=[view["id"]], volume_resolution=64, interval_seconds=0.1))
    service.thread.join(10)
    assert not service.busy
    assert service.result["triangles"] > 0
    assert service.error == "Camera stopped"


def test_direction_labels_are_reference_relative_and_cover_top_bottom():
    from depthcloud.reconstruction.live_geometry import view_direction

    assert view_direction(np.eye(4)) == "front"
    top = np.eye(4)
    top[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    assert view_direction(top) == "top"
    assert view_direction(np.linalg.inv(top)) == "bottom"


def test_missing_object_refreshes_preview_without_changing_last_surface():
    import cv2

    service = ReconstructionService(TestModels(), CalibrationService())

    class MissingObjectCamera(TestCamera):
        def latest_for_scan(self):
            if self.sequence >= 12:
                self.image[:] = 180
            return super().latest_for_scan()

    camera = MissingObjectCamera()
    snap = service.freeze(FramePacket(1, 1, 1, camera.image.copy()), {})
    view = service.capture(CaptureView(snapshot_id=snap['id'], roi=ROI(x=.15, y=.1, width=.7, height=.8)))
    scanner = LiveScanner(service, camera)
    scanner.start(LiveOptions(scan_mode="rotating_object", reference_ids=[view['id']], volume_resolution=64, interval_seconds=.1))
    deadline = time.monotonic() + 10
    while service.busy and service.live_status['rejected'] < 2 and time.monotonic() < deadline:
        time.sleep(.02)
    scanner.stop()
    service.thread.join(5)
    assert service.live_status['rejected'] >= 2
    assert service.live_status['integrated'] == 1
    assert service.live_status['matched_reference'] is None
    assert service.live_status['preview_sequence'] == service.live_status['last_sequence']
    preview = cv2.imdecode(np.frombuffer(scanner.preview, np.uint8), cv2.IMREAD_COLOR)
    assert abs(float(preview.mean()) - 36) < 2  # current empty frame, not old object mask
    assert service.result['triangles'] > 0


def test_moving_camera_mode_fuses_using_scene_pose_without_object_icp(monkeypatch):
    import cv2

    rng = np.random.default_rng(20)
    background = rng.integers(110, 230, (480, 640), dtype=np.uint8)
    base = np.repeat(background[:, :, None], 3, axis=2)
    base[160:320, 285:355] = [200, 55, 20]

    class SceneCamera(TestCamera):
        def latest_for_scan(self):
            self.sequence += 1
            shift = -min(24, max(0, self.sequence - 12) * 4)
            image = cv2.warpAffine(base, np.float32([[1, 0, shift], [0, 1, 0]]), (640, 480))
            return FramePacket(self.sequence, time.time(), time.monotonic(), image), self.generation

    def forbidden(*args):
        raise AssertionError('Object-only ICP must not run in moving-camera mode')

    monkeypatch.setattr('depthcloud.workflow.live.register_clouds', forbidden)
    models = TestModels()
    service = ReconstructionService(models, CalibrationService())
    service.calibration.set(Intrinsics(fx=400, fy=400, cx=319.5, cy=239.5, width=640, height=480))
    snap = service.freeze(FramePacket(1, 1, 1, base), {})
    view = service.capture(CaptureView(snapshot_id=snap['id'], roi=ROI(x=.4, y=.3, width=.2, height=.4)))
    scanner = LiveScanner(service, SceneCamera())
    scanner.start(LiveOptions(reference_ids=[view['id']], scan_mode='moving_camera', volume_resolution=64, interval_seconds=.1))
    deadline = time.monotonic() + 15
    while service.busy and service.live_status['integrated'] < 3 and time.monotonic() < deadline:
        time.sleep(.02)
    scanner.stop()
    service.thread.join(5)
    assert service.live_status['integrated'] >= 3, service.status()
    assert service.live_status['registration']['scene_inliers'] >= 18
    assert service.live_status['camera_pose'][0][3] > .005
    assert service.result['triangles'] > 0


def test_camera_localization_continues_when_foreground_matching_is_lost(monkeypatch):
    from depthcloud.vision.reference import ReferenceBank

    rng = np.random.default_rng(30)
    gray = rng.integers(100, 230, (240, 320), dtype=np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    image[60:180, 140:180] = [200, 55, 20]
    camera = TestCamera()
    camera.image = image
    service = ReconstructionService(TestModels(), CalibrationService())
    snap = service.freeze(FramePacket(1, 1, 1, image), {})
    view = service.capture(CaptureView(snapshot_id=snap['id'], roi=ROI(x=.4,y=.2,width=.2,height=.6)))

    def missing(*args):
        raise ValueError('No reference matches this occluded object')

    monkeypatch.setattr(ReferenceBank, 'locate', missing)
    scanner = LiveScanner(service, camera)
    scanner.start(LiveOptions(reference_ids=[view['id']], scan_mode='moving_camera', volume_resolution=64, interval_seconds=.1))
    deadline = time.monotonic() + 10
    while service.busy and service.live_status['rejected'] < 2 and time.monotonic() < deadline:
        time.sleep(.02)
    assert service.live_status['camera_tracking'] == 'tracked', service.status()
    scanner.stop()
    service.thread.join(5)
    assert service.live_status['camera_tracking'] == 'stopped', service.status()
    assert service.live_status['rejected'] >= 2
    assert service.live_status['integrated'] == 0
    assert service.live_status['camera_pose'] is not None
    assert 'Camera tracked; mesh paused' in service.live_status['last_error']
    assert service.result is None
