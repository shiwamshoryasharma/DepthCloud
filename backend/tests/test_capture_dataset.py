import cv2
import numpy as np
import pytest

from depthcloud.vision.stability import StabilityGate
from depthcloud.workflow.capture_dataset import CaptureDataset


def test_capture_cadence_seal_and_restart(tmp_path):
    data = CaptureDataset(tmp_path / 'session')
    data.create({'target_frames': 2}, [])
    image = np.full((40, 60, 3), 120, np.uint8)
    mask = np.full((40, 60), 255, np.uint8)
    data.append(image, mask, {'timestamp': 10.0})
    with pytest.raises(ValueError, match='1.25'):
        data.append(image, mask, {'timestamp': 11.0})
    data.append(image, mask, {'timestamp': 11.25})
    assert data.sealed and data.count == 2
    with pytest.raises(ValueError, match='sealed'):
        data.append(image, mask, {'timestamp': 13.0})
    restored = CaptureDataset(tmp_path / 'session')
    assert restored.count == 2 and restored.sealed
    np.testing.assert_array_equal(restored.image(0), image)


def test_failed_write_does_not_increment_count(tmp_path, monkeypatch):
    data = CaptureDataset(tmp_path / 'session')
    data.create({'target_frames': 1000}, [])
    monkeypatch.setattr(data, 'check_space', lambda *a: (_ for _ in ()).throw(ValueError('Disk budget')))
    with pytest.raises(ValueError, match='Disk budget'):
        data.append(np.zeros((30, 30, 3), np.uint8), np.ones((30, 30), np.uint8), {'timestamp': 1})
    assert data.count == 0


def test_blurred_and_shaken_frames_are_not_stable():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    gate = StabilityGate()
    assert not gate.update(image, 0)['stable']
    assert not gate.update(image, .1)['stable']
    assert gate.update(image, .4)['stable']
    moved = cv2.warpAffine(image, np.float32([[1, 0, 30], [0, 1, 0]]), (320, 240))
    assert not gate.update(moved, .5)['stable']
    assert not gate.update(cv2.GaussianBlur(image, (31, 31), 9), .6)['stable']


def test_exact_thousand_limit_and_immutable_snapshot(tmp_path, monkeypatch):
    data = CaptureDataset(tmp_path / 'session')
    data.create({'target_frames': 1000}, [])
    monkeypatch.setattr(data, 'write_image', lambda *a: None)
    monkeypatch.setattr(data, 'save', lambda: None)
    image = np.zeros((4, 4, 3), np.uint8)
    mask = np.ones((4, 4), np.uint8)
    for i in range(1000):
        data.append(image, mask, {'timestamp': i * 1.25})
    assert data.count == 1000 and data.sealed
    with pytest.raises(ValueError, match='sealed'):
        data.append(image, mask, {'timestamp': 2000})
    assert data.count == 1000


def test_corrupt_manifest_does_not_prevent_backend_recovery(tmp_path):
    from depthcloud.camera.calibration import CalibrationService
    from depthcloud.workflow.capture_scan import CaptureScanner
    from depthcloud.workflow.service import ReconstructionService
    folder = tmp_path / 'session'
    folder.mkdir()
    (folder / 'manifest.json').write_text('{broken', encoding='utf-8')
    service = ReconstructionService(None, CalibrationService())
    scanner = CaptureScanner(service, None, folder)
    assert service.capture_status['phase'] == 'failed'
    scanner.discard()
    assert service.capture_status['phase'] == 'empty'


def test_resolution_change_cannot_enter_existing_set(tmp_path):
    data = CaptureDataset(tmp_path / 'session')
    data.create({'target_frames': 1000}, [])
    data.append(np.zeros((40, 60, 3), np.uint8), np.ones((40, 60), np.uint8), {'timestamp': 0})
    with pytest.raises(ValueError, match='resolution'):
        data.append(np.zeros((40, 80, 3), np.uint8), np.ones((40, 80), np.uint8), {'timestamp': 2})
    assert data.count == 1


def test_nested_native_cache_is_accounted_and_discarded(tmp_path):
    data = CaptureDataset(tmp_path / 'session')
    data.create({'target_frames': 1000}, [])
    nested = data.root / 'sfm' / 'key' / 'selected'
    nested.mkdir(parents=True)
    (nested / 'points.bin').write_bytes(b'1234567')
    assert data.refresh_bytes() == 7
    unrelated = tmp_path / 'keep.txt'
    unrelated.write_text('keep')
    data.discard()
    assert not list(data.root.iterdir()) and unrelated.read_text() == 'keep'


def test_validated_mesh_restores_after_restart_and_rejects_modified_file(tmp_path):
    import hashlib

    import open3d as o3d

    from depthcloud.camera.calibration import CalibrationService
    from depthcloud.workflow.capture_scan import CaptureScanner
    from depthcloud.workflow.schemas import CaptureOptions
    from depthcloud.workflow.service import ReconstructionService
    data = CaptureDataset(tmp_path / 'session')
    data.create(CaptureOptions(reference_ids=['example']).model_dump(), [])
    path = data.root / 'surface-v2.ply'
    mesh = o3d.geometry.TriangleMesh.create_sphere(.1)
    mesh.paint_uniform_color([0, 0, 1])
    o3d.io.write_triangle_mesh(str(path), mesh)
    data.data.update(sealed=True, report={'algorithm_version': 2, 'accepted': ['0','1','2'],
        'rejected': [], 'quality': {'median_silhouette_iou': .9}, 'geometry_kind': 'visual_hull',
        'surface_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    data.save()
    service = ReconstructionService(None, CalibrationService())
    CaptureScanner(service, None, data.root)
    assert service.capture_status['phase'] == 'completed'
    assert service.result['geometry_kind'] == 'visual_hull'
    assert service.capture_status.get('quality') is None  # Capture quality is separate.
    assert service.buffers['mesh']
    path.write_bytes(path.read_bytes() + b'corrupt')
    service = ReconstructionService(None, CalibrationService())
    CaptureScanner(service, None, data.root)
    assert service.result is None and service.capture_status['phase'] == 'ready'
    assert 'integrity' in service.capture_status['message']
