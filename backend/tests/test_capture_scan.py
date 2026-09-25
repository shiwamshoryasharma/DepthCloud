import time
from types import SimpleNamespace

import numpy as np
import pytest

from depthcloud.camera.buffer import FramePacket
from depthcloud.camera.calibration import CalibrationService
from depthcloud.workflow.capture_scan import CaptureScanner
from depthcloud.workflow.schemas import CaptureOptions
from depthcloud.workflow.service import ReconstructionService, WorkflowBusy


class Camera:
    def __init__(self):
        self.calls = 0
        self.epoch = time.monotonic()

    def latest_for_scan(self):
        self.calls += 1
        return FramePacket(self.calls, 100 + self.calls * 1.3, self.epoch + self.calls * 1.3,
                           np.random.default_rng(9).integers(0, 255, (80, 100, 3), dtype=np.uint8)), 1


def test_collect_only_detections_then_build_sealed_set(tmp_path, monkeypatch):
    import depthcloud.workflow.capture_scan as module
    service = ReconstructionService(SimpleNamespace(infer_depth=lambda *_: pytest.fail('Depth ran during collection')), CalibrationService())
    service.store.views['object'] = SimpleNamespace(id='object', bytes_used=lambda: 0)
    camera = Camera()
    scanner = CaptureScanner(service, camera, tmp_path / 'session')
    mask = np.full((80, 100), 255, np.uint8)
    bank = SimpleNamespace(locate=lambda _: SimpleNamespace(box=(0, 0, 100, 80), mask=mask, metadata=lambda: {'label': 'object'}))
    monkeypatch.setattr(scanner, '_prepare_bank', lambda _: bank)
    monkeypatch.setattr(module.StabilityGate, 'update', lambda *a: {'stable': True, 'reason': 'Ready'})
    builds = []
    monkeypatch.setattr(scanner, '_build', lambda: builds.append((scanner.dataset.count, scanner.dataset.sealed, camera.calls)))
    # Avoid serializing our deliberately minimal reference stub through status().
    monkeypatch.setattr(service.store, 'list', lambda: [])
    scanner.start(CaptureOptions(reference_ids=['object'], target_frames=3, minimum_sharpness=5))
    service.thread.join(5)
    assert not service.busy, service.status()
    assert builds == [(3, True, 4)]
    assert scanner.dataset.count == 3
    camera.latest_for_scan()
    assert scanner.dataset.count == 3
    with pytest.raises(ValueError, match='unfinished'):
        scanner.resume()


def test_build_cancel_and_discard_preserve_single_job_owner(tmp_path):
    service = ReconstructionService(None, CalibrationService())
    scanner = CaptureScanner(service, Camera(), tmp_path / 'session')
    service.busy = True
    with pytest.raises(WorkflowBusy):
        scanner.discard()
    with pytest.raises(WorkflowBusy):
        scanner.build()
    service.busy = False
    with pytest.raises(ValueError, match='3'):
        scanner.build()


def test_pause_retains_captures_and_restart_can_build_without_camera(tmp_path):
    service = ReconstructionService(None, CalibrationService())
    scanner = CaptureScanner(service, None, tmp_path / 'session')
    options = CaptureOptions(reference_ids=['saved-reference'], target_frames=1000)
    scanner.dataset.create(options.model_dump(), [{'id': 'saved-reference', 'label': 'Object'}])
    image = np.zeros((40, 60, 3), np.uint8)
    mask = np.ones((40, 60), np.uint8)
    for i in range(3):
        scanner.dataset.append(image, mask, {'timestamp': i*1.25})
    recovered = CaptureScanner(service, None, tmp_path / 'session')
    assert service.capture_status['phase'] == 'paused'
    assert service.capture_status['collected'] == 3
    recovered.active = service.busy = True
    recovered.pause()
    assert service.cancel_event.is_set() and recovered.dataset.count == 3
    recovered.active = service.busy = False
    recovered.dataset.seal()
    sealed = CaptureScanner(service, None, tmp_path / 'session')
    assert service.capture_status['phase'] == 'ready' and sealed.dataset.sealed
