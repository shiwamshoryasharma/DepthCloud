import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest


@pytest.fixture(autouse=True)
def isolated_capture_sessions(tmp_path, monkeypatch):
    import depthcloud.main as main
    from depthcloud.workflow.capture_scan import CaptureScanner
    monkeypatch.setattr(main, 'CaptureScanner',
                        lambda service, camera, root: CaptureScanner(service, camera, tmp_path / 'api-capture'))
