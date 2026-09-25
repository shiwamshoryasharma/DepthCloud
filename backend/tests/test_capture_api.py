from fastapi.testclient import TestClient

from depthcloud.main import create_app
from depthcloud.workflow.capture_scan import CaptureScanner


def test_capture_api_validates_and_shares_job_ownership(tmp_path):
    app = create_app()
    service = app.state.reconstruction
    app.state.capture_scan = CaptureScanner(service, app.state.camera, tmp_path / 'session')
    # No lifespan needed: these routes must reject before loading models/camera.
    with TestClient(app, raise_server_exceptions=True) as client:
        assert client.post('/api/reconstruction/capture/start', json={'reference_ids': ['x'], 'target_frames': 1001}).status_code == 422
        assert client.post('/api/reconstruction/capture/start', json={'reference_ids': ['x', 'x']}).status_code == 422
        assert client.post('/api/reconstruction/capture/build').status_code == 422
        assert client.post('/api/reconstruction/capture/resume').status_code == 422
        assert client.get('/api/reconstruction/capture/image/-1').status_code == 422
        service.busy = True
        assert client.post('/api/reconstruction/capture/discard').status_code == 409
        service.busy = False
        response = client.post('/api/reconstruction/capture/discard')
        assert response.status_code == 200 and response.json()['capture']['phase'] == 'empty'
