import numpy as np
from fastapi.testclient import TestClient

from depthcloud.camera.buffer import FramePacket
from depthcloud.config import Settings
from depthcloud.main import create_app


def test_session_capture_images_delete_reset_and_validation():
    app=create_app(Settings())
    with TestClient(app) as client:
        assert client.get('/api/reconstruction/state').status_code==200
        assert client.post('/api/reconstruction/freeze').status_code==422
        snapshot=app.state.reconstruction.freeze(FramePacket(8,123.0,1.0,np.zeros((80,100,3),np.uint8)),{})
        payload={'snapshot_id':snapshot['id'],'roi':{'x':.2,'y':.2,'width':.5,'height':.5},'label':'front'}
        response=client.post('/api/reconstruction/views',json=payload)
        assert response.status_code==200
        view=response.json()
        image=client.get('/api/reconstruction/views/'+view['id']+'/image/crop')
        assert image.status_code==200 and image.content.startswith(b'\x89PNG')
        assert client.get('/api/reconstruction/views/'+view['id']+'/image/depth').status_code==422
        assert client.patch('/api/reconstruction/views/'+view['id'],json={'selected':False}).status_code==200
        assert client.post('/api/reconstruction/start',json={}).status_code==422
        assert client.delete('/api/reconstruction/views/'+view['id']).status_code==200
        assert client.post('/api/reconstruction/reset').json()['views']==[]
        assert client.get('/api/reconstruction/export/ply').status_code==422


def test_preview_options_and_calibration_are_validated_and_reported():
    with TestClient(create_app(Settings())) as client:
        assert client.patch('/api/camera/preview',json={'mode':'thermal'}).status_code==422
        assert client.patch('/api/camera/preview',json={'mode':'pseudo_ir'}).status_code==200
        assert client.get('/api/camera/status').json()['preview_options']['mode']=='pseudo_ir'
        calibration={'fx':600,'fy':600,'cx':320,'cy':240,'width':640,'height':480}
        assert client.put('/api/calibration',json=calibration).status_code==200
        assert client.get('/api/calibration').json()['intrinsics']['fx']==600
        assert client.post('/api/calibration/solve').status_code==422
        assert client.delete('/api/calibration').status_code==200
        assert client.get('/api/calibration').json()['intrinsics'] is None


def test_live_generation_rejects_bad_reference_and_idle_camera():
    with TestClient(create_app(Settings())) as client:
        assert client.post('/api/reconstruction/live/start', json={}).status_code == 422
        assert client.post('/api/reconstruction/live/start', json={'reference_ids':['missing']}).status_code == 422
        assert client.post('/api/reconstruction/live/start', json={'reference_ids':['missing'],'volume_resolution':1024}).status_code == 422
        assert client.post('/api/reconstruction/live/stop').status_code == 200
        assert client.get('/api/reconstruction/live/image').status_code == 404


def test_detection_only_reference_depth_request_returns_explicit_error():
    app = create_app(Settings())
    with TestClient(app) as client:
        snapshot = app.state.reconstruction.freeze(FramePacket(1,1,1,np.zeros((80,100,3),np.uint8)),{})
        payload = {'snapshot_id':snapshot['id'],'roi':{'x':.2,'y':.2,'width':.5,'height':.5}}
        view = client.post('/api/reconstruction/views',json=payload).json()
        app.state.reconstruction.store.get(view['id']).result = {'mask':np.ones((80,100),np.uint8)*255,'reference_only':True}
        response = client.get('/api/reconstruction/views/'+view['id']+'/image/depth')
        assert response.status_code == 422 and 'reference' in response.json()['detail']
        assert client.get('/api/reconstruction/views/'+view['id']+'/image/mask').status_code == 200
