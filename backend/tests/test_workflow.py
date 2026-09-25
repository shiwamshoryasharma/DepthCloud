import threading
import time

import numpy as np
import pytest

from depthcloud.camera.buffer import FramePacket
from depthcloud.camera.calibration import CalibrationService
from depthcloud.workflow.schemas import ROI, CaptureView, PipelineOptions
from depthcloud.workflow.service import ReconstructionService, WorkflowBusy


class Models:
    def infer_depth(self, image, focal_px=None):
        return np.ones(image.shape[:2],np.float32), 100.0

    def detect(self,image,threshold):
        return [{'label':'test object','score':.9,'box':[20,20,60,60]}]


def capture(service):
    image=np.full((80,80,3),240,np.uint8)
    image[25:55,25:55]=[20,20,180]
    frozen=service.freeze(FramePacket(1,1.0,1.0,image),{})
    return service.capture(CaptureView(snapshot_id=frozen['id'],roi=ROI(x=.2,y=.2,width=.6,height=.6)))


def wait_done(service):
    deadline=time.monotonic()+15
    while service.status()['busy'] and time.monotonic()<deadline:
        time.sleep(.02)
    assert not service.status()['busy']


def test_pipeline_produces_real_projected_geometry_and_reset_keeps_models():
    models=Models()
    service=ReconstructionService(models,CalibrationService())
    view=capture(service)
    service.start(PipelineOptions(create_mesh=False,stride=1,voxel_size=.003))
    wait_done(service)
    state=service.status()
    assert state['state']=='completed', state
    assert state['result']['points']>30
    assert len(service.geometry('cloud',state['result']['revision']))==state['result']['points']*9*4
    assert service.store.get(view['id']).result['depth'].shape==(80,80)
    service.reset()
    assert service.status()['views']==[] and service.models is models
    assert service.status()['result'] is None


def test_inference_failure_is_visible_and_does_not_publish_fake_geometry():
    class Failed(Models):
        def infer_depth(self,*args):
            raise RuntimeError('local checkpoint unavailable')
    service=ReconstructionService(Failed(),CalibrationService())
    capture(service)
    service.start(PipelineOptions())
    wait_done(service)
    state=service.status()
    assert state['state']=='failed' and state['result'] is None
    assert 'checkpoint' in state['views'][0]['error']


def test_busy_pipeline_rejects_mutations_and_cancel_discards_partial_result():
    entered,release=threading.Event(),threading.Event()
    class Slow(Models):
        def infer_depth(self,*args):
            entered.set()
            assert release.wait(5)
            return super().infer_depth(*args)
    service=ReconstructionService(Slow(),CalibrationService())
    capture(service)
    service.start(PipelineOptions(create_mesh=False))
    assert entered.wait(3)
    with pytest.raises(WorkflowBusy):
        service.reset()
    with pytest.raises(WorkflowBusy):
        service.start(PipelineOptions())
    service.cancel()
    release.set()
    wait_done(service)
    assert service.status()['state']=='cancelled'
    assert service.status()['result'] is None


def test_failed_registration_is_excluded_from_fusion(monkeypatch):
    service=ReconstructionService(Models(),CalibrationService())
    capture(service)
    capture(service)
    def reject(*args):
        raise ValueError('insufficient overlap')
    monkeypatch.setattr('depthcloud.workflow.service.register_clouds',reject)
    service.start(PipelineOptions(create_mesh=False))
    wait_done(service)
    result=service.status()['result']
    assert len(result['accepted'])==1 and len(result['rejected'])==1
    assert 'insufficient overlap' in result['rejected'][0]['reason']
