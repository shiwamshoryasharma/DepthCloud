import cv2
import numpy as np
import pytest

from depthcloud.reconstruction.scene_tracking import SceneTracker, solve_scene_pose
from depthcloud.workflow.observations import Observation
from depthcloud.workflow.schemas import Intrinsics


def intrinsics():
    return Intrinsics(fx=400, fy=400, cx=319.5, cy=239.5, width=640, height=480)


def test_pnp_recovers_camera_matrix_and_depth_scale_despite_outliers():
    rng = np.random.default_rng(12)
    points = np.column_stack([rng.uniform(-.5,.5,120), rng.uniform(-.35,.35,120), rng.uniform(1.,2.,120)])
    rvec = np.array([.02,.08,-.01])
    translation = np.array([-.07,.01,.02])
    intr = intrinsics()
    matrix = np.array([[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1.]])
    pixels, _ = cv2.projectPoints(points,rvec,translation,matrix,np.zeros(5))
    pixels = pixels.reshape(-1,2)
    rotated = points @ cv2.Rodrigues(rvec)[0].T + translation
    depths = rotated[:,2] / 1.12
    pixels[:20] = rng.uniform([0,0],[640,480],(20,2))
    transform, scale, metrics = solve_scene_pose(points,pixels,depths,intr)
    expected = np.eye(4)
    expected[:3,:3] = cv2.Rodrigues(rvec)[0]
    expected[:3,3] = translation
    assert np.allclose(transform,np.linalg.inv(expected),atol=1e-4)
    assert scale == pytest.approx(1.12,abs=1e-4)
    assert metrics['scene_inliers'] >= 95


def test_no_scene_support_rejects_instead_of_identity_pose():
    with pytest.raises(ValueError,match='scene'):
        solve_scene_pose(np.zeros((10,3)),np.zeros((10,2)),np.ones(10),intrinsics())


def observation(image, index):
    h,w=image.shape[:2]
    mask=np.zeros((h,w),np.uint8)
    mask[140:340,260:380]=255
    view=Observation(str(index),'live',index,index,image,{},(260,140,380,340),{})
    view.result={'mask':mask,'depth':np.ones((h,w),np.float32),'intrinsics':intrinsics().model_dump()}
    return view


def test_background_tracks_camera_without_any_object_features_and_bounds_keyframes():
    rng=np.random.default_rng(3)
    image=rng.integers(0,255,(480,640,3),dtype=np.uint8)
    image[130:350,250:390]=[200,55,20]  # featureless object, excluded from tracking
    first=observation(image,1)
    tracker=SceneTracker(max_keyframes=3)
    tracker.commit(first,np.eye(4),1.)
    shifted=cv2.warpAffine(image,np.float32([[1,0,-16],[0,1,0]]),(640,480))
    current=observation(shifted,2)
    pose,scale,metrics=tracker.estimate(current)
    assert pose[0,3] == pytest.approx(.04,abs=.004)
    assert scale == pytest.approx(1.,abs=.02)
    assert metrics['scene_inliers'] >= 18
    for index in range(2,8):
        next_pose = pose.copy()
        next_pose[0,3] += index * .04
        tracker.commit(observation(shifted,index),next_pose,scale)
    assert len(tracker.keyframes) == 3
    assert all('bgr' not in entry for entry in tracker.keyframes)
