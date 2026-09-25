import numpy as np
import pytest

from depthcloud.reconstruction.registration import make_cloud, register_clouds
from depthcloud.vision.masking import choose_target, foreground_mask
from depthcloud.workflow.schemas import PipelineOptions


def test_registration_refines_known_rigid_pose_and_records_real_quality():
    x, y = np.meshgrid(np.linspace(-.1, .1, 20), np.linspace(-.07, .07, 17))
    points = np.column_stack([x.ravel(), y.ravel(), (1 + x*x*3 + y*y).ravel()])
    colors = np.ones_like(points) * .5
    transform = np.eye(4)
    transform[:3, 3] = [.025, -.015, .03]
    target = points + transform[:3, 3]
    fitted, metrics = register_clouds(make_cloud(points, colors), make_cloud(target, colors), PipelineOptions(), transform)
    np.testing.assert_allclose(fitted[:3, 3], transform[:3, 3], atol=.003)
    assert metrics['fitness'] > .9 and metrics['rmse'] < .003
    assert metrics['accepted']


def test_registration_rejects_insufficient_geometry():
    cloud = make_cloud(np.zeros((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match='points'):
        register_clouds(cloud, cloud, PipelineOptions())


def test_detector_target_selection_does_not_pick_high_confidence_background():
    background = {'label':'person','score':.99,'box':[0,0,20,20]}
    object_box = {'label':'cup','score':.8,'box':[45,45,80,80]}
    selected, box, warning = choose_target([background,object_box],(40,40,90,90))
    assert selected['label'] == 'cup' and box == (45,45,80,80) and warning is None
    with pytest.raises(ValueError, match='No DETR'):
        choose_target([background],(40,40,90,90),True)
    selected, _, warning = choose_target([],(40,40,90,90))
    assert selected is None and 'manual ROI' in warning


def test_grabcut_mask_stays_inside_roi_and_retains_actual_foreground():
    image = np.full((100,100,3),240,np.uint8)
    image[30:70,35:65] = [20,20,180]
    mask = foreground_mask(image,(20,20,80,80))
    assert mask[50,50] == 255
    assert not mask[:20].any() and not mask[:,80:].any()
    assert np.count_nonzero(mask) < 2000


def test_depth_cleanup_rejects_isolated_background_depth_without_changing_raw_depth():
    from depthcloud.vision.masking import filter_mask_depth
    depth = np.full((20, 20), .4, np.float32)
    depth[:2, :2] = 3.0
    original = depth.copy()
    mask = np.full((20, 20), 255, np.uint8)
    cleaned, removed = filter_mask_depth(depth, mask, 8)
    assert not cleaned[:2, :2].any()
    assert cleaned[10, 10] == 255 and removed == 4
    np.testing.assert_array_equal(depth, original)
    np.testing.assert_array_equal(mask, np.full((20, 20), 255, np.uint8))


def test_icp_rejects_a_pose_with_no_overlap_instead_of_forcing_merge():
    x,y=np.meshgrid(np.linspace(-.1,.1,20),np.linspace(-.1,.1,20))
    points=np.column_stack([x.ravel(),y.ravel(),(1+x*x+y*y).ravel()])
    source=make_cloud(points,np.ones_like(points))
    target=make_cloud(points+[3,0,0],np.ones_like(points))
    with pytest.raises(ValueError,match='Registration rejected'):
        register_clouds(source,target,PipelineOptions(),np.eye(4))


def test_edge_overlap_cannot_replace_the_object_at_roi_center():
    background = {"label": "person", "score": .99, "box": [860, 3, 1035, 329]}
    roi = (921, 7, 1267, 526)
    target, box, warning = choose_target([background], roi)
    assert target is None and box == roi and "manual ROI" in warning
    with pytest.raises(ValueError, match="No DETR"):
        choose_target([background], roi, True)
