
import numpy as np

from depthcloud.reconstruction.saved_scan import build_saved_mesh
from depthcloud.workflow.capture_dataset import CaptureDataset
from depthcloud.workflow.schemas import CaptureOptions, Intrinsics


def test_saved_views_create_triangle_mesh_without_camera_or_reference_pixels(tmp_path, monkeypatch):
    import depthcloud.reconstruction.saved_scan as module
    width, height, focal = 160, 120, 160.
    intr = Intrinsics(fx=focal, fy=focal, cx=79.5, cy=59.5, width=width, height=height)
    options = CaptureOptions(reconstruction_method='depth_fusion', reference_ids=['reference-only'], target_frames=3, volume_resolution=64)
    dataset = CaptureDataset(tmp_path / 'session')
    dataset.create(options.model_dump(), [{'id': 'reference-only'}], intr.model_dump())
    y, x = np.mgrid[:height, :width]
    rays = np.stack([(x-79.5)/focal, (y-59.5)/focal, np.ones_like(x)], axis=-1)
    depths, camera_positions = [], [-.03, 0., .03]
    for i, camera_x in enumerate(camera_positions):
        center = np.array([-camera_x, 0, .6])
        a = (rays*rays).sum(axis=2)
        b = -2*(rays*center).sum(axis=2)
        discriminant = b*b-4*a*((center*center).sum()-.12**2)
        mask = discriminant > 0
        depth = np.where(mask, (-b-np.sqrt(np.maximum(discriminant, 0)))/(2*a), 2.).astype(np.float32)
        image = np.full((height, width, 3), [170, 90, 40], np.uint8)
        image[mask] = [220, 70, 20]
        dataset.append(image, (mask*255).astype(np.uint8), {'timestamp': i*1.25})
        depths.append(depth)
    class Models:
        def __init__(self):
            self.calls = 0
        def infer_depth(self, image, focal_px=None):
            index = self.calls
            assert np.all(image[0, 0] == [170, 90, 40]), 'Depth Pro must retain scene context'
            self.calls += 1
            return depths[index], focal
    class KnownPoseTracker:
        def __init__(self, **_):
            self.current = None
        def estimate(self, view):
            i = int(view.id)
            points = np.random.default_rng(4).uniform([-.4, -.3, 1.5], [.4, .3, 2], (100, 3))
            points[:, 0] -= camera_positions[i]
            self.current = {'points': points, 'pixels': np.zeros((100, 2)), 'descriptors': np.zeros((100, 32), np.uint8), 'intrinsics': intr}
            pose = np.eye(4)
            pose[0, 3] = camera_positions[i] - camera_positions[0]
            return pose, 1., {'keyframe_id': str(i-1)} if i else {}
        def commit(self, *args):
            pass
    monkeypatch.setattr(module, 'SceneTracker', KnownPoseTracker)
    stages = []
    models = Models()
    cloud, mesh, buffers, result = build_saved_mesh(dataset, models, options, lambda phase, *a, **k: stages.append(phase), lambda: None)
    assert models.calls == 3
    assert result['accepted'] == ['0', '1', '2']
    assert len(mesh.triangles) > 100 and len(cloud.points) > 100
    assert result['bounds'][1][2] < 1, 'Background plane must not enter the mesh'
    assert buffers['mesh'] and 'optimizing' in stages
    assert max(i for i, stage in enumerate(stages) if stage == 'filtering') < stages.index('depth_alignment')
    assert result['complete_coverage'] is False
    assert (dataset.root / 'manifest.json').exists()
