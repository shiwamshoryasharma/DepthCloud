import numpy as np
import pytest
from pydantic import ValidationError

from depthcloud.reconstruction.geometry import project_depth, rigid_transform
from depthcloud.vision.transforms import transform_preview
from depthcloud.workflow.schemas import ROI, Intrinsics, PipelineOptions, PreviewOptions


def test_roi_preserves_native_coordinates_and_rejects_out_of_image_selection():
    assert ROI(x=0.25, y=0.2, width=0.5, height=0.5).pixels(800, 600) == (200, 120, 600, 420)
    with pytest.raises(ValidationError):
        ROI(x=0.9, y=0, width=0.3, height=0.5)
    with pytest.raises(ValueError):
        ROI(x=0, y=0, width=0.001, height=0.001).pixels(800, 600)


@pytest.mark.parametrize("mode", ["rgb", "vivid", "mono", "chrome", "pixelated", "pseudo_ir", "grayscale"])
def test_preview_transforms_preserve_source_shape_and_storage(mode):
    image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    before = image.copy()
    output = transform_preview(image, PreviewOptions(mode=mode))
    assert output.shape == image.shape and output.dtype == np.uint8
    np.testing.assert_array_equal(image, before)
    if mode == "mono":
        assert set(np.unique(output)).issubset({0, 255})


def test_projection_excludes_background_invalid_depth_and_uses_rgb_meters():
    depth = np.array([[2, np.nan, 0], [3, np.inf, 4]], dtype=np.float32)
    mask = np.array([[255, 255, 255], [0, 255, 255]], dtype=np.uint8)
    bgr = np.full((2, 3, 3), [0, 0, 255], dtype=np.uint8)
    intrinsics = Intrinsics(fx=2, fy=2, cx=1, cy=0, width=3, height=2)
    points, colors = project_depth(depth, bgr, mask, intrinsics, PipelineOptions(depth_max=5, stride=1))
    np.testing.assert_allclose(points, [[-1, 0, 2], [2, 2, 4]])
    np.testing.assert_allclose(colors, [[1, 0, 0], [1, 0, 0]])


def test_calibration_scales_only_with_same_aspect_ratio():
    calibration = Intrinsics(fx=600, fy=610, cx=320, cy=240, width=640, height=480)
    scaled = calibration.scaled(1280, 960)
    assert scaled.fx == 1200 and scaled.cy == 480
    with pytest.raises(ValueError, match="aspect"):
        calibration.scaled(1280, 720)


def test_rigid_registration_recovers_rotation_translation_without_scale():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    target = source @ rotation.T + [2, 3, 4]
    transform = rigid_transform(source, target)
    np.testing.assert_allclose(source @ transform[:3, :3].T + transform[:3, 3], target, atol=1e-6)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)


def test_pipeline_rejects_inverted_depth_range_and_nonfinite_calibration():
    with pytest.raises(ValidationError):
        PipelineOptions(depth_min=5, depth_max=1)
    with pytest.raises(ValidationError):
        Intrinsics(fx=float("nan"), fy=1, cx=0, cy=0, width=3, height=2)

