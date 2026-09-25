import numpy as np
import pytest

from depthcloud.reconstruction.multiview import validate_agreement
from depthcloud.reconstruction.sfm import shared_metric_scale


def test_shared_scale_is_one_robust_number_not_per_view_deformation():
    scale, variation = shared_metric_scale([0.11, 0.10, 0.102, 0.098, 0.099, 0.3])
    assert 0.099 < scale < 0.103 and variation < 0.05


def test_inconsistent_scale_is_not_silently_merged():
    with pytest.raises(ValueError, match="disagree"):
        shared_metric_scale([0.1, 0.2, 0.4, 0.8])


def test_bad_shape_cannot_be_published_as_completed():
    with pytest.raises(ValueError, match="does not explain"):
        validate_agreement([{"silhouette_iou": x} for x in [0.3, 0.5, 0.42, 0.48]])
    assert validate_agreement([{"silhouette_iou": x} for x in [0.8, 0.9, 0.85]])[
        "median_silhouette_iou"
    ] == np.median([0.8, 0.9, 0.85])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_shape_validation_cannot_pass(invalid):
    with pytest.raises(ValueError, match="does not explain"):
        validate_agreement([{"silhouette_iou": x} for x in [invalid, 0.9, 0.9]])
