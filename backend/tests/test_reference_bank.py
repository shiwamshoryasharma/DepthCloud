import numpy as np
import pytest

from depthcloud.vision.reference import ReferenceBank
from depthcloud.workflow.schemas import LiveOptions


def sample(color, box=(15, 20, 40, 100), shape=(160, 240)):
    image = np.full((*shape, 3), 210, np.uint8)
    mask = np.zeros(shape, np.uint8)
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = color
    mask[y0:y1, x0:x1] = 255
    return image, mask


def bank():
    refs = ReferenceBank()
    for name, color in [('front', [200, 55, 20]), ('back', [20, 65, 200])]:
        image, mask = sample(color)
        refs.add(name, name, image, mask)
    return refs


@pytest.mark.parametrize('name,color', [('front', [200, 55, 20]), ('back', [20, 65, 200])])
def test_every_angle_relocates_outside_original_roi(name, color):
    image, expected = sample(color, (170, 45, 195, 125))
    match = bank().locate(image)
    assert match.reference_id == name
    assert np.count_nonzero((match.mask > 0) & (expected > 0)) > expected.sum() / 255 * .95
    assert match.box[0] > 150


def test_same_color_wrong_shape_is_not_the_target():
    image, _ = sample([200, 55, 20], (170, 45, 195, 125))
    image[20:70, 60:110] = [200, 55, 20]  # Larger square distractor
    match = bank().locate(image)
    assert match.mask[50, 80] == 0 and match.mask[80, 180] == 255


def test_indistinguishable_instances_reject_instead_of_picking_one():
    image, _ = sample([200, 55, 20], (170, 45, 195, 125))
    image[20:100, 15:40] = [200, 55, 20]
    with pytest.raises(ValueError, match='ambiguous'):
        bank().locate(image)


def test_missing_object_and_cutoff_object_are_rejected():
    with pytest.raises(ValueError, match='No reference'):
        bank().locate(np.full((160, 240, 3), 210, np.uint8))
    image, _ = sample([200, 55, 20], (0, 20, 25, 100))
    with pytest.raises(ValueError):
        bank().locate(image)


def test_duplicate_angles_of_same_object_are_not_ambiguous():
    refs = bank()
    image, mask = sample([200, 55, 20])
    refs.add('front2', 'front2', image, mask)
    assert refs.locate(image).reference_id in ('front', 'front2')


def test_reference_set_validation_rejects_empty_duplicates_and_legacy_single_reference():
    assert LiveOptions(reference_ids=['a', 'b']).reference_ids == ['a', 'b']
    for values in ({'reference_ids': []}, {'reference_ids': ['a', 'a']}, {'reference_ids': ['']}, {'reference_id': 'a'}):
        with pytest.raises(ValueError):
            LiveOptions(**values)


def test_visual_features_can_select_another_angle_and_reject_color_lookalikes():
    class Describer:
        calls = 0
        reject = False

        def __call__(self, crops):
            self.calls += 1
            vector = [1., 0., 0.] if self.calls == 1 else [0., 1., 0.]
            if self.reject:
                vector = [0., 0., 1.]
            return np.tile(vector, (len(crops), 1))

    describe = Describer()
    refs = ReferenceBank(describe)
    image, mask = sample([200, 55, 20])
    refs.add('first', 'first angle', image, mask)
    refs.add('second', 'second angle', image, mask)
    current, _ = sample([200, 55, 20], (170, 45, 195, 125))
    assert refs.locate(current).reference_id == 'second'
    describe.reject = True
    with pytest.raises(ValueError, match='visual appearance'):
        refs.locate(current)


def test_tracked_spatial_prior_disambiguates_identical_objects_without_fixed_reference_roi():
    image, _ = sample([200, 55, 20], (170, 45, 195, 125))
    image[20:100, 15:40] = [200, 55, 20]
    match = bank().locate(image, prior_box=(150, 25, 215, 145))
    assert match.mask[80, 180] == 255 and match.mask[50, 25] == 0
