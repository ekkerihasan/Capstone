"""Stage 6.1 — the preprocessing pipeline itself, on the phantom volumes."""
import numpy as np
import pytest
from monai.utils import set_determinism

from src.config import HU_MAX, HU_MIN, PATCH_SIZE, SEED, SPACING_MM
from src.data import load_msd_datalist
from src.transforms import build_transforms, patch_transform


@pytest.fixture(scope="module")
def one_case(msd_phantom):
    return dict(load_msd_datalist(msd_phantom)[0])


@pytest.fixture(scope="module")
def preprocessed(one_case):
    set_determinism(SEED)
    return build_transforms(train=False)(dict(one_case))


def test_output_is_channel_first_and_paired(preprocessed):
    image, label = preprocessed["image"], preprocessed["label"]
    assert image.shape[0] == 1 and label.shape[0] == 1
    assert image.shape == label.shape          # voxel-aligned after resampling


def test_resampled_to_target_spacing(preprocessed):
    spacing = preprocessed["image"].pixdim
    assert np.allclose(np.asarray(spacing), SPACING_MM, atol=1e-3)


def test_reoriented_to_ras(preprocessed):
    import nibabel as nib
    affine = np.asarray(preprocessed["image"].affine)
    assert "".join(nib.orientations.aff2axcodes(affine)) == "RAS"


def test_intensities_are_windowed_to_unit_range(preprocessed):
    image = np.asarray(preprocessed["image"])
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert image.max() > image.min()           # not a constant volume


def test_hu_window_maps_the_right_endpoints(one_case):
    """-100 HU must land on 0 and 240 HU on 1, with everything outside clipped."""
    from monai.transforms import ScaleIntensityRanged
    tf = ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX,
                              b_min=0.0, b_max=1.0, clip=True)
    probe = np.array([[-1000.0, HU_MIN, (HU_MIN + HU_MAX) / 2, HU_MAX, 3000.0]])
    out = np.asarray(tf({"image": probe})["image"]).ravel()
    assert np.allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_labels_survive_resampling_unchanged(preprocessed):
    """Nearest-neighbour interpolation must not invent or blur class ids."""
    label = np.asarray(preprocessed["label"])
    assert set(np.unique(label).tolist()) <= {0, 1, 2}
    assert (label == 2).sum() > 0              # the tumor is still there


def test_foreground_crop_removes_air(one_case, preprocessed):
    from src.transforms import base_transforms
    from monai.transforms import Compose
    uncropped = Compose(base_transforms(crop_foreground=False))(dict(one_case))
    assert np.prod(preprocessed["image"].shape) < np.prod(uncropped["image"].shape)


def test_augmentation_perturbs_image_and_label_together(one_case):
    set_determinism(SEED)
    train_tf = build_transforms(train=True, elastic_prob=1.0)
    augmented = train_tf(dict(one_case))
    set_determinism(SEED)
    plain = build_transforms(train=False)(dict(one_case))

    a_img, a_lab = np.asarray(augmented["image"]), np.asarray(augmented["label"])
    p_img = np.asarray(plain["image"])

    assert a_img.shape == a_lab.shape                     # image and mask stay aligned
    changed = a_img.shape != p_img.shape or not np.allclose(a_img, p_img)
    assert changed, "augmentation left the volume untouched"
    assert set(np.unique(a_lab).tolist()) <= {0, 1, 2}    # no interpolated class ids
    assert a_img.min() >= 0.0 and a_img.max() <= 1.0


def test_validation_pipeline_has_no_random_transforms():
    from monai.transforms import Randomizable
    val_tf = build_transforms(train=False)
    assert not any(isinstance(t, Randomizable) for t in val_tf.transforms)
    assert any(isinstance(t, Randomizable) for t in build_transforms(train=True).transforms)


def test_patch_sampling_returns_fixed_size_patches(preprocessed):
    """Stage 6.2 trains on fixed patches; the sampler must honour the size and
    prefer patches that contain annotated foreground."""
    set_determinism(SEED)
    sampler = patch_transform(patch_size=(16, 16, 16), num_samples=4, pos=1.0, neg=0.0)
    patches = sampler(preprocessed)
    assert len(patches) == 4
    for p in patches:
        assert tuple(p["image"].shape[1:]) == (16, 16, 16)
        assert tuple(p["label"].shape[1:]) == (16, 16, 16)
        assert (np.asarray(p["label"]) > 0).sum() > 0        # pos=1.0, neg=0.0


def test_patch_sampling_pads_volumes_smaller_than_the_patch(preprocessed):
    """A slim patient can be shorter than 96 voxels in one axis after foreground
    cropping; the sampler must pad rather than raise mid-epoch."""
    set_determinism(SEED)
    volume_shape = tuple(preprocessed["image"].shape[1:])
    oversized = tuple(s + 8 for s in volume_shape)
    patches = patch_transform(patch_size=oversized, num_samples=1)(preprocessed)
    assert tuple(patches[0]["image"].shape[1:]) == oversized


def test_train_pipeline_with_patches_yields_uniform_cubes(one_case):
    """The shape that stage 6.2 will actually collate into a batch of 2."""
    set_determinism(SEED)
    tf = build_transforms(train=True, patches=True, num_samples=2, elastic_prob=1.0)
    samples = tf(dict(one_case))
    assert len(samples) == 2
    for s in samples:
        assert tuple(s["image"].shape) == (1,) + PATCH_SIZE
        assert tuple(s["label"].shape) == (1,) + PATCH_SIZE
