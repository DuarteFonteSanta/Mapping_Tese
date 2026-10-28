"""Destrieux atlas based left-S1 (G_postcentral) ROI masking."""
import numpy as np
from nilearn.datasets import fetch_atlas_destrieux_2009
from nilearn.image import index_img, load_img, new_img_like, resample_to_img
from nilearn.maskers import NiftiMasker

ROI_LABEL_SUBSTRING = "L G_postcentral"


def build_s1_mask():
    atlas = fetch_atlas_destrieux_2009()
    atlas_img = load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    s1_indices = [
        i for i, lab in enumerate(atlas.labels)
        if ROI_LABEL_SUBSTRING in str(lab) and i != 0
    ]
    mask_data = np.isin(atlas_data, s1_indices).astype("uint8")
    return new_img_like(atlas_img, mask_data)


def build_masker(reference_bold_img, s1_mask=None, standardize=None):
    if s1_mask is None:
        s1_mask = build_s1_mask()
    ref_img = index_img(reference_bold_img, 0)
    s1_mask_resampled = resample_to_img(s1_mask, ref_img, interpolation="nearest")
    masker = NiftiMasker(mask_img=s1_mask_resampled, standardize=standardize)
    masker.fit(reference_bold_img)
    return masker

def fit_masker(reference_bold_path, s1_mask=None, standardize=None):
    """Load a reference BOLD run from disk and build its ROI masker."""
    reference_bold_img = load_img(str(reference_bold_path))
    return build_masker(reference_bold_img, s1_mask=s1_mask, standardize=standardize)