"""Single-trial windowed BOLD feature extraction (HRF-delay-centered volume averaging)."""
import numpy as np
from nilearn.image import index_img, load_img, mean_img, resample_to_img
from nilearn.maskers import NiftiMasker

from . import constants as C
from .events import load_subject_events
from .paths import resolve_bold_path


def extract_subject_samples(subject, tr_subject, s1_mask, expected_n_voxels=None, return_masker=False):
    _, stim_events = load_subject_events(subject)

    run1_path = resolve_bold_path(subject, run=1, space_name=C.SPACE)
    if run1_path is None:
        raise FileNotFoundError(f"Missing run-1 bold for {subject}")

    run1_img = load_img(str(run1_path))
    ref = index_img(run1_img, 0)
    local_mask = resample_to_img(s1_mask, ref, interpolation="nearest")
    masker = NiftiMasker(mask_img=local_mask, standardize=None).fit(run1_img)

    X_list, y_list, run_list = [], [], []

    for run in range(1, C.N_RUNS_PER_SUBJECT + 1):
        bold_path = resolve_bold_path(subject, run=run, space_name=C.SPACE)
        if bold_path is None:
            continue

        img = load_img(str(bold_path))
        run_len = img.shape[3]
        run_events = stim_events[stim_events["run"] == run].sort_values("onset")

        for _, ev in run_events.iterrows():
            center = int(np.round((float(ev["onset"]) + C.HRF_DELAY) / tr_subject))
            vols = list(range(max(0, center - C.WINDOW), min(run_len, center + C.WINDOW + 1)))
            if len(vols) == 0:
                continue

            avg_img = mean_img(index_img(img, vols), copy_header=True)
            feat = masker.transform(avg_img).ravel()

            if expected_n_voxels is not None and feat.shape[0] != expected_n_voxels:
                raise ValueError(f"Voxel mismatch for {subject}, run {run}: {feat.shape[0]} vs {expected_n_voxels}")

            X_list.append(feat.astype(np.float32))
            y_list.append(int(ev["label"]))
            run_list.append(int(run))

    if len(X_list) == 0:
        raise RuntimeError(f"No usable samples for {subject}")

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(run_list, dtype=np.int64)

    if return_masker:
        return X, y, groups, tr_subject, masker
    return X, y, groups, tr_subject