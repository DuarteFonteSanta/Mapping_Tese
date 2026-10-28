"""Compressed .npz caching of per-subject extracted samples (X, y, groups, tr_subject)."""
from pathlib import Path

import joblib
import numpy as np

from . import constants as C
from .events import load_subject_events
from .extraction import extract_run_features, extract_subject_samples
from .paths import discover_subjects, get_tr, resolve_bold_path
from .roi import build_s1_mask, fit_masker


def cache_path(cache_dir, scheme, subject):
    return Path(cache_dir) / scheme / f"{subject}.npz"


def save_subject_samples(path, X, y, groups, tr_subject):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, groups=groups, tr_subject=np.float64(tr_subject))


def load_subject_samples(path):
    data = np.load(path)
    return data["X"], data["y"], data["groups"], float(data["tr_subject"])


def get_subject_samples(subject, tr_subject, s1_mask, path, expected_n_voxels=None, force_recompute=False):
    """Load from cache if present, otherwise extract (int labels) and save."""
    path = Path(path)
    if path.exists() and not force_recompute:
        return load_subject_samples(path)

    X, y, groups, tr_subject = extract_subject_samples(subject, tr_subject, s1_mask, expected_n_voxels=expected_n_voxels)
    save_subject_samples(path, X, y, groups, tr_subject)
    return X, y, groups, tr_subject


def build_subject_cache(subject, s1_mask, cache_dir):
    """Extract one subject's region-labeled features + masker and write them to cache_dir. Returns the dataset path, or None if skipped."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tr_subject = get_tr(subject, run=1, default_tr=2.0)
    _, stim_events = load_subject_events(subject)

    run1_path = resolve_bold_path(subject, run=1, space_name=C.SPACE)
    if run1_path is None:
        return None

    masker = fit_masker(run1_path, s1_mask)
    joblib.dump(masker, cache_dir / f"{subject}_masker.joblib")

    feature_rows, labels, groups = [], [], []
    for run in range(1, C.N_RUNS_PER_SUBJECT + 1):
        bold_path = resolve_bold_path(subject, run=run, space_name=C.SPACE)
        if bold_path is None:
            continue

        run_events = stim_events[
            (stim_events["subject"] == subject) & (stim_events["run"] == run)
        ].sort_values("onset")
        X_run, y_run = extract_run_features(
            bold_path, run_events, masker, tr_subject, hrf_delay=C.HRF_DELAY, window=C.WINDOW
        )
        if X_run.size > 0:
            feature_rows.append(X_run)
            labels.extend(y_run)
            groups.extend([run] * len(y_run))

    if not feature_rows:
        return None

    X = np.vstack(feature_rows)
    y = np.asarray(labels)
    groups = np.asarray(groups)

    path = cache_dir / f"{subject}_dataset.npz"
    save_subject_samples(path, X, y, groups, tr_subject)
    return path


def build_all_subjects_cache(cache_dir, subjects=None, s1_mask=None):
    """Build (or rebuild) the feature cache for every discovered subject."""
    if subjects is None:
        subjects = discover_subjects()
    if s1_mask is None:
        s1_mask = build_s1_mask()

    built = []
    for subject in subjects:
        print(f"Extracting features for {subject}...")
        path = build_subject_cache(subject, s1_mask, cache_dir)
        if path is None:
            print(f"  Skipped {subject}: missing data")
            continue
        built.append(path)
        print(f"  Saved {path}")
    return built