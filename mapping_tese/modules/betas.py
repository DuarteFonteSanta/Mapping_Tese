"""Single-trial beta estimates (least-squares-separate GLM) on the S1 ROI."""
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import load_img

from . import constants as C
from .events import load_subject_events
from .paths import resolve_bold_path
from .roi import fit_masker

CONFOUND_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter"]
HIGH_PASS_HZ = 1 / 128
NON_STIM_TYPES = ["Baseline", "Jitter"]


def load_run_confounds(subject, run):
    func_dir = C.DERIVATIVES / subject / C.SESSION / "func"
    path = func_dir / f"{subject}_{C.SESSION}_{C.TASK}_run-{run}_desc-confounds_timeseries.tsv"
    return pd.read_csv(path, sep="\t", usecols=CONFOUND_COLS).fillna(0.0).to_numpy()


def run_single_trial_betas(Y, run_events, confounds, tr):
    """Return one beta per stimulus trial (rows in onset order) for every voxel."""
    if confounds.shape[0] != Y.shape[0]:
        raise ValueError(f"Confounds rows {confounds.shape[0]} != volumes {Y.shape[0]}")

    frame_times = np.arange(Y.shape[0]) * tr
    events = run_events[["onset", "duration", "trial_type"]].reset_index(drop=True)
    is_stim = ~events["trial_type"].isin(NON_STIM_TYPES)
    stim_rows = np.flatnonzero(is_stim.to_numpy())

    betas = np.empty((len(stim_rows), Y.shape[1]), dtype=np.float32)
    for k, row in enumerate(stim_rows):
        ev = events.copy()
        ev.loc[is_stim, "trial_type"] = "other_stim"
        ev.loc[row, "trial_type"] = "target"
        design = make_first_level_design_matrix(
            frame_times,
            ev,
            hrf_model="spm",
            drift_model="cosine",
            high_pass=HIGH_PASS_HZ,
            add_regs=confounds,
            add_reg_names=CONFOUND_COLS,
        )
        coef, *_ = np.linalg.lstsq(design.to_numpy(), Y, rcond=None)
        betas[k] = coef[design.columns.get_loc("target")]
    return betas


def extract_subject_betas(subject, tr_subject, s1_mask, expected_n_voxels=None):
    """Same return signature as extract_subject_samples: X (trials x voxels), y, run groups, tr."""
    all_events, stim_events = load_subject_events(subject)

    run1_path = resolve_bold_path(subject, run=1, space_name=C.SPACE)
    if run1_path is None:
        raise FileNotFoundError(f"Missing run-1 bold for {subject}")
    masker = fit_masker(run1_path, s1_mask)

    X_list, y_list, run_list = [], [], []
    for run in range(1, C.N_RUNS_PER_SUBJECT + 1):
        bold_path = resolve_bold_path(subject, run=run, space_name=C.SPACE)
        if bold_path is None:
            continue

        run_events = all_events[all_events["run"] == run].sort_values("onset")
        Y = masker.transform(load_img(str(bold_path)))
        if expected_n_voxels is not None and Y.shape[1] != expected_n_voxels:
            raise ValueError(f"Voxel mismatch for {subject}, run {run}: {Y.shape[1]} vs {expected_n_voxels}")

        betas = run_single_trial_betas(Y, run_events, load_run_confounds(subject, run), tr_subject)
        labels = stim_events.loc[stim_events["run"] == run].sort_values("onset")["label"].to_numpy()
        if len(labels) != betas.shape[0]:
            raise ValueError(f"Trial count mismatch for {subject}, run {run}: {len(labels)} vs {betas.shape[0]}")

        X_list.append(betas)
        y_list.append(labels)
        run_list.append(np.full(len(labels), run, dtype=np.int64))

    if not X_list:
        raise RuntimeError(f"No usable runs for {subject}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list).astype(np.int64)
    groups = np.concatenate(run_list)
    return X, y, groups, tr_subject