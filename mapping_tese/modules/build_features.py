"""CLI entry point to (re)build the per-subject feature caches for each method."""
import argparse

import joblib

from mapping_tese.modules import constants as C
from mapping_tese.modules.cache import EXTRACTORS, get_features
from mapping_tese.modules.paths import discover_subjects, get_tr, resolve_bold_path
from mapping_tese.modules.roi import build_s1_mask, fit_masker

SCHEME = "s1_hrf6_win1"
CACHE_DIR = C.PROJECT_ROOT / "data" / "extracted" / SCHEME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(EXTRACTORS), choices=list(EXTRACTORS))
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--force", action="store_true", help="recompute even if the cache file exists")
    args = parser.parse_args()

    subjects = args.subjects or discover_subjects()
    s1_mask = build_s1_mask()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        run1_path = resolve_bold_path(subject, run=1, space_name=C.SPACE)
        if run1_path is None:
            print(f"Skipped {subject}: missing run-1 bold")
            continue

        joblib.dump(fit_masker(run1_path, s1_mask), CACHE_DIR / f"{subject}_masker.joblib")
        tr_subject = get_tr(subject, run=1, default_tr=2.0)

        for method in args.methods:
            try:
                X, y, groups, _ = get_features(
                    subject, tr_subject, s1_mask, CACHE_DIR, method=method, force_recompute=args.force
                )
            except FileNotFoundError as e:
                print(f"Skipped {subject} [{method}]: {e}")
                continue
            print(f"{subject} [{method}]: X={X.shape}")

if __name__ == "__main__":
    main()