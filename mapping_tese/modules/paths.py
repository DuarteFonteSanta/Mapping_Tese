"""BOLD path resolution, TR lookup, and subject discovery."""
import json
import re
from . import constants as C

def resolve_bold_path(subject, run, space_name):
    func_dir = C.DERIVATIVES / subject / C.SESSION / "func"
    candidates = [
        func_dir / f"{subject}_{C.SESSION}_{C.TASK}_run-{run}_space-{space_name}_desc-preproc_bold.nii.gz",
        func_dir / f"{subject}_{C.SESSION}_{C.TASK}_run-{run}_space-{space_name}_desc-preproc_bold.nii",
        func_dir / f"{subject}_{C.SESSION}_{C.TASK}_run-{run}_desc-preproc_bold.nii.gz",
        func_dir / f"{subject}_{C.SESSION}_{C.TASK}_run-{run}_desc-preproc_bold.nii",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def get_tr(subject, run=1, default_tr=2.0):
    func_dir = C.DERIVATIVES / subject / C.SESSION / "func"
    json_candidates = [
        func_dir / f"{subject}_{C.SESSION}_{C.TASK}_run-{run}_space-{C.SPACE}_desc-preproc_bold.json",
        func_dir / f"{subject}_{C.SESSION}_{C.TASK}_run-{run}_desc-preproc_bold.json",
    ]
    for p in json_candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("RepetitionTime") is not None:
                return float(meta["RepetitionTime"])
    return float(default_tr)


def discover_subjects():
    derivative_subjects = sorted(
        d.name for d in C.DERIVATIVES.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    event_subjects = set()
    for ev_path in C.EVENTS_DIR.glob(f"sub-*_{C.SESSION}_{C.TASK}_run-*_events.tsv"):
        m = re.match(r"^(sub-[^_]+)_", ev_path.name)
        if m:
            event_subjects.add(m.group(1))
    return sorted(set(derivative_subjects) & set(event_subjects))