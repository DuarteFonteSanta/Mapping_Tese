"""Per-subject event-file loading and electrode-to-region/label assignment."""
import pandas as pd
from . import constants as C

def load_subject_events(subject):
    all_events = []
    required_cols = {"onset", "duration", "trial_type"}

    for run in range(1, C.N_RUNS_PER_SUBJECT + 1):
        ev_path = C.EVENTS_DIR / f"{subject}_{C.SESSION}_{C.TASK}_run-{run}_events.tsv"
        if not ev_path.exists():
            raise FileNotFoundError(f"Missing events file: {ev_path}")

        df = pd.read_csv(ev_path, sep="\t")
        df.columns = [str(c).strip().lower() for c in df.columns]

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{ev_path.name} missing columns {sorted(missing)}")

        df["subject"] = subject
        df["run"] = run
        all_events.append(df)

    events_df = pd.concat(all_events, ignore_index=True)
    stim_events = events_df[~events_df["trial_type"].isin(["Baseline", "Jitter"])].copy()
    stim_events["region"] = stim_events["trial_type"].map(C.ELECTRODE_TO_REGION)

    unmapped = stim_events["region"].isna().sum()
    if unmapped > 0:
        bad = sorted(stim_events.loc[stim_events["region"].isna(), "trial_type"].unique())
        raise ValueError(f"Unmapped trial_type values for {subject}: {bad}")

    stim_events["label"] = stim_events["region"].map(C.REGION_TO_LABEL).astype(int)
    return events_df, stim_events