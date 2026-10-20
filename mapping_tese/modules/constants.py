"""ROI region/electrode/label mappings and shared BIDS experiment parameters."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BIDS_ROOT = PROJECT_ROOT / "data" / "BIDS-somatosensory" / "BIDS-somatosensory"
DERIVATIVES = BIDS_ROOT / "derivatives" / "fmriprep"
EVENTS_DIR = BIDS_ROOT / "events"

SESSION = "ses-01"
TASK = "task-S1Map"
SPACE = "MNI152NLin2009cAsym"
N_RUNS_PER_SUBJECT = 4

HRF_DELAY = 6.0
WINDOW = 1

REGION_TO_ELECTRODES = {
    "Middle_Finger": ["E1", "E2", "E3"],
    "Hand": ["E4", "E5", "E6", "E7"],
    "Forearm": ["E8", "E9", "E10", "E11", "E12", "E13"],
    "Arm": ["E14", "E15", "E16", "E17", "E18", "E19", "E20"],
}
ELECTRODE_TO_REGION = {
    elec: region
    for region, elecs in REGION_TO_ELECTRODES.items()
    for elec in elecs
}
REGION_TO_LABEL = {"Middle_Finger": 0, "Hand": 1, "Forearm": 2, "Arm": 3}
LABEL_TO_REGION = {v: k for k, v in REGION_TO_LABEL.items()}