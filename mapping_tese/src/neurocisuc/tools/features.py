import numpy as np
from nilearn.image import load_img, index_img, mean_img, resample_to_img
from nilearn.maskers import NiftiMasker



# Fit a NiftiMasker to the reference image and resampled mask.
def fit_masker(reference_img_path, mask_img):
    """Resamples the mask to the reference image and initializes the NiftiMasker."""
    ref_img = load_img(str(reference_img_path))
    first_vol = index_img(ref_img, 0)
    
    mask_resampled = resample_to_img(mask_img, first_vol, interpolation="nearest")
    masker = NiftiMasker(mask_img=mask_resampled, standardize=None)
    masker.fit(ref_img)
    return masker


# Extracts window-averaged voxel features for a single BOLD run.
def extract_run_features(func_path, run_events, masker, tr, hrf_delay=6.0, window=1):
    """Extracts window-averaged voxel features for a single BOLD run."""
    img = load_img(str(func_path))
    run_length = img.shape[3]
    
    features, labels = [], []

    # Iterate over each event in the run_events DataFrame to extract features.
    for _, ev in run_events.iterrows():
        peak_vol = int(np.round((float(ev["onset"]) + hrf_delay) / tr))
        
        if peak_vol >= run_length:
            continue

        # Define the range of volumes to average around the peak volume, ensuring we stay within bounds.    
        vols = list(range(max(0, peak_vol - window), min(run_length, peak_vol + window + 1)))

        # Average the selected volumes and extract features using the masker.
        averaged = mean_img(index_img(img, vols), copy_header=True)

        # Apply the masker to the averaged image and store the features and corresponding label.
        features.append(masker.transform(averaged).ravel())
        labels.append(str(ev["region"]))
        
    if not features:
        return np.array([]), np.array([])
        
    return np.vstack(features), np.asarray(labels)



# Extracts window-averaged voxel features for a single BOLD run.
def extract_run_multiple_features(func_path, run_events, masker, tr, hrf_delay=6.0, window=3):
    """Extracts 3 voxel features for a single BOLD run."""
    img = load_img(str(func_path))
    run_length = img.shape[3]
    
    features, labels = [], []

    # Iterate over each event in the run_events DataFrame to extract features.
    for _, ev in run_events.iterrows():
        peak_vol = int(np.round((float(ev["onset"]) + hrf_delay) / tr))
        
        if peak_vol >= run_length:
            continue

        # Define the range of volumes around the peak volume, ensuring we stay within bounds.    
        vols = list(range(max(0, peak_vol - window), min(run_length, peak_vol + window + 1)))

        print(f"Extracting features for event {ev['region']} at peak volume {peak_vol} with volumes {vols}")

        # Extract features for each volume in the defined range.
        for vol in vols:
            averaged = index_img(img, vol)
            features.append(masker.transform(averaged).ravel())
            labels.append(str(ev["region"]))


        
    if not features:
        return np.array([]), np.array([])
        
    return np.vstack(features), np.asarray(labels)