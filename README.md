# Somatosensory Mapping Analysis

Decoding upper limb representations in the brain using fMRI and nilearn.

## Quick Start

### 1. Create Conda Environment

```powershell
conda env create -f environment.yml
conda activate somatosensory_mapping
```

### 2. Launch Jupyter Lab

```powershell
jupyter lab
```

### 3. Run Analysis

Open `notebooks/01_somatosensory_decoding.ipynb` and run all cells.

## What It Does

Performs MVPA decoding to identify which upper limb area (E1-E20) was stimulated based on brain activation patterns.

- Uses event files from `data/BIDS-somatosensory/sub-p0001/ses-01/func/`
- Uses nilearn's Decoder with SVM classifier
- Leave-one-run-out cross-validation
- Visualizes important brain regions (somatosensory & motor cortex)