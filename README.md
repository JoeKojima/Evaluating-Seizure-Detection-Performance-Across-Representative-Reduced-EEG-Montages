# Evaluating Seizure Detection Performance Across Reduced EEG Montages

**Kojima\*, Shi\*, Jaikumar\* et al. — University of Pennsylvania Litt Lab**

This repository contains the code used in our study evaluating the performance of automated seizure detection algorithms across computationally simulated sub-scalp EEG montages. We benchmark three detection algorithms—SPaRCNet, NDD (Neural Dynamic Divergence), and a one-class SVM baseline—on a large retrospective cohort of EMU admissions, simulating the reduced spatial configurations of next-generation implantable devices such as the Epiminder Minder system and UNEEG 24/7 SubQ.

> **Paper:** *Evaluating Seizure Detection Performance Across Representative Reduced EEG Montages* (in preparation)
> 
> **Dataset:** Available on reasonable request with a data use agreement per IRB #835008, University of Pennsylvania.

---

## Overview

Chronic sub-scalp EEG monitoring devices record from as few as 2–4 channels, enabling years of continuous outpatient monitoring. A key open question is whether automated seizure detection algorithms can maintain accuracy under this drastic spatial reduction. This study provides a large-scale, systematic answer.

**Key findings:**
- Seizure detection was largely preserved across reduced montages using the best-performing model (SPaRCNet), with event-wise F1 scores ranging from 0.50–0.59 across sub-scalp configurations versus 0.59 for the full montage.
- The dominant sources of variance were the **patient** and **choice of algorithm**, not the montage.
- Algorithm performance on the full 10-20 montage was correlated with performance on reduced montages, supporting the use of pre-implantation EMU data for patient selection.
- Epilepsy laterality interacted predictably with montage laterality: left-sided montages detected left-sided seizures more effectively, and vice versa.

---

## Repository Structure

```
.
├── run_full_pipeline.py      # SPaRCNet + NDD end-to-end pipeline (Steps 1–4)
├── svm_youdens.py            # One-class SVM baseline with Youden's J thresholding
├── dynasd_fixed.py           # NDD (DynaSD) patient-wise inference pipeline
│
├── pipeline_functions/       # Shared utilities (must be provided separately)
│   ├── utils.py              # Preprocessor, smoothing helpers
│   ├── feat_funcs.py         # Signal filtering and downsampling
│   └── utils_baseline.py     # SVM feature extraction and detection logic
│
├── SPARCNET/                 # SPaRCNet model files (must be provided separately)
│   ├── DenseNetClassifier.py
│   └── sparcnet_pretrain.pt  # Pre-trained weights (see below)
│
└── DynaSD-wo_dev/            # DynaSD submodule (see below)
```

---

## Pipeline

The evaluation pipeline consists of four steps, run sequentially:

```
Step 1: Inference
         EDF files → model probability scores (.csv per clip)

Step 2: Thresholding
         Probability scores → binary predictions
         (Youden's J statistic optimized globally per montage)

Step 3: Metric Calculation
         Binary predictions → event-wise sensitivity, FA/hr, F1, AUROC, AUPRC
         (using the SzCORE framework, stratified by patient clinical info)

Step 4: Stats & Figures
         TableOne summaries + boxplots across montages and clinical subgroups
```

---

## Montages Evaluated

| Montage Label | Channels | Device Simulated |
|---|---|---|
| `full` | 16 standard bipolar pairs (10-20) | Full scalp EMU |
| `epiminder_2` | C3-P3, C4-P4 | Epiminder Minder (centroparietal) |
| `uneeg_diag_bilateral_front` | F3-T3, F4-T4 | UNEEG 24/7 SubQ (bilateral frontotemporal) |
| `uneeg_diag_left_front` | F3-T3 | UNEEG 24/7 SubQ (left frontotemporal) |
| `uneeg_diag_right_front` | F4-T4 | UNEEG 24/7 SubQ (right frontotemporal) |
| `uneeg_bilateral_front2` | F7-T3, F8-T4 | Temporal supplemental |
| `uneeg_vert_bilateral` | C3-T3, C4-T4 | Centrotemporal supplemental |
| `uneeg_diag_bilateral_back` | P3-T3, P4-T4 | Temporoparietal supplemental |
| `uneeg_bilateral_back2` | T3-T5, T4-T6 | Posterior temporal supplemental |
| *(+ left/right unilateral variants of each)* | | |

---

## Installation

**Python ≥ 3.9** is recommended.

```bash
git clone https://github.com/littlab/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Required packages:**
```
numpy
pandas
scipy
mne
scikit-learn
torch
joblib
tqdm
matplotlib
seaborn
tableone
```

**External dependencies (not included):**

1. **DynaSD** — Clone the DynaSD repository into `DynaSD-wo_dev/`:
   ```bash
   git clone https://github.com/<dynasd-repo> DynaSD-wo_dev
   ```

2. **SPaRCNet weights** — Download `sparcnet_pretrain.pt` from [Jing et al. 2023](https://doi.org/10.1038/s41591-023-02380-1) and place it in `SPARCNET/`.

---

## Usage

### Full pipeline (SPaRCNet + NDD)

```bash
python run_full_pipeline.py \
    --data_folder /path/to/edf_data \
    --output_folder /path/to/output \
    --patient_info /path/to/dataset_admission_info.csv \
    --montage all \
    --n_jobs 10 \
    --do_plot
```

### SVM baseline

```bash
python svm_youdens.py \
    --data_folder /path/to/edf_data \
    --output_folder /path/to/output \
    --patient_info /path/to/dataset_admission_info.csv \
    --montage all \
    --setting optimal \
    --n_jobs 10
```

### NDD only

```bash
python dynasd_fixed.py \
    --data_folder /path/to/edf_data \
    --output_folder /path/to/ndd_output \
    --patient_info /path/to/dataset_admission_info.csv \
    --montage all \
    --n_jobs 10
```

**Key arguments:**

| Argument | Description | Default |
|---|---|---|
| `--data_folder` | Root folder containing `seizure/` and `interictal/` EDF subfolders | required |
| `--output_folder` | Output directory for probabilities, predictions, metrics, and figures | required |
| `--patient_info` | CSV with columns: `admission_id`, `patient_id`, `epilepsy_type`, `laterality`, `location` | required |
| `--montage` | Comma-separated montage keys, or `all` | `all` |
| `--setting` | Thresholding strategy: `optimal` (Youden's J) or `fixed` | `optimal` |
| `--n_jobs` | Parallel workers | `10` |
| `--force` | Overwrite existing output files | `False` |
| `--do_plot` | Generate figures and TableOne stats | `False` |

---

## Input Data Format

EDF files should be named `{admission_id}_{clip_type}_{index}.edf`, e.g.:
```
data/
  seizure/
    EMU0001_seizure_001.edf
    EMU0001_seizure_002.edf
  interictal/
    EMU0001_interictal_001.edf
```

EDF annotations should mark seizure onset and offset using MNE-compatible annotation format. The `patient_info` CSV maps admission IDs to patient-level clinical metadata.

---

## Performance Evaluation

Detection performance is evaluated using the [SzCORE framework](https://github.com/esl-epfl/szcore) with the following primary metrics:

- **Event-wise sensitivity** — fraction of true seizure events with any overlapping prediction (±30s pre-ictal, ±60s post-ictal tolerance)
- **False alarm rate (FA/hr)** — detected events with no overlap with any true seizure, normalized by recording duration
- **Event-wise F1 score** — harmonic mean of event-wise sensitivity and precision
- **AUROC / AUPRC** — threshold-independent performance

Decision thresholds are optimized separately per montage using **Youden's J statistic** (TPR − FPR) computed across all patients. Binary predictions are post-processed with morphological closing (10s) and opening (20s) operations, and events shorter than 20s are discarded.

---

## Cohort

466 EMU admissions from 436 unique patients, Hospital of the University of Pennsylvania (HUP), 2017–2024.

| Characteristic | Value |
|---|---|
| Mean age at admission | 39.0 years (SD 14.4) |
| Female | 54.4% |
| Median seizures per admission | 2.0 [IQR 1–4] |
| Mean seizure duration | 81.4s (SD 70.8) |
| Total seizure events | 1,683 |
| Total interictal clips | 1,527 |
| Focal epilepsy | 75.9% |
| Temporal localization (focal/mixed) | 67.0% |

IRB #835008, University of Pennsylvania (approved 2/3/2020).

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{kojima2025montage,
  title   = {Evaluating Seizure Detection Performance Across Representative Reduced EEG Montages},
  author  = {Kojima, Joe and Shi, Haoer and Jaikumar, Svanik and Ojemann, William K.S. and Kim, Juri and Ganguly, Mindy and Litt, Brian and Conrad, Erin},
  journal = {(in preparation)},
  year    = {2025}
}
```

Please also cite the following if you use the respective models:

- **SPaRCNet:** Jing et al. (2023). *Nat. Med.* https://doi.org/10.1038/s41591-023-02380-1
- **NDD/DynaSD:** *(cite DynaSD repository)*
- **SzCORE:** Dan et al. (2025)

---

## Contact

Questions about the code: **haoershi@seas.upenn.edu**

For data access requests, please contact the corresponding author with a data use agreement per IRB requirements.

---

## License

MIT License. See `LICENSE` for details. Note that SPaRCNet model weights and DynaSD are subject to their own licenses.
