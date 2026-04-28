# Evaluating Seizure Detection Performance Across Reduced EEG Montages

**Kojima\*, Shi\*, Jaikumar\* et al. — University of Pennsylvania Litt Lab**

This repository contains the code used in our study evaluating the performance of automated seizure detection algorithms across computationally simulated sub-scalp EEG montages. We benchmark three detection algorithms—SPaRCNet, NDD (Neural Dynamic Divergence), and a one-class SVM baseline—on a large retrospective cohort of 466 EMU admissions, simulating the reduced spatial configurations of next-generation implantable devices such as the Epiminder Minder system and UNEEG 24/7 SubQ.

> **Paper:** *Evaluating Seizure Detection Performance Across Representative Reduced EEG Montages* (in preparation)
>
> **Dataset:** Available on reasonable request with a data use agreement per IRB #835008, University of Pennsylvania.

---

## Overview

Chronic sub-scalp EEG monitoring devices record from as few as 2–4 channels, enabling years of continuous outpatient monitoring. A key open question is whether automated seizure detection algorithms maintain accuracy under this drastic spatial reduction. This study provides a large-scale, systematic answer across a heterogeneous clinical cohort.

**Key findings:**
- Seizure detection was largely preserved across reduced montages using the best-performing model (SPaRCNet), with event-wise F1 scores of 0.50–0.59 across sub-scalp configurations vs. 0.59 for the full 10-20 montage.
- The dominant sources of performance variance were the **patient** and **choice of algorithm**, not the specific montage.
- Algorithm performance on the full 10-20 montage correlated with performance on reduced montages, supporting the use of pre-implantation EMU data for patient selection ("montage matching").
- Epilepsy laterality interacted predictably with montage laterality: left-sided montages detected left-sided seizures more effectively, and vice versa.

---

## Repository Structure

```
.
├── run_sparcnet.py             # SPaRCNet pipeline: EDF → probabilities → predictions → metrics → figures
├── run_ndd.py                  # NDD (DynaSD) pipeline: patient-wise training and inference
├── run_svm.py                  # One-class SVM baseline with Youden's J thresholding
│
├── pipeline_functions/         # Shared utilities (see "Required Utility Files" below)
│   ├── utils.py
│   ├── feat_funcs.py
│   └── utils_baseline.py
│
├── SPARCNET/                   # SPaRCNet model (see "External Dependencies" below)
│   ├── DenseNetClassifier.py
│   └── sparcnet_pretrain.pt
│
├── DynaSD-wo_dev/              # DynaSD submodule (see "External Dependencies" below)
│
└── requirements.txt
```

---

## Pipeline

Each script runs four steps sequentially for a given model:

```
Step 1: Inference
        EDF clips → raw model probability scores (one .csv per clip, per montage)

Step 2: Thresholding
        Probability scores → binary predictions
        Threshold optimized via Youden's J statistic (TPR − FPR), globally per montage

Step 3: Metric Calculation
        Binary predictions → event-wise sensitivity, FA/hr, F1, AUROC, AUPRC
        Stratified by epilepsy type, laterality, and seizure location

Step 4: Stats & Figures (optional, --do_plot)
        TableOne summaries + strip/box plots across montages and clinical subgroups
```

---

## Montages Evaluated

| Montage Label | Channels | Device Simulated |
|---|---|---|
| `full` | 16 standard bipolar pairs (10-20 system) | Full scalp EMU |
| `epiminder_2` | C3-P3, C4-P4 | Epiminder Minder (centroparietal bilateral) |
| `uneeg_diag_bilateral_front` | F3-T3, F4-T4 | UNEEG 24/7 SubQ (bilateral frontotemporal) |
| `uneeg_diag_left_front` | F3-T3 | UNEEG 24/7 SubQ (left frontotemporal) |
| `uneeg_diag_right_front` | F4-T4 | UNEEG 24/7 SubQ (right frontotemporal) |
| `uneeg_bilateral_front2` | F7-T3, F8-T4 | Temporal (supplemental) |
| `uneeg_vert_bilateral` | C3-T3, C4-T4 | Centrotemporal (supplemental) |
| `uneeg_diag_bilateral_back` | P3-T3, P4-T4 | Temporoparietal (supplemental) |
| `uneeg_bilateral_back2` | T3-T5, T4-T6 | Posterior temporal (supplemental) |
| *(+ left/right unilateral variants of the above)* | | |

---

## Installation

**Python ≥ 3.9** is recommended. A GPU is strongly recommended for the SPaRCNet and NDD pipelines.

```bash
git clone https://github.com/littlab/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### External Dependencies

These are not included in this repository and must be set up separately.

**1. DynaSD** — used for the NDD pipeline. Clone the `wo_dev` branch into `DynaSD-wo_dev/`:
```bash
git clone --branch wo_dev https://github.com/wojemann/DynaSD.git DynaSD-wo_dev
```

**2. SPaRCNet weights** — download `sparcnet_pretrain.pt` from [Jing et al. 2023](https://doi.org/10.1038/s41591-023-02380-1) and place it in `SPARCNET/`. The `DenseNetClassifier.py` model definition must also be present in that folder.

### Required Utility Files

The following files must be present in `pipeline_functions/`. They are not included in this repository as they contain lab-internal preprocessing code:

| File | Used By | What It Provides |
|---|---|---|
| `utils.py` | `run_sparcnet.py`, `run_svm.py` | `Preprocessor` class (bandpass filter, bipolar re-referencing); `get_event_smoothed_pred`, `smooth_pred` (prediction post-processing) |
| `feat_funcs.py` | `run_sparcnet.py` | `bandpass_filter`, `downsample` (signal preprocessing for SPaRCNet input windows) |
| `utils_baseline.py` | `run_svm.py` | `extract_features`, `train_one_class_svm`, `compute_novelty_scores`, `estimate_outlier_fraction`, `detect_seizure`, `apply_persistence` (SVM feature extraction and novelty detection logic) |

> **Note for reproducibility:** The `Preprocessor` class handles bandpass filtering (1–40 Hz), 60 Hz notch filtering, and bipolar re-referencing to the standard 10-20 montage. It is the central preprocessing object shared across all three pipelines and should be applied before any montage subselection step.

---

## Usage

### SPaRCNet pipeline

```bash
python run_sparcnet.py \
    --data_folder /path/to/edf_data \
    --output_folder /path/to/output \
    --patient_info /path/to/dataset_admission_info.csv \
    --montage all \
    --setting optimal \
    --n_jobs 10 \
    --do_plot
```

### NDD pipeline

```bash
python run_ndd.py \
    --data_folder /path/to/edf_data \
    --output_folder /path/to/ndd_output \
    --patient_info /path/to/dataset_admission_info.csv \
    --montage all \
    --n_jobs 10
```

### SVM baseline

```bash
python run_svm.py \
    --data_folder /path/to/edf_data \
    --output_folder /path/to/svm_output \
    --patient_info /path/to/dataset_admission_info.csv \
    --montage all \
    --setting optimal \
    --n_jobs 10
```

**Key arguments (shared across all scripts):**

| Argument | Description | Default |
|---|---|---|
| `--data_folder` | Root folder with `seizure/` and `interictal/` EDF subfolders | required |
| `--output_folder` | Output directory for probabilities, predictions, metrics, and figures | required |
| `--patient_info` | CSV with columns: `admission_id`, `patient_id`, `epilepsy_type`, `laterality`, `location` | required |
| `--montage` | Comma-separated montage keys, or `all` | `all` |
| `--setting` | Thresholding: `optimal` (Youden's J) or leave blank for fixed | `optimal` |
| `--thres` | Fixed threshold value (used when `--setting` is not `optimal`) | `0.5` |
| `--n_jobs` | Number of parallel workers | `10` |
| `--force` | Recompute and overwrite existing output files | `False` |
| `--do_plot` | Generate figures and TableOne stats tables | `False` |

---

## Input Data Format

EDF files should be organized as:
```
data/
  seizure/
    {admission_id}_seizure_{index}.edf
    ...
  interictal/
    {admission_id}_interictal_{index}.edf
    ...
```

Seizure onset and offset are read from MNE-compatible EDF annotations. The `patient_info` CSV maps admission IDs to patient-level clinical metadata (epilepsy type, laterality, seizure localization).

---

## Performance Evaluation

Detection is evaluated using the [SzCORE framework](https://github.com/esl-epfl/szcore) with the following primary metrics:

- **Event-wise sensitivity** — fraction of true seizure events overlapping any prediction (±30s pre-ictal, ±60s post-ictal tolerance)
- **False alarm rate (FA/hr)** — predicted events with no overlap with any true seizure, normalized by interictal recording duration
- **Event-wise F1** — harmonic mean of event-wise sensitivity and precision
- **AUROC / AUPRC** — threshold-independent performance

Decision thresholds are optimized per montage via **Youden's J statistic** (TPR − FPR) pooled across all patients. Binary predictions are post-processed with morphological closing (10s kernel) and opening (20s kernel); events shorter than 20s are discarded.

---

## Study Cohort

466 EMU admissions from 436 unique patients, Hospital of the University of Pennsylvania (HUP), January 2017 – December 2024.

| Characteristic | Value |
|---|---|
| Mean age at first admission | 39.0 years (SD 14.4) |
| Female | 54.4% |
| Median seizures per admission | 2.0 [IQR 1–4] |
| Mean seizure duration | 81.4 s (SD 70.8) |
| Total seizure events | 1,683 |
| Total interictal clips | 1,527 |
| Focal epilepsy | 75.9% |
| Temporal localization (focal/mixed) | 67.0% |

IRB #835008, University of Pennsylvania (approved 2/3/2020).

---

## Citation

If you use this code or findings, please cite:

```bibtex
@article{kojima2025montage,
  title   = {Evaluating Seizure Detection Performance Across Representative Reduced EEG Montages},
  author  = {Kojima, Joe and Shi, Haoer and Jaikumar, Svanik and Ojemann, William K.S. and
             Kim, Juri and Ganguly, Mindy and Litt, Brian and Conrad, Erin},
  journal = {(in preparation)},
  year    = {2025}
}
```

Please also cite the models used:

- **SPaRCNet:** Jing et al. (2023). *Nature Medicine.* https://doi.org/10.1038/s41591-023-02380-1
- **DynaSD / NDD:** Wojemann et al. https://github.com/wojemann/DynaSD
- **SzCORE:** Dan et al. (2025)

---

## Contact

Code correspondence: **haoershi@seas.upenn.edu**

For raw data access, contact the corresponding author with a data use agreement per IRB requirements.

---

## License

MIT License. See `LICENSE` for details.

Note: SPaRCNet model weights and DynaSD are subject to their own respective licenses.
