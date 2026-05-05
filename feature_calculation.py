import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import scipy.signal
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
from mne.filter import filter_data

warnings.filterwarnings("ignore")

# --- Path Configuration ---
working_dir = os.path.abspath("")
sparcnet_path = os.path.join(working_dir, "SPARCNET")
funcs_path = os.path.join(working_dir, "pipeline_functions")

sys.path.append(working_dir)
sys.path.append(funcs_path)
sys.path.append(sparcnet_path)

try:
    from utils import (
        load_edf_file,
    )
except ImportError as e:
    print("Error: Could not import dependency. Make sure files exist.")
    print(f"Missing: {e}")

# --- CONFIGURATION ---
DATA_DIR = "../emu_dataset"
METRICS_FILE = "sparcnet_results/metrics/thres_optimal_f1/full/segment_metrics.csv"
OUTPUT_DIR = "feature_data"
WINDOW_SEC = 5  # 5-second non-overlapping windows

STANDARD_BIPOLAR = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "F3-T3",
    "F4-T4",
]


# --- HELPER FUNCTIONS ---
def clean_array(arr):
    if not np.all(np.isfinite(arr)):
        arr = np.where(np.isposinf(arr), np.finfo(arr.dtype).max / 2, arr)
        arr = np.where(np.isneginf(arr), np.finfo(arr.dtype).min / 2, arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def create_bipolar_data(df):
    data = pd.DataFrame(index=df.index)

    def find_col(ch_name):
        if ch_name in df.columns:
            return ch_name
        for c in df.columns:
            c_clean = (
                c.upper()
                .replace("EEG", "")
                .replace("-REF", "")
                .replace("-LE", "")
                .strip()
            )
            if c_clean == ch_name.upper():
                return c
        return None

    for p in STANDARD_BIPOLAR:
        ch1, ch2 = p.split("-")
        col1, col2 = find_col(ch1), find_col(ch2)
        if col1 is not None and col2 is not None:
            data[p] = df[col1] - df[col2]
    return data


def compute_bandpower(data, fs, band):
    freqs, psd = scipy.signal.welch(data, fs, nperseg=min(int(fs * 2), data.shape[-1]))
    band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[:, band_idx], axis=1)


def extract_window_features(clip, fs):
    features = {}
    features["variance"] = np.var(clip, axis=1)
    features["amplitude_abs_mean"] = np.mean(np.abs(clip), axis=1)
    analytic_signal = scipy.signal.hilbert(clip, axis=1)
    features["amplitude_env_mean"] = np.mean(np.abs(analytic_signal), axis=1)
    features["line_length"] = np.mean(np.abs(np.diff(clip, axis=1)), axis=1)

    features["delta_pow"] = compute_bandpower(clip, fs, [1, 4])
    features["theta_pow"] = compute_bandpower(clip, fs, [4, 8])
    features["alpha_pow"] = compute_bandpower(clip, fs, [8, 13])
    features["sigma_pow"] = compute_bandpower(clip, fs, [12, 16])
    features["beta_pow"] = compute_bandpower(clip, fs, [13, 30])
    features["gamma_pow"] = compute_bandpower(clip, fs, [30, 40])
    return features


# --- PARALLEL WORKER FUNCTION ---
def process_single_file(file_path):
    base_name = os.path.basename(file_path)
    event_id = base_name.replace(".edf", "")

    records = []

    try:
        raw, df, label_df, fs = load_edf_file(file_path)
        bipolar_df = create_bipolar_data(df)
        channel_names = bipolar_df.columns.tolist()

        eeg_data = bipolar_df.values.T
        eeg_data = clean_array(eeg_data)
        eeg_data = filter_data(eeg_data, fs, 1.0, 40.0, method="iir", verbose=False)
        labels = label_df["labels"].values

        win_samples = int(WINDOW_SEC * fs)
        n_windows = eeg_data.shape[1] // win_samples

        for w in range(n_windows):
            start_idx = w * win_samples
            end_idx = start_idx + win_samples

            clip = eeg_data[:, start_idx:end_idx]
            win_labels = labels[start_idx:end_idx]

            label_val = 1 if np.any(win_labels == 1) else 0

            feats = extract_window_features(clip, fs)

            for ch_idx, ch_name in enumerate(channel_names):
                row = {
                    "event_id": event_id,
                    "window_idx": w,
                    "time_start_sec": w * WINDOW_SEC,
                    "channel": ch_name,
                    "label": label_val,
                }
                for f_name, f_array in feats.items():
                    row[f_name] = f_array[ch_idx]
                records.append(row)

    except Exception:
        return []

    return records


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_jobs", type=int, default=20, help="Number of parallel workers"
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        all_files = glob.glob(f"{DATA_DIR}/**/*.edf", recursive=True)
        if not all_files:
            print(f"Warning: No .edf files found in {DATA_DIR}")
    except Exception as e:
        print(f"Error finding EDF files: {e}")
        all_files = []

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_single_file)(f) for f in tqdm(all_files, desc="Processing EDFs")
    )

    flat_records = [item for sublist in results for item in sublist]
    detailed_df = pd.DataFrame(flat_records)

    feature_cols = [
        "variance",
        "amplitude_abs_mean",
        "amplitude_env_mean",
        "line_length",
        "delta_pow",
        "theta_pow",
        "alpha_pow",
        "sigma_pow",
        "beta_pow",
        "gamma_pow",
    ]

    # Save 1: The original, massive file (per-window, per-channel)
    detailed_csv_path = os.path.join(OUTPUT_DIR, "detailed_windowed_features.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(
        f"   -> Saved massive windowed features to {detailed_csv_path} ({len(detailed_df)} rows)."
    )
