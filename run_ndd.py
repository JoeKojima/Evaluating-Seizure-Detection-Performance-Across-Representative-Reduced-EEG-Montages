# Functions and imports
import argparse
import sys
import os
import warnings
import glob
import random
import gc

import mne
import numpy as np
import pandas as pd
import scipy.signal
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from mne.filter import filter_data

# --- Setup Environment ---
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# --- 0. DYNASD SETUP ---
working_dir = os.path.abspath("")
ndd_path = os.path.join(working_dir, "DynaSD-wo_dev")
funcs_path = os.path.join(working_dir, "pipeline_functions")

sys.path.append(working_dir)
sys.path.append(funcs_path)
sys.path.append(ndd_path)

# --- DYNASD IMPORTS ---
try:
    from DynaSD import NDD
    from DynaSD.utils import ar_one

    print("Successfully imported DynaSD components", flush=True)
except ImportError as e:
    print(
        f"FATAL ERROR: Could not import DynaSD. Ensure 'DynaSD-wo_dev' is in path. Error: {e}",
        flush=True,
    )
    sys.exit(1)

# --- Import User Dependencies ---
# These files must exist in your 'pipeline_functions' and 'DynaSD-wo_dev' folders
try:
    from feat_funcs import get_event_smoothed_pred, smooth_pred
    from get_metrics import (
        calculate_metrics_for_montages,
        generate_stats_tables,
        get_optimal_thres,
    )
    from utils import Preprocessor, load_edf_file
except ImportError as e:
    print(f"FATAL ERROR: Could not import required pipeline components. Missing: {e}")
    sys.exit(1)

# --- 3. CONSTANTS ---

# Model Settings (Updated to match run_full_pipeline.py)
FS_NDD = 256
W_SIZE_SEC = 1
W_STRIDE_SEC = 0.5
TRAIN_MIN = 1
TRAIN_DURATION_SEC = 60 * TRAIN_MIN

feat_setting_ndd = {
    "win": W_SIZE_SEC,
    "stride": W_STRIDE_SEC,
    "reref": "BIPOLAR",
    "lowcut": 1,
    "highcut": 40,
}

montage_dict = {
    "full": [
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
    ],
    "uneeg_left_front": ["F7-T3"],
    "uneeg_left_back": ["T3-T5"],
    "uneeg_right_front": ["F8-T4"],
    "uneeg_right_back": ["T4-T6"],
    "uneeg_bilateral_back2": ["T3-T5", "T4-T6"],
    "uneeg_bilateral_front2": ["F7-T3", "F8-T4"],
    "uneeg_vert_left": lambda df: custom_bipolar(df, ["C3-T3"]),
    "uneeg_vert_right": lambda df: custom_bipolar(df, ["C4-T4"]),
    "uneeg_diag_left_front": lambda df: custom_bipolar(df, ["F3-T3"]),
    "uneeg_diag_left_back": lambda df: custom_bipolar(df, ["P3-T3"]),
    "uneeg_diag_right_front": lambda df: custom_bipolar(df, ["F4-T4"]),
    "uneeg_diag_right_back": lambda df: custom_bipolar(df, ["P4-T4"]),
    "uneeg_diag_bilateral_front": lambda df: custom_bipolar(df, ["F3-T3", "F4-T4"]),
    "uneeg_diag_bilateral_back": lambda df: custom_bipolar(df, ["P3-T3", "P4-T4"]),
    "uneeg_vert_bilateral": lambda df: custom_bipolar(df, ["C3-T3", "C4-T4"]),
    "epiminder_2": ["C3-P3", "C4-P4"],
    "ceribell": [
        "Fp1-F7",
        "F7-T3",
        "T3-T5",
        "T5-O1",
        "Fp2-F8",
        "F8-T4",
        "T4-T6",
        "T6-O2",
    ],
}


# --- Helper Functions ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


set_seed(5210)


def custom_bipolar(df, pairs):
    filtered = df["filtered"]
    # FIX: Use filtered.index, since df is a dictionary here, not a DataFrame
    data = pd.DataFrame(index=filtered.index)

    # Helper to safely find EDF column names despite prefixes/suffixes
    def find_col(ch_name):
        if ch_name in filtered.columns:
            return ch_name

        for c in filtered.columns:
            # Strip common EDF cruft to find the base channel name
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

    for p in pairs:
        ch1, ch2 = p.split("-")
        col1 = find_col(ch1)
        col2 = find_col(ch2)

        if col1 is not None and col2 is not None:
            data[p] = filtered[col1] - filtered[col2]
        else:
            pass  # Channel truly missing from this patient's recording

    return data


# --- 4. DATA HANDLING HELPERS ---
def _preprocess_edf(df, raw):
    """Run DynaSD Preprocessor on raw EDF frame (same order as run_svm.process_pat)."""
    fs = raw.info["sfreq"]
    prepro = Preprocessor()
    prepro.fit(
        {
            "samplingFreq": fs,
            "samplingFreqRaw": fs,
            "channelNames": df.columns,
            "studyType": "eeg",
            "numberOfChannels": df.shape[1],
        }
    )
    return prepro.preprocess(df)


def _montage_from_preprocessed(preprocessed, montage_key):
    """Select montage channels from an already-preprocessed dict (after preprocess)."""
    montage_processor = montage_dict[montage_key]
    if isinstance(montage_processor, list):
        data_df = preprocessed["BIPOLAR"].copy()
        valid_cols = [c for c in montage_processor if c in data_df.columns]
        data_df = data_df[valid_cols].copy()
    else:
        data_df = montage_processor(preprocessed)
    return data_df


def clean_array(arr, name="array"):
    if not np.all(np.isfinite(arr)):
        arr = np.where(np.isposinf(arr), np.finfo(arr.dtype).max / 2, arr)
        arr = np.where(np.isneginf(arr), np.finfo(arr.dtype).min / 2, arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# --- 5. PREPROCESSING & MODEL HELPERS ---
def preprocess_signal(data_df_montage, fs_raw):
    """Applies 1-40Hz filtering, resamples to 200Hz, and performs AR(1) whitening"""
    valid_channels = data_df_montage.columns.tolist()
    # eeg_data_array = clean_array(data_df_montage[valid_channels].values.T, "raw EEG")
    try:
        data_df_montage.iloc[1:, :] = ar_one(data_df_montage.values)
        return data_df_montage
    except Exception as e:
        print(f"    Preprocessing failed: {e}")
        return None


def process_patient_dataset(
    patient_id, sz_files, ii_files, montage_keys, prob_folder, force
):
    """Like run_svm.process_pat: preprocess each EDF once, then loop montages to sample."""
    warnings.filterwarnings("ignore")
    all_files = sorted(sz_files + ii_files)

    training_candidates = sorted(ii_files)
    if not training_candidates:
        return

    def montage_has_pending(montage_key):
        if force:
            return True
        return any(
            not os.path.exists(
                os.path.join(
                    prob_folder,
                    montage_key,
                    os.path.basename(f).replace(".edf", ".csv"),
                )
            )
            for f in all_files
        )

    montages_active = [m for m in montage_keys if montage_has_pending(m)]
    if not montages_active:
        return

    models = {}
    trained = set()

    # --- TRAINING: one preprocess per interictal file, then sample each montage ---
    for training_file in training_candidates:
        if len(trained) == len(montages_active):
            break
        try:
            raw, df_raw, _, fs_raw = load_edf_file(training_file)
            preprocessed = _preprocess_edf(df_raw, raw)
        except Exception as e:
            print(
                f"  [{patient_id}] Could not load/preprocess {os.path.basename(training_file)}: {e}"
            )
            continue

        for montage_key in montages_active:
            if montage_key in trained:
                continue
            try:
                data_df_montage = _montage_from_preprocessed(preprocessed, montage_key)
                if data_df_montage.shape[1] == 0:
                    print(
                        f"  [{patient_id}:{montage_key}] No valid channels in {os.path.basename(training_file)}, trying next file for this montage..."
                    )
                    continue

                data_ndd_final = preprocess_signal(data_df_montage, fs_raw)
                if data_ndd_final is None:
                    print(
                        f"  [{patient_id}:{montage_key}] NDD pipeline preprocess failed for {os.path.basename(training_file)}, trying next file..."
                    )
                    continue

                train_end_idx = int(TRAIN_DURATION_SEC * FS_NDD)
                X_train = (
                    data_ndd_final.iloc[:train_end_idx]
                    if len(data_ndd_final) > train_end_idx
                    else data_ndd_final
                )
                model = NDD(
                    hidden_size=10,
                    fs=FS_NDD,
                    sequence_length=12,
                    forecast_length=1,
                    w_size=W_SIZE_SEC,
                    w_stride=W_STRIDE_SEC,
                    num_epochs=10,
                    batch_size="full",
                    lr=0.01,
                    use_cuda=torch.cuda.is_available(),
                    verbose=False,
                )
                model.fit(X_train)
                models[montage_key] = model
                trained.add(montage_key)
            except Exception as e:
                print(
                    f"  [{patient_id}:{montage_key}] Training failed on {os.path.basename(training_file)}: {e}"
                )
                continue

    for montage_key in montages_active:
        if montage_key not in trained:
            print(
                f"  [{patient_id}:{montage_key}] No valid training file found across all candidates. Skipping inference for this montage."
            )

    # --- INFERENCE: one preprocess per recording, then sample each montage ---
    for file_name in all_files:
        base_name = os.path.basename(file_name)
        if not force and not any(
            montage_key in models
            and not os.path.exists(
                os.path.join(
                    prob_folder, montage_key, base_name.replace(".edf", ".csv")
                )
            )
            for montage_key in montages_active
        ):
            continue

        try:
            raw, df_raw, label_df, fs_raw = load_edf_file(file_name)
            preprocessed = _preprocess_edf(df_raw, raw)
        except Exception as e:
            print(f"  [{patient_id}] Error loading {base_name}: {e}")
            continue

        for montage_key in montages_active:
            if montage_key not in models:
                continue
            output_file = os.path.join(
                prob_folder, montage_key, base_name.replace(".edf", ".csv")
            )
            if os.path.exists(output_file) and not force:
                continue

            try:
                data_df_montage = _montage_from_preprocessed(preprocessed, montage_key)
                if data_df_montage.shape[1] == 0:
                    print(
                        f"  [{patient_id}:{montage_key}] Skipping {base_name}: no valid channels for this montage"
                    )
                    continue

                data_ndd_final = preprocess_signal(data_df_montage, fs_raw)
                if data_ndd_final is None:
                    print(
                        f"  [{patient_id}:{montage_key}] Skipping {base_name}: preprocessing returned None"
                    )
                    continue

                model = models[montage_key]
                sz_prob_df = model(data_ndd_final)
                sz_prob_df = sz_prob_df.apply(
                    lambda col: clean_array(col.values, col.name), axis=0
                )
                sz_prob_times = model.get_win_times(len(data_ndd_final))

                min_len = min(len(sz_prob_df), len(sz_prob_times))
                sz_prob_df = sz_prob_df.iloc[:min_len]
                sz_prob_times = sz_prob_times[:min_len]

                sz_prob_agg = np.nanmean(sz_prob_df.values, axis=1)

                feature_time_index = df_raw.index.min() + sz_prob_times
                label_time = label_df.set_index("time")["labels"]
                label = clean_array(
                    label_time.reindex(feature_time_index, method="nearest").values[
                        :min_len
                    ]
                )

                out_data = {"sz_prob": sz_prob_agg, "label": label}
                for col in sz_prob_df.columns:
                    out_data[f"prob_{col}"] = sz_prob_df[col].values

                pred_df = pd.DataFrame(out_data, index=feature_time_index)
                pred_df.index = pd.to_datetime(pred_df.index, unit="s")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                pred_df.to_csv(output_file)

            except Exception as e:
                print(
                    f"  [{patient_id}:{montage_key}] Error inferring {base_name}: {e}"
                )
                continue

    for _k in list(models.keys()):
        del models[_k]
    gc.collect()


def _get_prob_ndd(prob_df):
    prob_mat = prob_df[[c for c in prob_df.columns if c.startswith("prob")]].values
    return prob_mat.mean(axis=1)


def process_file_pred(file_name, thres=None):
    warnings.filterwarnings("ignore")
    out_file = os.path.join(
        pred_folder, setting_folder_name, m, file_name.split("/")[-1]
    )
    if not force and os.path.exists(out_file):
        return
    prob_df = pd.read_csv(file_name, index_col=0)
    sz_prob = _get_prob_ndd(prob_df)
    pred = (sz_prob >= thres).astype(int)
    pred = get_event_smoothed_pred(
        smooth_pred(pred),
        gap_num=int(4 / feat_setting_ndd["stride"]),
        min_event_num=int(20 / feat_setting_ndd["stride"]),
    )
    pred_df = pd.DataFrame(
        np.vstack([sz_prob, pred]).T, columns=["sz_prob", "pred"], index=prob_df.index
    )
    pred_df = pd.concat([pred_df, prob_df[["label"]]], axis=1)
    pred_df.to_csv(out_file)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DynaSD Patient-Wise Pipeline")
    parser.add_argument("-d", "--data_folder", type=str, default="emu_dataset")
    parser.add_argument("-o", "--output_folder", type=str, default="ndd_results")
    parser.add_argument(
        "-p",
        "--patient_info",
        type=str,
        default="emu_dataset/dataset_admission_info.csv",
    )
    parser.add_argument("-m", "--montage", type=str, default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--setting", type=str, default="")
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument(
        "--thres_file",
        type=str,
        default="threses_all.csv",
        help="File containing thresholds for each montage",
    )
    parser.add_argument("--n_jobs", type=int, default=10)

    params = vars(parser.parse_args())
    print(f"Parsed arguments: {params}", flush=True)

    # --- Setup Base Paths ---
    base_data_folder = params["data_folder"]
    base_output_folder = params["output_folder"]
    patient_map_file = params["patient_info"]
    force = params["force"]
    n_jobs = params["n_jobs"]

    # --- Setting Folder Name (for preds, metrics, plots) ---
    thres_val = params["thres"]
    setting_val = params["setting"]

    if setting_val:
        setting_folder_name = setting_val
    else:
        setting_folder_name = f"thres{thres_val:.1f}"

    prob_folder = os.path.join(base_output_folder, "prob")
    pred_folder = os.path.join(base_output_folder, "pred")
    metric_folder = os.path.join(base_output_folder, "metrics")

    # --- Montage List ---
    if params["montage"] == "all":
        montage_keys = list(montage_dict.keys())
    else:
        montage_keys = [
            m.strip() for m in params["montage"].split(",") if m.strip() in montage_dict
        ]
    for m in montage_keys:
        os.makedirs(os.path.join(prob_folder, m), exist_ok=True)

    # Group by Patient
    try:
        all_files = glob.glob(f"{base_data_folder}/**/*.edf", recursive=True)
        all_files = sorted(all_files)
        if not all_files:
            print(f"Warning: No .edf files found in {base_data_folder}")
    except Exception as e:
        print(f"Error finding EDF files: {e}")
        all_files = []

    patient_map_files = {}
    for f in all_files:
        pid = os.path.basename(f).split("_")[0]
        if pid not in patient_map_files:
            patient_map_files[pid] = {"sz": [], "ii": []}
        if "seizure" in os.path.basename(f).lower():
            patient_map_files[pid]["sz"].append(f)
        else:
            patient_map_files[pid]["ii"].append(f)

    print(
        f"Identified {len(patient_map_files)} unique patients admissions.", flush=True
    )

    # --- STEP 1: Process Patients (Train & Inference) ---
    print(f"\n--- STEP 1: Generating Probabilities ---", flush=True)
    all_tasks = []

    for pid, files in patient_map_files.items():
        if not files["sz"] and not files["ii"]:
            continue
        all_tasks.append(
            delayed(process_patient_dataset)(
                pid,
                files["sz"],
                files["ii"],
                montage_keys,
                prob_folder,
                params["force"],
            )
        )

    Parallel(n_jobs=params["n_jobs"])(tqdm(all_tasks, desc="Processing patient"))

    # =================================================================
    # STEP 2: Generate Predictions
    # =================================================================
    print("\n--- STEP 2: Generating Predictions ---")
    pred_folder = os.path.join(base_output_folder, "pred")

    for m in montage_keys:
        print(f"Processing montage: {m}")
        prob_files = glob.glob(os.path.join(prob_folder, m, "*.csv"))
        if not prob_files:
            print(f"  No probability files found for montage {m}, skipping.")
            continue

        if "optimal_f1" in setting_folder_name:
            current_thres = get_optimal_thres(
                prob_files, _get_prob_ndd, params["thres_file"], method="f1"
            )
        elif "optimal" in setting_folder_name:
            current_thres = get_optimal_thres(
                prob_files, _get_prob_ndd, params["thres_file"], method="yodenj"
            )
        else:
            current_thres = thres_val
        print(f"  Optimal threshold for {m}: {current_thres:.4f}")

        os.makedirs(os.path.join(pred_folder, setting_folder_name, m), exist_ok=True)

        with tqdm(total=len(prob_files), desc=f"Step 2/4: Predicting: ") as pbar:
            Parallel(n_jobs=n_jobs)(
                delayed(process_file_pred)(file_name, current_thres)
                for file_name in prob_files
            )
            pbar.update(len(prob_files))

    # =================================================================
    # STEP 3: Calculate Metrics
    # =================================================================
    print("\n--- STEP 3: Calculating Metrics ---")
    metric_folder = os.path.join(base_output_folder, "metrics", setting_folder_name)
    pred_folder_setting = os.path.join(pred_folder, setting_folder_name)
    calculate_metrics_for_montages(
        montage_keys=montage_keys,
        pred_folder_setting=pred_folder_setting,
        metric_folder=metric_folder,
        stride=feat_setting_ndd["stride"],
        force=force,
    )

    # =================================================================
    # STEP 4: Generate Stats and Plots
    # =================================================================
    print("\n--- STEP 4: Generating Stats and Plots ---")
    stats_folder = os.path.join(base_output_folder, "stats", setting_folder_name)
    generate_stats_tables(
        montage_keys=montage_keys,
        metric_folder=metric_folder,
        stats_folder=stats_folder,
        patient_map_file=patient_map_file,
    )

    print("\n--- Pipeline Complete ---")
