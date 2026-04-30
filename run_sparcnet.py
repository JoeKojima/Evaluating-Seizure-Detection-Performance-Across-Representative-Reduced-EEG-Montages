# Functions and imports
import argparse
import glob
import os
import sys
import warnings

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm

# --- Setup Environment ---
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# --- Path Configuration ---
working_dir = os.path.abspath("")
sparcnet_path = os.path.join(working_dir, "SPARCNET")
funcs_path = os.path.join(working_dir, "pipeline_functions")

sys.path.append(working_dir)
sys.path.append(funcs_path)
sys.path.append(sparcnet_path)

# --- Import User Dependencies ---
# These files must exist in your 'pipeline_functions' and 'SPaRCNet' folders
try:
    from DenseNetClassifier import *  # noqa: F401 (needed for torch.load)
    from feat_funcs import get_event_smoothed_pred, smooth_pred
    from get_metrics import (
        calculate_metrics_for_montages,
        generate_stats_tables,
        get_optimal_thres,
    )
    from utils import (
        bandpass_filter,
        downsample,
        Preprocessor,
        load_edf_file,
    )
except ImportError as e:
    print("Error: Could not import dependency. Make sure files exist.")
    print(f"Missing: {e}")
    print("Please ensure 'utils.py', 'feat_funcs.py' are in 'pipeline_functions/'")
    print("and 'DenseNetClassifier.py' is in 'SPARCNET/'.")
    sys.exit(1)

# --- Model & Feature Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model_cnn = torch.load(
        os.path.join(sparcnet_path, "sparcnet_pretrain.pt"),
        map_location=torch.device(device),
        weights_only=False,
    )
    model_cnn.eval()
    print("SPaRCNet model loaded successfully.")
except Exception as e:
    print(f"Error loading model 'sparcnet_pretrain.pt' from '{sparcnet_path}': {e}")
    sys.exit(1)

feat_setting_sparcnet = {
    "name": "sparcnet",
    "win": int(10),
    "stride": int(2),
    "reref": "BIPOLAR",
    "resample": 200,
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

# This dictionary is defined globally so process_file_sparcnet can access it
# It will be populated in the main block
process_file_globals = {"prob_folder": None, "force": False, "montage_keys": []}

# This dictionary is defined globally so process_file_pred can access it
process_file_pred_globals = {
    "pred_folder": None,
    "setting_folder": None,
    "montage_key": None,
    "force": False,
    "thres": 0.5,
}


# --- Helper Functions (from run_sparcnet.py) ---
def custom_bipolar(df, pairs):
    filtered = df["filtered"]
    columns = [
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
    ]
    data = df["BIPOLAR"][columns].copy()  # Use .copy() to avoid SettingWithCopyWarning
    data.loc[:, columns] = 0
    for p in pairs:
        ch1, ch2 = p.split("-")
        try:
            ch_data = filtered[ch1] - filtered[ch2]
            # Find the correct column to assign to
            target_col = [c for c in columns if c.startswith(ch1)][0]
            data.loc[:, target_col] = ch_data
        except KeyError:
            print(f"Warning: Channel {ch1} or {ch2} not found in custom_bipolar.")
            pass
    return data


def sparcnet_single(data, fs):
    if "Fz-Cz" in data.columns:
        data = data.drop(columns=["Fz-Cz"])
    if "Cz-Pz" in data.columns:
        data = data.drop(columns=["Cz-Pz"])
    data = data.values
    data = bandpass_filter(
        data,
        fs,
        lo=feat_setting_sparcnet["lowcut"],
        hi=feat_setting_sparcnet["highcut"],
    )
    data = downsample(data, fs, feat_setting_sparcnet["resample"])
    data = np.where(data <= 500, data, 500)
    data = np.where(data >= -500, data, -500)
    data = torch.from_numpy(data).float()
    data = data.T.unsqueeze(0)
    data = data.to(device)
    output, _ = model_cnn(data)
    sz_prob = F.softmax(output, 1).detach().cpu().numpy().flatten()
    return sz_prob


def process_file_sparcnet(file_name):
    """
    Main processing function for run_sparcnet.py logic.
    Uses globals from process_file_globals dict.

    --- VERSION 2: Fixed timestamp/indexing bug ---
    """
    warnings.filterwarnings("ignore")

    # Access globals
    prob_folder = process_file_globals["prob_folder"]
    force = process_file_globals["force"]
    montage_keys = process_file_globals["montage_keys"]

    try:
        # fs_from_load is the variable from the original function
        raw, df, label_df, fs_from_load = load_edf_file(file_name)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return

    prepro = Preprocessor()
    prepro.fit(
        {
            "samplingFreq": fs_from_load,
            "samplingFreqRaw": fs_from_load,
            "channelNames": df.columns,
            "studyType": "eeg",
            "numberOfChannels": df.shape[1],
        }
    )
    preprocessed = prepro.preprocess(df)

    win_sec = feat_setting_sparcnet["win"]
    stride_sec = feat_setting_sparcnet["stride"]

    # --- START FIX ---
    # Get the original sampling frequency from the raw file
    fs = raw.info["sfreq"]

    # Convert window and stride from seconds to samples
    win_samples = int(win_sec * fs)
    stride_samples = int(stride_sec * fs)
    # --- END FIX ---

    for m in montage_keys:
        prob_path = os.path.join(
            prob_folder, m, os.path.basename(file_name).replace(".edf", ".csv")
        )
        if os.path.exists(prob_path) and not force:
            continue

        montage_processor = montage_dict[m]
        if isinstance(montage_processor, list):
            data_df = preprocessed["BIPOLAR"]
            data_df = data_df[montage_dict["full"]]
            # Create a copy to avoid SettingWithCopyWarning
            data_df = data_df.copy()
            data_df.loc[:, ~data_df.columns.isin(montage_processor)] = 0
        elif "simulate" in m:
            data_df = montage_processor(raw)
        else:
            data_df = montage_processor(preprocessed)

        # --- START FIX: Create sample-based windows ---
        n_samples_total = len(data_df)
        window_starts_samples = np.arange(
            0, n_samples_total - win_samples + 1, stride_samples
        )

        if not np.any(window_starts_samples):
            print(
                f"No windows generated for {file_name}, montage {m} (Total samples: {n_samples_total})"
            )
            continue
        # --- END FIX ---

        sz_prob_df = []
        # --- START FIX: Loop over samples, not time ---
        for win_start_sample in window_starts_samples:
            win_end_sample = win_start_sample + win_samples

            # Use .iloc for integer-based slicing
            clip = data_df.iloc[win_start_sample:win_end_sample]

            # The clip length check
            if clip.shape[0] < win_samples:
                continue

            # Get the index from the DataFrame for the *end* of the window
            feat_index = data_df.index[win_end_sample - 1]

            # Get label (this still uses time, which is fine)
            win_start_time = data_df.index[win_start_sample]
            win_end_time = data_df.index[win_end_sample - 1]
            # Use .any() in case a seizure spans the boundary
            feat_label = (
                label_df.loc[
                    (label_df["time"] >= win_start_time)
                    & (label_df["time"] <= win_end_time),
                    "labels",
                ]
                .any()
                .astype(int)
            )

            sz_prob = sparcnet_single(clip, fs)
            sz_prob_df.append(
                pd.DataFrame(
                    [np.r_[sz_prob, [feat_label]]],
                    columns=["SZ", "LPD", "GPD", "LRDA", "GRDA", "OTHER", "label"],
                    index=[feat_index],
                )
            )
        # --- END FIX ---

        if not sz_prob_df:
            print(
                f"No data processed for {file_name}, montage {m} (No valid clips found)"
            )
            continue

        sz_prob_df = pd.concat(sz_prob_df)
        sz_prob_df.to_csv(prob_path)


# =============================================================================
# PART 2: SPaRCNet Prediction Thresholding (from sparcnet_pred.py)
# =============================================================================


def _get_prob_sparcnet(prob_df):
    return prob_df.iloc[:, 1].values


def process_file_pred(file_name):
    """
    Main processing function for sparcnet_pred.py logic.
    Uses globals from process_file_pred_globals dict.
    """
    warnings.filterwarnings("ignore")

    # Access globals
    pred_folder = process_file_pred_globals["pred_folder"]
    setting_folder = process_file_pred_globals["setting_folder"]
    m = process_file_pred_globals["montage_key"]
    force = process_file_pred_globals["force"]
    thres = process_file_pred_globals["thres"]

    out_file = os.path.join(pred_folder, setting_folder, m, os.path.basename(file_name))
    if not force and os.path.exists(out_file):
        return
    try:
        prob_df = pd.read_csv(file_name, index_col=0)
    except Exception as e:
        print(f"Error reading prob file {file_name}: {e}")
        return

    sz_prob = _get_prob_sparcnet(prob_df)
    pred = (sz_prob >= thres).astype(int)
    pred = get_event_smoothed_pred(
        smooth_pred(pred),
        gap_num=int(4 / feat_setting_sparcnet["stride"]),
        min_event_num=int(20 / feat_setting_sparcnet["stride"]),
    )
    pred_df = pd.DataFrame(
        np.vstack([sz_prob, pred]).T, columns=["sz_prob", "pred"], index=prob_df.index
    )
    pred_df = pd.concat([pred_df, prob_df.iloc[:, -1]], axis=1)
    pred_df.to_csv(out_file)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        description="Run Full SPaRCNet Pipeline: Probs -> Preds -> Metrics -> Plots"
    )

    # --- Define Arguments ---
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        default="emu_dataset",
        help="Path to the emu_dataset folder (containing seizure/ and interictal/)",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="sparcnet_results",
        help="Main output folder for probs, preds, metrics, etc.",
    )
    parser.add_argument(
        "-p",
        "--patient_info",
        type=str,
        default="emu_dataset/emu_patient_info.csv",
        help="Path to emu_patient_info.csv",
    )
    parser.add_argument(
        "-m",
        "--montage",
        type=str,
        default="all",
        help="Comma-separated list of montages (or 'all')",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-running all steps"
    )
    parser.add_argument(
        "-s",
        "--setting",
        type=str,
        default="",
        help="Threshold setting ('optimal_f1' or 'optimal' (yodenj) or leave blank for fixed)",
    )
    parser.add_argument(
        "-t",
        "--thres",
        type=float,
        default=0.5,
        help="Fixed threshold to use if 'optimal' is not set",
    )
    parser.add_argument(
        "--thres_file",
        type=str,
        default="threses_all.csv",
        help="File containing thresholds for each montage",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=40, help="Number of parallel jobs"
    )

    params = vars(parser.parse_args())

    # --- Setup Base Paths ---
    base_data_folder = params["data_folder"]
    base_output_folder = params["output_folder"]
    patient_map_file = params["patient_info"]
    force = params["force"]
    n_jobs = params["n_jobs"]

    # --- Montage List ---
    if params["montage"] == "all":
        montage_keys = list(montage_dict.keys())
    else:
        montage_keys = params["montage"].split(",")

    # --- Setting Folder Name (for preds, metrics, plots) ---
    thres_val = params["thres"]
    setting_val = params["setting"]

    if setting_val:
        setting_folder_name = setting_val
    else:
        setting_folder_name = f"thres{thres_val:.1f}"

    print(f"Using setting: {setting_folder_name}")

    # =================================================================
    # STEP 1: Generate Probabilities (from run_sparcnet.py)
    # =================================================================
    print("\n--- STEP 1: Generating Probabilities ---")
    prob_folder = os.path.join(base_output_folder, "prob")
    os.makedirs(prob_folder, exist_ok=True)

    for m in montage_keys:
        os.makedirs(os.path.join(prob_folder, m), exist_ok=True)

    try:
        all_files = glob.glob(f"{base_data_folder}/**/*.edf", recursive=True)
        if not all_files:
            print(f"Warning: No .edf files found in {base_data_folder}")
    except Exception as e:
        print(f"Error finding EDF files: {e}")
        all_files = []

    if all_files:
        # Update globals for the parallel function
        process_file_globals["prob_folder"] = prob_folder
        process_file_globals["force"] = force
        process_file_globals["montage_keys"] = montage_keys

        with tqdm(total=len(all_files), desc="Step 1/4: Processing EDFs") as pbar:
            Parallel(n_jobs=n_jobs)(
                delayed(process_file_sparcnet)(file_name) for file_name in all_files
            )
            pbar.update(len(all_files))
    else:
        print("Skipping Step 1, no files found.")

    # =================================================================
    # STEP 2: Generate Predictions
    # =================================================================
    print("\n--- STEP 2: Generating Predictions ---")
    pred_folder = os.path.join(base_output_folder, "pred")

    process_file_pred_globals["pred_folder"] = pred_folder
    process_file_pred_globals["setting_folder"] = setting_folder_name
    process_file_pred_globals["force"] = force

    for m in montage_keys:
        print(f"Processing montage: {m}")
        prob_files = glob.glob(os.path.join(prob_folder, m, "*.csv"))
        if not prob_files:
            print(f"  No probability files found for montage {m}, skipping.")
            continue

        if "optimal_f1" in setting_folder_name:
            current_thres = get_optimal_thres(
                params["thres_file"], prob_files, _get_prob_sparcnet, method="f1"
            )
        elif "optimal" in setting_folder_name:
            current_thres = get_optimal_thres(
                params["thres_file"], prob_files, _get_prob_sparcnet, method="yodenj"
            )
        else:
            current_thres = thres_val
        print(f"  Optimal threshold for {m}: {current_thres:.4f}")

        os.makedirs(os.path.join(pred_folder, setting_folder_name, m), exist_ok=True)

        # Update globals for parallel function
        process_file_pred_globals["montage_key"] = m
        process_file_pred_globals["thres"] = current_thres

        with tqdm(total=len(prob_files), desc=f"Step 2/4: Predicting: ") as pbar:
            Parallel(n_jobs=n_jobs)(
                delayed(process_file_pred)(file_name) for file_name in prob_files
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
        stride=feat_setting_sparcnet["stride"],
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

    print("\n--- SPaRCNetPipeline Complete ---")
