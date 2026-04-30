import argparse
import numpy as np
import pandas as pd
import os
import sys
import warnings
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_curve
import mne

# MNE logging level to avoid clutter
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# --- 1. SETUP PIPELINE PATHS ---
WORKING_DIR = os.path.abspath("")
PIPELINE_FUNCS_PATH = os.path.join(WORKING_DIR, "pipeline_functions")

sys.path.append(WORKING_DIR)
sys.path.append(PIPELINE_FUNCS_PATH)

# --- 2. CORE UTILITY IMPORTS ---
try:
    from feat_funcs import (
        apply_persistence,
        compute_novelty_scores,
        detect_seizure,
        estimate_outlier_fraction,
        extract_features,
        get_event_smoothed_pred,
        smooth_pred,
        train_one_class_svm,
    )
    from get_metrics import (
        calculate_metrics_for_montages,
        generate_stats_tables,
        get_optimal_thres,
    )
    from utils import Preprocessor, load_edf_file
except ImportError as e:
    print(f"FATAL ERROR: Could not import required pipeline components. Missing: {e}")
    sys.exit(1)

# --- CONSTANTS ---

MONTAGE_DICT = {
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

FS_FEAT = 2.0
TRAIN_DURATION_SEC = 60

feat_setting_svm = {
    "win": 1,
    "stride": 0.5,
    "reref": "BIPOLAR",
    "lowcut": 1,
    "highcut": 40,
}


# --- 3. HELPER FUNCTIONS (LOCAL DEFINITIONS) ---
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


def process_pat(pat, group):
    warnings.filterwarnings("ignore")
    group = group.sort_values("file")
    iic_file = group[group["type"] == "iic"].iloc[0]["file"]
    raw, df, label_df, fs = load_edf_file(iic_file)
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
    preprocessed = prepro.preprocess(df)
    for m in montage_keys:
        montage_processor = MONTAGE_DICT[m]
        if isinstance(montage_processor, list):
            data_df = preprocessed["BIPOLAR"]
            data_df = data_df[MONTAGE_DICT[m]]
        else:
            data_df = montage_processor(preprocessed)
        train_data = data_df.iloc[: int(fs * 60), :].values
        clf_list = []
        for i in range(train_data.shape[1]):
            X_train = extract_features(train_data[:, i], fs=fs)
            X_train = np.nan_to_num(X_train)
            clf = train_one_class_svm(X_train)
            clf_list.append(clf)

        for _, row in group.iterrows():
            file_name = row["file"]
            raw, df, label_df, fs = load_edf_file(file_name)
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
            preprocessed = prepro.preprocess(df)
            if isinstance(montage_processor, list):
                data_df = preprocessed["BIPOLAR"]
                data_df = data_df[MONTAGE_DICT[m]]
            else:
                data_df = montage_processor(preprocessed)

            prob_path = os.path.join(
                prob_folder, m, file_name.split("/")[-1].replace(".edf", ".csv")
            )
            if os.path.exists(prob_path) and not force:
                continue

            len_feat = extract_features(data_df.iloc[:, 0].values, fs=fs)
            start_time_s = data_df.index.min()
            time_vals = (
                start_time_s
                + feat_setting_svm["stride"]
                + np.arange(0, len(len_feat)) * feat_setting_svm["stride"]
            )
            feat_labels = [
                label_df.loc[
                    (data_df.index >= time_vals[i] - feat_setting_svm["win"])
                    & (data_df.index < time_vals[i]),
                    "labels",
                ]
                .any()
                .astype(int)
                for i in range(len(time_vals))
            ]
            pred_df_final = pd.DataFrame(index=time_vals)
            pred_df_final["label"] = feat_labels
            for i in range(data_df.shape[1]):
                X_test = extract_features(data_df.iloc[:, i].values, fs=fs)
                X_test = np.nan_to_num(X_test)

                y_pred = compute_novelty_scores(clf_list[i], X_test)
                nu_hat = estimate_outlier_fraction(y_pred, n=20)

                smoothing_sigma = 2 * int(len(nu_hat) / 1000 + 1)
                nu_filt = np.round(gaussian_filter1d(nu_hat, smoothing_sigma), 100)
                pred_df_final["nu_hat_" + data_df.columns[i]] = nu_hat
                pred_df_final["sz_prob_" + data_df.columns[i]] = nu_filt

            pred_df_final.index = pd.to_datetime(pred_df_final.index, unit="s")
            pred_df_final.to_csv(prob_path)


def _get_prob_svm(prob_df):
    prob_mat = prob_df[[c for c in prob_df.columns if c.startswith("nu_hat")]].values
    return prob_mat.mean(axis=1)


def process_file_pred(file_name, thres=0.99, avg=True):
    warnings.filterwarnings("ignore")
    out_file = os.path.join(pred_folder_setting, m, file_name.split("/")[-1])
    if not force and os.path.exists(out_file):
        return
    prob_df = pd.read_csv(file_name, index_col=0)
    sz_prob = _get_prob_svm(prob_df)
    pred = detect_seizure(sz_prob, threshold=thres)
    pred = get_event_smoothed_pred(
        smooth_pred(pred),
        gap_num=int(4 / feat_setting_svm["stride"]),
        min_event_num=int(20 / feat_setting_svm["stride"]),
    )  # int(4/feat_setting['stride'])
    pred = apply_persistence(pred)
    pred_df = pd.DataFrame(
        np.vstack([sz_prob, pred]).T, columns=["sz_prob", "pred"], index=prob_df.index
    )
    pred_df = pd.concat([pred_df, prob_df[["label"]]], axis=1)
    pred_df.to_csv(out_file)


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SVM Baseline Pipeline (Patient-Specific)."
    )
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        default="emu_dataset",
        help="Path to emu_dataset",
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, default="svm_results", help="Output folder"
    )
    parser.add_argument(
        "-p",
        "--patient_info",
        type=str,
        default="emu_dataset/dataset_admission_info.csv",
        help="Patient info CSV",
    )
    parser.add_argument("-m", "--montage", type=str, default="all", help="Montages")
    parser.add_argument(
        "-s",
        "--setting",
        type=str,
        default="",
        help="Setting for threshold ('optimal' or 'fixed')",
    )
    parser.add_argument(
        "-t",
        "--thres",
        type=float,
        default=0.9,
        help="Fixed threshold to use if setting is not optimal",
    )
    parser.add_argument("--n_jobs", type=int, default=40, help="Parallel jobs")
    parser.add_argument(
        "--thres_file",
        type=str,
        default="threses_all.csv",
        help="File containing thresholds for each montage",
    )
    parser.add_argument("--force", action="store_true", help="Force rerun")

    params = vars(parser.parse_args())

    base_data_folder = params["data_folder"]
    base_output_folder = params["output_folder"]
    patient_map_file = params["patient_info"]
    force = params["force"]
    n_jobs = params["n_jobs"]

    # Establish dynamic settings folder name based on Youden's optimization or fixed thres
    thres_val = params["thres"]
    setting_val = params["setting"]

    if setting_val:
        setting_folder_name = setting_val
    else:
        setting_folder_name = f"thres{thres_val:.1f}"

    prob_folder = os.path.join(base_output_folder, "prob")
    pred_folder_setting = os.path.join(base_output_folder, "pred", setting_folder_name)
    metric_folder_setting = os.path.join(
        base_output_folder, "metrics", setting_folder_name
    )
    stats_folder_setting = os.path.join(
        base_output_folder, "stats", setting_folder_name
    )

    if params["montage"] == "all":
        montage_keys = list(MONTAGE_DICT.keys())
    else:
        montage_keys = [
            m.strip() for m in params["montage"].split(",") if m.strip() in MONTAGE_DICT
        ]
    for m in montage_keys:
        os.makedirs(os.path.join(prob_folder, m), exist_ok=True)

    try:
        all_files = glob.glob(f"{base_data_folder}/**/*.edf", recursive=True)
        if not all_files:
            print(f"Warning: No .edf files found in {base_data_folder}")
    except Exception as e:
        print(f"Error finding EDF files: {e}")
        all_files = []

    file_df = pd.DataFrame({"file": all_files})
    file_df["patient"] = file_df["file"].apply(lambda x: x.split("/")[-1].split("_")[0])
    file_df["type"] = file_df["file"].apply(
        lambda x: "seizure" if "seizure" in x else "iic"
    )
    n_pat = len(file_df["patient"].unique())

    print("\n--- STEP 1: Generating Probabilities ---")

    with tqdm(total=n_pat, desc="Processing patient"):
        results = Parallel(n_jobs=20)(
            delayed(process_pat)(pat, group)
            for pat, group in file_df.groupby("patient")
        )

    print(f"Probabilities generated in {prob_folder}.")

    print("\n--- STEP 2: Generating Predictions ---")

    for m in montage_keys:
        prob_files = glob.glob(os.path.join(prob_folder, m, "*.csv"))
        if not prob_files:
            continue

        if "optimal_f1" in setting_folder_name:
            current_thres = get_optimal_thres(
                prob_files, _get_prob_svm, params["thres_file"], method="f1"
            )
        elif "optimal" in setting_folder_name:
            current_thres = get_optimal_thres(
                prob_files, _get_prob_svm, params["thres_file"], method="yodenj"
            )
        else:
            current_thres = thres_val
        print(f"  Optimal threshold for {m}: {current_thres:.4f}")

        os.makedirs(os.path.join(pred_folder_setting, m), exist_ok=True)
        with tqdm(total=len(prob_files), desc="Processing file"):
            results = Parallel(n_jobs=40)(
                delayed(process_file_pred)(file_name, current_thres)
                for file_name in prob_files
            )

    # =================================================================
    # STEP 3: Calculate Metrics
    # =================================================================
    print("\n--- STEP 3: Calculating Metrics ---")
    calculate_metrics_for_montages(
        montage_keys=montage_keys,
        pred_folder_setting=pred_folder_setting,
        metric_folder=metric_folder_setting,
        stride=feat_setting_svm["stride"],
        force=force,
    )

    # =================================================================
    # STEP 4: Generate Stats and Plots
    # =================================================================
    print("\n--- STEP 4: Generating Stats and Plots ---")
    generate_stats_tables(
        montage_keys=montage_keys,
        metric_folder=metric_folder_setting,
        stats_folder=stats_folder_setting,
        patient_map_file=patient_map_file,
    )

    print("\n--- SVM Baseline Pipeline Complete ---")
