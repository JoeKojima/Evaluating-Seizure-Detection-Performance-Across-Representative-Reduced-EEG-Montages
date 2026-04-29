# Functions and imports
import sys
import os

# --- 0. DYNASD SETUP ---
project_root = 'DynaSD-wo_dev'
if os.path.exists(project_root):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
else:
    sys.path.append(os.path.abspath("."))

print("SCRIPT STARTED - Top of file", file=sys.stderr, flush=True)
sys.stderr.flush()

import argparse
import numpy as np
import pandas as pd
import warnings
import glob
import shutil
import scipy.signal
import random
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, balanced_accuracy_score, average_precision_score, roc_auc_score
from tableone import TableOne
import torch
from mne.filter import filter_data
from scipy.ndimage import binary_opening, binary_closing 

# --- DYNASD IMPORTS ---
try:
    from DynaSD import NDD
    from DynaSD.utils import ar_one
    print("Successfully imported DynaSD components", flush=True)
except ImportError as e:
    print(f"FATAL ERROR: Could not import DynaSD. Ensure 'DynaSD-wo_dev' is in path. Error: {e}", flush=True)
    sys.exit(1)

# --- UTILS IMPORT (Legacy support) ---
try:
    from utils import Preprocessor 
except ImportError:
    class Preprocessor:
        def fit(self, settings): pass
        def preprocess(self, df): 
            df['filtered'] = df 
            return df

mne.set_log_level('ERROR')

# --- 1. SETUP PIPELINE PATHS ---
WORKING_DIR = os.path.abspath("")
sys.path.append(WORKING_DIR)

warnings.filterwarnings('ignore')
os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"
os.makedirs(".matplotlib_cache", exist_ok=True)

# --- 2. SEED & DEVICE SETUP ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

set_seed(5210)

# --- 3. CONSTANTS ---
STANDARD_BIPOLAR = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
                    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

# Model Settings (Updated to match run_full_pipeline.py)
FS_NDD = 200  
W_SIZE_SEC = 1      
W_STRIDE_SEC = 0.5  
TRAIN_MIN = 1 
TRAIN_DURATION_SEC = 60 * TRAIN_MIN 

# --- CUSTOM MONTAGE HELPERS ---
def custom_bipolar(df, pairs):
    filtered = df['filtered']
    # FIX: Use filtered.index, since df is a dictionary here, not a DataFrame
    data = pd.DataFrame(index=filtered.index) 
    
    # Helper to safely find EDF column names despite prefixes/suffixes
    def find_col(ch_name):
        if ch_name in filtered.columns: 
            return ch_name
            
        for c in filtered.columns:
            # Strip common EDF cruft to find the base channel name
            c_clean = c.upper().replace('EEG', '').replace('-REF', '').replace('-LE', '').strip()
            if c_clean == ch_name.upper(): 
                return c
        return None

    for p in pairs:
        ch1, ch2 = p.split('-')
        col1 = find_col(ch1)
        col2 = find_col(ch2)
        
        if col1 is not None and col2 is not None:
            data[p] = filtered[col1] - filtered[col2]
        else:
            pass # Channel truly missing from this patient's recording
            
    return data

montage_dict = {
    'full': ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
             'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],
    'uneeg_left_front': ['F7-T3'],
    'uneeg_left_back': ['T3-T5'],
    'uneeg_right_front': ['F8-T4'],
    'uneeg_right_back': ['T4-T6'],
    'uneeg_bilateral_back2': ['T3-T5', 'T4-T6'],
    'uneeg_bilateral_front2': ['F7-T3', 'F8-T4'],
    'uneeg_vert_left': lambda df: custom_bipolar(df, ['C3-T3']),
    'uneeg_vert_right': lambda df: custom_bipolar(df, ['C4-T4']),
    'uneeg_diag_left_front': lambda df: custom_bipolar(df, ['F3-T3']),
    'uneeg_diag_left_back': lambda df: custom_bipolar(df, ['P3-T3']),
    'uneeg_diag_right_front': lambda df: custom_bipolar(df, ['F4-T4']),
    'uneeg_diag_right_back': lambda df: custom_bipolar(df, ['P4-T4']),
    'uneeg_diag_bilateral_front': lambda df: custom_bipolar(df, ['F3-T3', 'F4-T4']),
    'uneeg_diag_bilateral_back': lambda df: custom_bipolar(df, ['P3-T3', 'P4-T4']),
    'uneeg_vert_bilateral': lambda df: custom_bipolar(df, ['C3-T3', 'C4-T4']),
    'epiminder_2': ['C3-P3', 'C4-P4'],
    'ceribell': ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2']
}

# --- 4. DATA HANDLING HELPERS ---

def _get_montage_data(df, raw, montage_key):
    montage_processor = montage_dict[montage_key]
    
    fs = raw.info['sfreq']
    prepro = Preprocessor()
    prepro.fit({'samplingFreq':fs, 'samplingFreqRaw':fs, 'channelNames':df.columns, 'studyType':'eeg', 'numberOfChannels':df.shape[1]})
    preprocessed = prepro.preprocess(df)

    if isinstance(montage_processor, list):
        data_df = preprocessed['BIPOLAR'].copy()
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
    """ Applies 1-40Hz filtering, resamples to 200Hz, and performs AR(1) whitening """
    valid_channels = data_df_montage.columns.tolist()
    eeg_data_array = clean_array(data_df_montage[valid_channels].values.T, "raw EEG")
    
    try:
        # 1-40Hz Bandpass filter matched to SPaRCNet
        filtered_data = filter_data(eeg_data_array, fs_raw, 1.0, 40.0, method='iir', verbose=False)
        filtered_data = clean_array(filtered_data, "filtered")
        
        # Resample to FS_NDD (200 Hz)
        num_samples_ndd = int(filtered_data.shape[1] / fs_raw * FS_NDD)
        data_ndd_array = scipy.signal.resample(filtered_data, num_samples_ndd, axis=1).T
        data_ndd_array = clean_array(data_ndd_array, "resampled")
        
        # DynaSD AR(1) Whitening
        data_whitened = ar_one(data_ndd_array)
        data_whitened = clean_array(data_whitened, "whitened")
        
        return pd.DataFrame(data_whitened, columns=valid_channels)
    except Exception as e:
        print(f"    Preprocessing failed: {e}")
        return None


def process_patient_dataset(patient_id, sz_files, ii_files, montage_key, prob_folder, force):
    all_files = sorted(sz_files + ii_files)

    if not force:
        pending_files = [f for f in all_files if not os.path.exists(os.path.join(prob_folder, montage_key, os.path.basename(f).replace('.edf', '.csv')))]
        if not pending_files:
            return  # All outputs already exist for this patient/montage; nothing to do
    
    # --- TRAINING STEP (First 60s of Earliest Interictal with valid channels) ---
    # Try interictal files in order, fall back to seizure files if all interictal fail.
    training_candidates = sorted(ii_files) + sorted(sz_files)
    if not training_candidates: return

    model = NDD(
        hidden_size = 10, fs = FS_NDD, sequence_length = 12, forecast_length = 1,
        w_size = W_SIZE_SEC, w_stride = W_STRIDE_SEC, num_epochs = 10, 
        batch_size = 'full', lr = 0.01, use_cuda = torch.cuda.is_available(), verbose = False
    )

    model_trained = False
    for training_file in training_candidates:
        try:
            raw, df_raw, _, fs_raw = load_edf_file(training_file)
            data_df_montage = _get_montage_data(df_raw, raw, montage_key)
            if data_df_montage.shape[1] == 0:
                print(f"  [{patient_id}:{montage_key}] No valid channels in {os.path.basename(training_file)}, trying next file...")
                continue

            data_ndd_final = preprocess_signal(data_df_montage, fs_raw)
            if data_ndd_final is None:
                print(f"  [{patient_id}:{montage_key}] Preprocessing failed for {os.path.basename(training_file)}, trying next file...")
                continue

            train_end_idx = int(TRAIN_DURATION_SEC * FS_NDD)
            X_train = data_ndd_final.iloc[:train_end_idx] if len(data_ndd_final) > train_end_idx else data_ndd_final
            model.fit(X_train)
            model_trained = True
            break

        except Exception as e:
            print(f"  [{patient_id}:{montage_key}] Training failed on {os.path.basename(training_file)}: {e}, trying next file...")
            continue

    if not model_trained:
        print(f"  [{patient_id}:{montage_key}] No valid training file found across all candidates. Skipping.")
        return

    # --- INFERENCE STEP ---
    for file_name in all_files:
        base_name = os.path.basename(file_name)
        output_file = os.path.join(prob_folder, montage_key, base_name.replace('.edf', '.csv'))
        if os.path.exists(output_file) and not force: continue

        try:
            raw, df_raw, label_df, fs_raw = load_edf_file(file_name)
            data_df_montage = _get_montage_data(df_raw, raw, montage_key)
            if data_df_montage.shape[1] == 0:
                print(f"  [{patient_id}:{montage_key}] Skipping {base_name}: no valid channels for this montage")
                continue

            data_ndd_final = preprocess_signal(data_df_montage, fs_raw)
            if data_ndd_final is None:
                print(f"  [{patient_id}:{montage_key}] Skipping {base_name}: preprocessing returned None")
                continue
            
            sz_prob_df = model(data_ndd_final) 
            sz_prob_df = sz_prob_df.apply(lambda col: clean_array(col.values, col.name), axis=0)
            sz_prob_times = model.get_win_times(len(data_ndd_final))
            
            min_len = min(len(sz_prob_df), len(sz_prob_times))
            sz_prob_df = sz_prob_df.iloc[:min_len]
            sz_prob_times = sz_prob_times[:min_len]

            sz_prob_agg = np.nanmean(sz_prob_df.values, axis=1)
            
            feature_time_index = df_raw.index.min() + sz_prob_times
            label_time = label_df.set_index('time')['labels']
            label = clean_array(label_time.reindex(feature_time_index, method='nearest').values[:min_len])

            out_data = {
                'SZ': sz_prob_agg, 'sz_prob': sz_prob_agg, 
                'label': label,
                'pred': np.zeros(min_len), 
                'smoothed_pred': np.zeros(min_len) 
            }
            # Save Per-Channel Probabilities
            for col in sz_prob_df.columns:
                out_data[f'prob_{col}'] = sz_prob_df[col].values

            pred_df = pd.DataFrame(out_data, index=feature_time_index)
            pred_df.index = pd.to_datetime(pred_df.index, unit='s')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            pred_df.to_csv(output_file)
            
        except Exception as e:
            print(f"  [{patient_id}:{montage_key}] Error inferring {base_name}: {e}")
            continue
    del model; gc.collect()

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DynaSD Patient-Wise Pipeline")
    parser.add_argument("-d", "--data_folder", type=str, default='emu_dataset')
    parser.add_argument("-o", "--output_folder", type=str, default='ndd_results')
    parser.add_argument("-p", "--patient_info", type=str, default='emu_dataset/dataset_admission_info.csv')
    parser.add_argument("-m", "--montage", type=str, default='all')
    parser.add_argument("--force", action='store_true')
    parser.add_argument("--setting", type=str, default='') 
    parser.add_argument("--thres", type=float, default=0.5) 
    parser.add_argument("--do_plot", action='store_true') 
    parser.add_argument("--n_jobs", type=int, default=10)
    
    params = vars(parser.parse_args())
    print(f"Parsed arguments: {params}", flush=True)

    if params['montage'] == 'all': montage_keys = list(montage_dict.keys())
    else: montage_keys = [m.strip() for m in params['montage'].split(',') if m.strip() in montage_dict]
    
    prob_folder = os.path.join(params['output_folder'], 'prob')
    metric_folder = os.path.join(params['output_folder'], 'metrics')
    
    # --- Locate Files ---
    try:
        all_files = glob.glob(f"{params['data_folder']}/**/*.edf", recursive=True)
        if not all_files:
            print(f"Warning: No .edf files found in {params['data_folder']}")
    except Exception as e:
        print(f"Error finding EDF files: {e}")
        all_files = []

    # Group by Patient
    patient_map_files = {}
    for f in all_files:
        pid = os.path.basename(f).split('_')[0]
        if pid not in patient_map_files: patient_map_files[pid] = {'sz': [], 'ii': []}
        if 'seizure' in os.path.basename(f).lower(): patient_map_files[pid]['sz'].append(f)
        else: patient_map_files[pid]['ii'].append(f)

    print(f"Identified {len(patient_map_files)} unique patients.", flush=True)

    # --- STEP 1: Process Patients (Train & Inference) ---
    print(f"\n{'='*60}\nSTEP 1: Processing Patients\n{'='*60}", flush=True)
    all_tasks = []
    for m in montage_keys:
        for pid, files in patient_map_files.items():
            if not files['sz'] and not files['ii']: continue
            all_tasks.append(delayed(process_patient_dataset)(
                pid, files['sz'], files['ii'], m, prob_folder, params['force']
            ))
    
    Parallel(n_jobs=params['n_jobs'])(tqdm(all_tasks, desc="Processing patients"))
    
    # --- STEP 2: Metrics & Global Youden's J Optimization ---
    print(f"\n{'='*60}\nSTEP 2: Global Threshold Optimization & Metrics\n{'='*60}", flush=True)
    
    for m in tqdm(montage_keys, desc="Calculating metrics"):
        metric_dest = os.path.join(metric_folder, m)
        os.makedirs(metric_dest, exist_ok=True)
        
        pred_files = glob.glob(os.path.join(prob_folder, m, '*.csv'))
        sz_pred_files = [f for f in pred_files if 'seizure' in os.path.basename(f).lower() or 'event' in os.path.basename(f).lower()]
        
        if not sz_pred_files: continue

        # --- A. Calculate Youden's J across ALL patients ---
        print(f"  {m}: Optimizing threshold via Youden's J...", flush=True)
        y_true_all = []
        y_score_all = []
        
        for f in sz_pred_files:
            try:
                df_temp = pd.read_csv(f, usecols=['label', 'sz_prob'])
                y_true_all.append(df_temp['label'].values)
                y_score_all.append(df_temp['sz_prob'].values)
            except: continue
        
        if not y_true_all: continue
        
        y_true_flat = np.concatenate(y_true_all)
        y_score_flat = np.concatenate(y_score_all)
        
        # Calculate ROC and Optimal Threshold
        fpr, tpr, thresholds = roc_curve(y_true_flat, y_score_flat)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        max_j = j_scores[best_idx]
        
        print(f"  {m}: Optimal Threshold = {best_thresh:.6f} (J={max_j:.4f})", flush=True)
        with open(os.path.join(metric_dest, 'threshold_info.txt'), 'w') as tf:
            tf.write(f"Global Youden's J Threshold: {best_thresh}\nMax J: {max_j}\n")
        
        # --- B. Re-calculate metrics ---
        patient_metrics_list = []
        df_files = pd.DataFrame({'file': sz_pred_files})
        df_files['pid'] = df_files['file'].apply(lambda x: os.path.basename(x).split('_')[0])
        
        for pid, group in df_files.groupby('pid'):
            seg_metrics = []
            auc_prob, auc_label, auc_pred = [], [], []
            
            for f in group['file']:
                try:
                    df = pd.read_csv(f, index_col=0)
                    l = df['label'].values
                    p = df['sz_prob'].values
                    
                    # Apply Optimal Threshold
                    pr_raw = (p >= best_thresh).astype(int)
                    # Apply Manuscript-Specific Smoothing (Closing + Opening 20s)
                    pr_smooth = smooth_predictions(pr_raw)
                    
                    seg_metrics.append(compute_metrics(l, pr_smooth, p))
                    auc_prob.extend(p); auc_label.extend(l); auc_pred.extend(pr_smooth)
                except: continue
            
            if not seg_metrics: continue
            seg_df = pd.DataFrame(seg_metrics)
            agg = compute_metrics(np.array(auc_label), np.array(auc_pred), np.array(auc_prob))
            for col in ['avg_sz_dura', 'recall_event', 'balanced_acc', 'fp']:
                agg[col] = seg_df[col].mean()
            agg['num_sz'] = seg_df['num_sz'].sum()
            agg['patient_id'] = pid
            patient_metrics_list.append(agg)
            
        if patient_metrics_list:
            final_df = pd.DataFrame(patient_metrics_list).set_index('patient_id')
            final_df.to_csv(os.path.join(metric_dest, 'patient_metrics.csv'))
            print(f"  {m}: Saved metrics for {len(final_df)} patients.")

    print(f"\n{'='*60}\nPIPELINE COMPLETE\n{'='*60}", flush=True)
