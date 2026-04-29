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

feat_setting_ndd = {
    'win':W_SIZE_SEC,
    'stride':W_STRIDE_SEC,
    'reref':'BIPOLAR',
    'lowcut':1,
    'highcut':40
}

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


plot_vars = ['auroc_sample', 'auprc_sample', 'recall_event', 'precision_event', 'f1_event', 'fp']
plot_labels = [metric_labels.get(k, k) for k in plot_vars]

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
    parser.add_argument("--thres_file", type=str, default='threses_all.csv', help="File containing thresholds for each montage")
    parser.add_argument("--n_jobs", type=int, default=10)
    
    params = vars(parser.parse_args())
    print(f"Parsed arguments: {params}", flush=True)

    # --- Setup Base Paths ---
    base_data_folder = params['data_folder']
    base_output_folder = params['output_folder']
    patient_map_file = params['patient_info']
    force = params['force']
    n_jobs = params['n_jobs']

    if params['montage'] == 'all': montage_keys = list(montage_dict.keys())
    else: montage_keys = [m.strip() for m in params['montage'].split(',') if m.strip() in montage_dict]
    
    # --- Setting Folder Name (for preds, metrics, plots) ---
    thres_val = params['thres']
    setting_val = params['setting']
    
    if setting_val:
        setting_folder_name = setting_val
    else:
        setting_folder_name = f"thres{thres_val:.1f}"

    prob_folder = os.path.join(base_output_folder, 'prob')
    pred_folder = os.path.join(base_output_folder, 'pred')
    metric_folder = os.path.join(base_output_folder, 'metrics')
    
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

    print(f"Identified {len(patient_map_files)} unique patients admissions.", flush=True)

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
    
    # =================================================================
    # STEP 2: Generate Predictions
    # =================================================================
    print("\n--- STEP 2: Generating Predictions ---")
    pred_folder = os.path.join(base_output_folder, 'pred')

    for m in montage_keys:
        print(f"Processing montage: {m}")
        prob_files = glob.glob(os.path.join(prob_folder, m, '*.csv'))
        if not prob_files:
            print(f"  No probability files found for montage {m}, skipping.")
            continue
            
        if 'optimal_f1' in setting_folder_name:
            if params['thres_file']:
                threses = pd.read_csv(params['thres_file'])
                current_thres = threses[(threses['model']=='SPaRCNet')&(threses['montage']==m)]['thres_f1'].iloc[0]
            else:
                print("  Calculating optimal threshold...")
                current_thres = get_optimal_thres_f1(prob_files)
        elif 'optimal' in setting_folder_name:
            if params['thres_file']:
                threses = pd.read_csv(params['thres_file'])
                current_thres = threses[(threses['model']=='SPaRCNet')&(threses['montage']==m)]['thres_yodenj'].iloc[0]
            else:
                current_thres = get_optimal_thres(prob_files)
        else:
            current_thres = thres_val
        print(f"  Optimal threshold for {m}: {current_thres:.4f}")
            
        os.makedirs(os.path.join(pred_folder, setting_folder_name, m), exist_ok=True)

        with tqdm(total=len(prob_files), desc=f'Step 2/4: Predicting [{m}]') as pbar:
            Parallel(n_jobs=n_jobs)(delayed(process_file_pred)(file_name) for file_name in prob_files)
            pbar.update(len(prob_files))

    
    # =================================================================
    # STEP 3: Calculate Metrics (from calc_metrics.py and get_metrics.py)
    # =================================================================
    print("\n--- STEP 3: Calculating Metrics ---")
    metric_folder = os.path.join(base_output_folder, 'metrics', setting_folder_name)
    pred_folder_setting = os.path.join(pred_folder, setting_folder_name)
    
    for m in tqdm(montage_keys, desc='Step 3/4: Calculating Metrics'):
        os.makedirs(os.path.join(metric_folder, m), exist_ok=True)
        pred_files = glob.glob(os.path.join(pred_folder_setting, m, '*.csv'))
        if not pred_files:
            print(f"  No prediction files found for {m}. Skipping.")
            continue

        # segment metrics
        out_file = os.path.join(metric_folder,m,'segment_metrics.csv')
        if not force and os.path.exists(out_file):
            continue
        else:
            full_metrics = []
            for f in pred_files:
                pred_df = pd.read_csv(f,index_col=0)
                label = pred_df['label'].values
                prob = pred_df['sz_prob'].values
                pred = pred_df['pred'].values
                event_id = f.split('/')[-1][:-4]
                metrics = compute_metrics(label, pred, prob, stride=feat_setting_ndd['stride'])
                metric_row = pd.DataFrame([metrics],index=[event_id])
                full_metrics.append(metric_row)
            full_metrics = pd.concat(full_metrics,axis=0).sort_index()
            full_metrics[['precision_event','f1_event']] = full_metrics[['precision_event','f1_event']].fillna(0.0)
            full_metrics.to_csv(out_file)

        # patient metrics
        out_file = os.path.join(metric_folder, m, 'patient_metrics.csv')
        if not force and os.path.exists(out_file):
            continue
        else:
            pred_file_df = pd.DataFrame(pred_files, columns=['pred_file'])
            pred_file_df['admission_id'] = pred_file_df['pred_file'].apply(lambda x: x.split('/')[-1].split('_')[0])
            pred_file_df['event_id'] = pred_file_df['pred_file'].apply(lambda x: x.split('/')[-1][:-4])
            pred_file_df['is_sz'] = pred_file_df['event_id'].apply(lambda x: 'seizure' in x)

            # Calculate metrics for all files
            all_p_metrics = patient_metrics(pred_file_df, feat_setting_ndd['stride'])
            if not all_p_metrics.empty:
                all_p_metrics.to_csv(out_file)

    # =================================================================
    # STEP 4: Generate Stats and Plots (from plot_metrics.py)
    # =================================================================
    print("\n--- STEP 4: Generating Stats and Plots ---")
    figure_folder = os.path.join(base_output_folder, 'figures', setting_folder_name)
    stats_folder = os.path.join(base_output_folder, 'stats', setting_folder_name)
    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)

    # file_names = {'': 'patient_metrics.csv'}
    
    if not os.path.exists(patient_map_file):
        print(f"Error: Patient info file not found at {patient_map_file}. Cannot generate stats by type.")
    else:
        patient_map = pd.read_csv(patient_map_file, dtype={'patient_id': str, 'admission_id':str})
        file_name = 'patient_metrics.csv'
        # --- Generate Stats Tables ---
        print("    Generating stats tables...")
        try:
            all_metrics = []
            for m in montage_keys:
                metric_file = os.path.join(metric_folder, m, file_name)
                metrics = pd.read_csv(metric_file, index_col=0)
                metrics['montage'] = m
                all_metrics.append(metrics)
            all_metrics = pd.concat(all_metrics,axis=0).reset_index().rename(columns={'index':'admission_id'})
            all_metrics['admission_id'] = all_metrics['admission_id'].astype(str)

            # Overall comparison
            table1 = TableOne(all_metrics, columns=plot_vars,
                                groupby='montage', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels)
            table1_df = table1.tableone.get('Grouped by montage')
            if table1_df is not None:
                flatten_tableone(table1_df).T.to_csv(os.path.join(stats_folder, f'comparison.csv'))

            # By epilepsy type
            metrics_with_info = all_metrics.merge(patient_map, on='admission_id', how='left')
            metrics_with_info = metrics_with_info[~metrics_with_info['epilepsy_type'].isna()]
            metrics_with_info['tmp_group'] = metrics_with_info['epilepsy_type'] + '_' + metrics_with_info['montage']
            table_type = TableOne(metrics_with_info, columns=plot_vars,
                                    groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels)
            table_type_df = table_type.tableone.get('Grouped by tmp_group')
            if table_type_df is not None:
                flatten_tableone(table_type_df).T.to_csv(os.path.join(stats_folder, f'comparison_by_type.csv'))

            # By laterality
            metrics_with_info = all_metrics.merge(patient_map, on='admission_id', how='left')
            metrics_with_info = metrics_with_info[~metrics_with_info['laterality'].isna()]
            metrics_with_info['tmp_group'] = metrics_with_info['laterality'] + '_' + metrics_with_info['montage']
            table_lat = TableOne(metrics_with_info, columns=plot_vars,
                                    groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels)
            table_lat_df = table_lat.tableone.get('Grouped by tmp_group')
            if table_lat_df is not None:
                flatten_tableone(table_lat_df).T.to_csv(os.path.join(stats_folder, f'comparison_by_laterality.csv'))

            # By location
            metrics_with_info = all_metrics.merge(patient_map, on='admission_id', how='left')
            metrics_with_info = metrics_with_info[~metrics_with_info['location'].isna()]
            metrics_with_info['tmp_group'] = metrics_with_info['location'] + '_' + metrics_with_info['montage']
            table_loc = TableOne(metrics_with_info, columns=plot_vars,
                                    groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels)
            table_loc_df = table_loc.tableone.get('Grouped by tmp_group')
            if table_loc_df is not None:
                flatten_tableone(table_loc_df).T.to_csv(os.path.join(stats_folder, f'comparison_by_location.csv'))

        except Exception as e:
            print(f"    Error generating TableOne stats: {e}")

    print("\n--- Pipeline Complete ---")
