# Functions and imports
import argparse
import numpy as np
import pandas as pd
import os
import sys
import scipy
import mne
from mne.filter import filter_data, notch_filter
from tqdm import tqdm
import pickle
import json
import warnings
import glob
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, balanced_accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score
from tableone import TableOne
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

# --- Setup Environment ---
os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"
os.makedirs(".matplotlib_cache", exist_ok=True)
warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# --- Path Configuration ---
working_dir = os.path.abspath("")
sparcnet_path = os.path.join(working_dir, 'SPARCNET')
funcs_path = os.path.join(working_dir, 'pipeline_functions')

sys.path.append(working_dir)
sys.path.append(funcs_path)
sys.path.append(sparcnet_path)

# --- Import User Dependencies ---
# These files must exist in your 'pipeline_functions' and 'SPaRCNet' folders
try:
    from utils import * # Expects Preprocessor, get_event_smoothed_pred, smooth_pred
    from feat_funcs import * # Expects bandpass_filter, downsample
    from DenseNetClassifier import * # Expects DenseNetClassifier class
    from calc_metrics import * # Expects compute_metrics
except ImportError as e:
    print(f"Error: Could not import dependency. Make sure files exist.")
    print(f"Missing: {e}")
    print("Please ensure 'utils.py', 'feat_funcs.py' are in 'pipeline_functions/'")
    print("and 'DenseNetClassifier.py' is in 'SPARCNET/'.")
    sys.exit(1)

# =============================================================================
# PART 1: SPaRCNet Probability Generation (from run_sparcnet.py)
# =============================================================================

# --- Model & Feature Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model_cnn = torch.load(os.path.join(sparcnet_path, "sparcnet_pretrain.pt"), map_location=torch.device(device), weights_only=False)
    model_cnn.eval()
    print("SPaRCNet model loaded successfully.")
except Exception as e:
    print(f"Error loading model 'sparcnet_pretrain.pt' from '{sparcnet_path}': {e}")
    sys.exit(1)

feat_setting_sparcnet = {
    'name': 'sparcnet',
    'win': int(10), 'stride': int(2),
    'reref': 'BIPOLAR', 'resample': 200,
    'lowcut': 1, 'highcut': 40
}

# --- Helper Functions (from run_sparcnet.py) ---

def sparcnet_single(data, fs):
    if 'Fz-Cz' in data.columns:
        data = data.drop(columns=['Fz-Cz'])
    if 'Cz-Pz' in data.columns:
        data = data.drop(columns=['Cz-Pz'])
    data = data.values
    data = bandpass_filter(data, fs, lo=feat_setting_sparcnet['lowcut'], hi=feat_setting_sparcnet['highcut'])
    data = downsample(data, fs, feat_setting_sparcnet['resample'])
    data = np.where(data <= 500, data, 500)
    data = np.where(data >= -500, data, -500)
    data = torch.from_numpy(data).float()
    data = data.T.unsqueeze(0)
    data = data.to(device)
    output, _ = model_cnn(data)
    sz_prob = F.softmax(output, 1).detach().cpu().numpy().flatten()
    return sz_prob

def custom_bipolar(df, pairs):
    filtered = df['filtered']
    columns = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
               'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
    data = df['BIPOLAR'][columns].copy() # Use .copy() to avoid SettingWithCopyWarning
    data.loc[:, columns] = 0
    for p in pairs:
        ch1, ch2 = p.split('-')
        try:
            ch_data = filtered[ch1] - filtered[ch2]
            # Find the correct column to assign to
            target_col = [c for c in columns if c.startswith(ch1)][0]
            data.loc[:, target_col] = ch_data
        except KeyError:
            print(f"Warning: Channel {ch1} or {ch2} not found in custom_bipolar.")
            pass
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

# This dictionary is defined globally so process_file_sparcnet can access it
# It will be populated in the main block
process_file_globals = {
    'prob_folder': None,
    'force': False,
    'montage_keys': []
}


def process_file_sparcnet(file_name):
    """
    Main processing function for run_sparcnet.py logic.
    Uses globals from process_file_globals dict.
    
    --- VERSION 2: Fixed timestamp/indexing bug ---
    """
    warnings.filterwarnings('ignore')
    
    # Access globals
    prob_folder = process_file_globals['prob_folder']
    force = process_file_globals['force']
    montage_keys = process_file_globals['montage_keys']
    
    try:
        # fs_from_load is the variable from the original function
        raw, df, label_df, fs_from_load = load_edf_file(file_name)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return

    prepro = Preprocessor()
    prepro.fit({'samplingFreq': fs_from_load, 'samplingFreqRaw': fs_from_load, 'channelNames': df.columns, 'studyType': 'eeg', 'numberOfChannels': df.shape[1]})
    preprocessed = prepro.preprocess(df)
    
    win_sec = feat_setting_sparcnet['win']
    stride_sec = feat_setting_sparcnet['stride']

    # --- START FIX ---
    # Get the original sampling frequency from the raw file
    fs = raw.info['sfreq'] 
    
    # Convert window and stride from seconds to samples
    win_samples = int(win_sec * fs)
    stride_samples = int(stride_sec * fs)
    # --- END FIX ---
    
    for m in montage_keys:
        prob_path = os.path.join(prob_folder, m, os.path.basename(file_name).replace('.edf', '.csv'))
        if os.path.exists(prob_path) and not force:
            continue
        
        montage_processor = montage_dict[m]
        if isinstance(montage_processor, list):
            data_df = preprocessed['BIPOLAR']
            data_df = data_df[montage_dict['full']]
            # Create a copy to avoid SettingWithCopyWarning
            data_df = data_df.copy()
            data_df.loc[:, ~data_df.columns.isin(montage_processor)] = 0
        elif 'simulate' in m:
            data_df = montage_processor(raw)
        else:
            data_df = montage_processor(preprocessed)
        
        
        # --- START FIX: Create sample-based windows ---
        n_samples_total = len(data_df)
        window_starts_samples = np.arange(0, n_samples_total - win_samples + 1, stride_samples)
        
        if not np.any(window_starts_samples):
            print(f"No windows generated for {file_name}, montage {m} (Total samples: {n_samples_total})")
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
            feat_label = label_df.loc[(label_df['time'] >= win_start_time) & (label_df['time'] <= win_end_time), 'labels'].any().astype(int)
            
            sz_prob = sparcnet_single(clip, fs)
            sz_prob_df.append(pd.DataFrame([np.r_[sz_prob, [feat_label]]], columns=['SZ', 'LPD', 'GPD', 'LRDA', 'GRDA', 'OTHER', 'label'], index=[feat_index]))
        # --- END FIX ---
        
        if not sz_prob_df:
            print(f"No data processed for {file_name}, montage {m} (No valid clips found)")
            continue
            
        sz_prob_df = pd.concat(sz_prob_df)
        sz_prob_df.to_csv(prob_path)

# =============================================================================
# PART 2: SPaRCNet Prediction Thresholding (from sparcnet_pred.py)
# =============================================================================

montage_list_pred = list(montage_dict.keys()) # Use keys from montage_dict

# This dictionary is defined globally so process_file_pred can access it
process_file_pred_globals = {
    'pred_folder': None,
    'setting_folder': None,
    'montage_key': None,
    'force': False,
    'thres': 0.5
}

def get_optimal_thres(prob_files):
    all_prob = []
    all_label = []
    for f in prob_files:
        try:
            prob_df = pd.read_csv(f, index_col=0)
            sz_prob = prob_df['LPD'].values  # Assuming SZ is column 1 (0-indexed)
            label = prob_df['label'].values
            all_prob.extend(sz_prob)
            all_label.extend(label)
        except Exception as e:
            print(f"Error reading prob file {f}: {e}")
            continue
    
    if not all_label:
        print("Warning: No labels found for optimal threshold calculation. Defaulting to 0.5.")
        return 0.5
        
    fpr, tpr, thres = roc_curve(all_label, all_prob)
    opt_thres = thres[np.argmax(tpr - fpr)]
    return opt_thres

from timescoring.annotations import Annotation
from timescoring import scoring 
from joblib import Parallel, delayed

def compute_eventwise_f1(true, prob, t, stride):
    pred = (prob >= t).astype(int)
    labels = Annotation(true, 1/stride)
    preds = Annotation(pred, 1/stride)
    param = scoring.EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=5 * 60,
        minDurationBetweenEvents=90)
    scores = scoring.EventScoring(labels, preds, param)
    return scores.f1
        
def get_optimal_thres_f1(prob_files, stride):
    true = []
    prob = []
    for f in prob_files:
        prob_df = pd.read_csv(f, index_col=0)
        sz_prob = prob_df['LPD'].values
        label = prob_df['label'].values
        prob.extend(sz_prob)
        true.extend(label)
    _, _, thres = roc_curve(true, prob)
    N = 200
    if len(thres) > N:
        idx = np.linspace(0, len(thres) - 1, N).astype(int)
        thres = thres[idx]
    results = Parallel(n_jobs=40)(
            delayed(compute_eventwise_f1)(true, prob, t, stride) for t in thres
        )
    opt_ind = np.argmax(results)
    opt_thres = thres[opt_ind] 
    return opt_thres

def process_file_pred(file_name):
    """
    Main processing function for sparcnet_pred.py logic.
    Uses globals from process_file_pred_globals dict.
    """
    warnings.filterwarnings('ignore')
    
    # Access globals
    pred_folder = process_file_pred_globals['pred_folder']
    setting_folder = process_file_pred_globals['setting_folder']
    m = process_file_pred_globals['montage_key']
    force = process_file_pred_globals['force']
    thres = process_file_pred_globals['thres']

    out_file = os.path.join(pred_folder, setting_folder, m, os.path.basename(file_name))
    if not force and os.path.exists(out_file):
        return
    try:
        prob_df = pd.read_csv(file_name, index_col=0)
    except Exception as e:
        print(f"Error reading prob file {file_name}: {e}")
        return
        
    prob = prob_df.iloc[:, :6].values
    sz_prob = prob[:,1]
    pred = (sz_prob >= thres).astype(int)
    pred = get_event_smoothed_pred(smooth_pred(pred), gap_num=int(4/feat_setting_sparcnet['stride']), min_event_num=int(20/feat_setting_sparcnet['stride']))
    pred_df = pd.DataFrame(np.vstack([sz_prob, pred]).T, columns=['sz_prob', 'pred'], index=prob_df.index)
    pred_df = pd.concat([pred_df, prob_df.iloc[:, -1]], axis=1)
    pred_df.to_csv(out_file)


# =============================================================================
# PART 4: Statistics
# =============================================================================

plot_vars = ['auroc_sample', 'auprc_sample', 'recall_event', 'precision_event', 'f1_event', 'fp']
plot_labels = [metric_labels.get(k, k) for k in plot_vars]

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Run Full SPaRCNet Pipeline: Probs -> Preds -> Metrics -> Plots"
    )

    # --- Define Arguments ---
    parser.add_argument("-d", "--data_folder", type=str, default='emu_dataset', help="Path to the emu_dataset folder (containing seizure/ and interictal/)")
    parser.add_argument("-o", "--output_folder", type=str, default='sparcnet_results', help="Main output folder for probs, preds, metrics, etc.")
    parser.add_argument("-p", "--patient_info", type=str, default='emu_dataset/emu_patient_info.csv', help="Path to emu_patient_info.csv")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Comma-separated list of montages (or 'all')")
    parser.add_argument("--force", action='store_true', help="Force re-running all steps")
    parser.add_argument("-s", "--setting", type=str, default='', help="Threshold setting ('optimal' or leave blank for fixed)")
    parser.add_argument("-t", "--thres", type=float, default=0.5, help="Fixed threshold to use if 'optimal' is not set")
    parser.add_argument("--thres_file", type=str, default='threses_all.csv', help="File containing thresholds for each montage")
    parser.add_argument("--n_jobs", type=int, default=40, help="Number of parallel jobs")
    
    params = vars(parser.parse_args())

    # --- Setup Base Paths ---
    base_data_folder = params['data_folder']
    base_output_folder = params['output_folder']
    patient_map_file = params['patient_info']
    force = params['force']
    n_jobs = params['n_jobs']

    # --- Montage List ---
    if params['montage'] == 'all':
        montage_keys = list(montage_dict.keys())
    else:
        montage_keys = params['montage'].split(',')
    
    # --- Setting Folder Name (for preds, metrics, plots) ---
    thres_val = params['thres']
    setting_val = params['setting']
    
    if setting_val:
        setting_folder_name = setting_val
    else:
        setting_folder_name = f"thres{thres_val:.1f}"
            
    print(f"Using setting: {setting_folder_name}")

    # =================================================================
    # STEP 1: Generate Probabilities (from run_sparcnet.py)
    # =================================================================
    print("\n--- STEP 1: Generating Probabilities ---")
    prob_folder = os.path.join(base_output_folder, 'prob')
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
        process_file_globals['prob_folder'] = prob_folder
        process_file_globals['force'] = force
        process_file_globals['montage_keys'] = montage_keys
        
        with tqdm(total=len(all_files), desc='Step 1/4: Processing EDFs') as pbar:
            Parallel(n_jobs=n_jobs)(delayed(process_file_sparcnet)(file_name) for file_name in all_files)
            pbar.update(len(all_files))
    else:
        print("Skipping Step 1, no files found.")

    # =================================================================
    # STEP 2: Generate Predictions
    # =================================================================
    print("\n--- STEP 2: Generating Predictions ---")
    pred_folder = os.path.join(base_output_folder, 'pred')
    
    process_file_pred_globals['pred_folder'] = pred_folder
    process_file_pred_globals['setting_folder'] = setting_folder_name
    process_file_pred_globals['force'] = force

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
        
        # Update globals for parallel function
        process_file_pred_globals['montage_key'] = m
        process_file_pred_globals['thres'] = current_thres

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
                metrics = compute_metrics(label, pred, prob, stride=feat_setting_sparcnet['stride'])
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
            all_p_metrics = patient_metrics(pred_file_df, feat_setting_sparcnet['stride'])
            if not all_p_metrics.empty:
                all_p_metrics.to_csv(out_file)

    # =================================================================
    # STEP 4: Generate Stats and Plots (from plot_metrics.py)
    # =================================================================
    print("\n--- STEP 4: Generating Stats and Plots ---")
    stats_folder = os.path.join(base_output_folder, 'stats', setting_folder_name)
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
