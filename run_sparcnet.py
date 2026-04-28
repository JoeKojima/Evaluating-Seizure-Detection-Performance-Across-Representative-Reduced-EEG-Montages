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
def load_edf_file(file_name):
    raw = mne.io.read_raw_edf(file_name, preload=True, verbose=0)
    fs = raw.info['sfreq']
    df = raw.to_data_frame().set_index('time')
    times = raw.times
    annotations = raw.annotations
    label = np.zeros(len(times)).astype(int)
    if annotations:
        for anno in annotations:
            sz_onset = anno['onset']
            sz_dura = anno['duration']
            sz_end = sz_onset + sz_dura
            label[(times >= sz_onset) & (times <= sz_end)] = 1
    label_df = pd.DataFrame({'time': times, 'labels': label})
    return raw, df, label_df, fs

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

def epiminder_simulate(raw):
    fs = raw.info['sfreq']
    n_times = raw.n_times
    new_ch_names = ['CP5', 'CP6', 'CP1', 'CP2']
    new_ch_types = ['eeg'] * len(new_ch_names)
    new_data = np.zeros((len(new_ch_names), n_times))
    new_info = mne.create_info(new_ch_names, fs, new_ch_types)
    new_raw = mne.io.RawArray(new_data, new_info)
    raw.add_channels([new_raw], force_update_info=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    raw.info['bads'] = ['CP5', 'CP6', 'CP1', 'CP2']
    raw.interpolate_bads(reset_bads=True)

    df = raw.to_data_frame().set_index('time')
    new_prepro = Preprocessor()
    new_prepro.fit({'samplingFreq': fs, 'samplingFreqRaw': fs, 'channelNames': df.columns, 'studyType': 'eeg', 'numberOfChannels': df.shape[1]})
    df = new_prepro.preprocess(df)
    filtered = df['filtered']
    cp5 = filtered['CP5'] - filtered['CP1']
    cp6 = filtered['CP6'] - filtered['CP2']
    columns = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
               'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
    data = df['BIPOLAR'][columns].copy() # Use .copy() to avoid SettingWithCopyWarning
    data.loc[:, columns] = 0
    data.loc[:, 'C3-P3'] = cp5
    data.loc[:, 'C4-P4'] = cp6
    return data

montage_dict = {
    'full': ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
             'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],
    'uneeg_left_front': ['F7-T3'],
    'uneeg_left_back': ['T3-T5'],
    'uneeg_right_front': ['F8-T4'],
    'uneeg_right_back': ['T4-T6'],
    'uneeg_right': ['F8-T4', 'T4-T6'],
    'uneeg_left': ['F7-T3', 'T3-T5'],
    'uneeg_bilateral4': ['F7-T3', 'T3-T5', 'F8-T4', 'T4-T6'],
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
    'epiminder_4': ['C3-P3', 'C4-P4', 'T3-T5', 'T4-T6'],
    'zero': []
}

# This dictionary is defined globally so process_file_sparcnet can access it
# It will be populated in the main block
process_file_globals = {
    'prob_folder': None,
    'force': False,
    'montage_keys': []
}

# =============================================================================
# ========= START OF BUG FIX: Replaced process_file_sparcnet ============
# =============================================================================

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
# ================== END OF BUG FIX SECTION ===================================
# =============================================================================


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
            sz_prob = prob_df.iloc[:, 1].values  # Assuming SZ is column 1 (0-indexed)
            label = prob_df.iloc[:, -1].values
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
    
    # Use LPD probability (column 1) as the seizure probability
    sz_prob = prob[:, 0]
    pred = (sz_prob >= thres).astype(int)
    
    pred_df = pd.DataFrame(np.vstack([sz_prob, pred]).T, columns=['sz_prob', 'pred'], index=prob_df.index)
    
    # These functions are from utils.py, which must be in 'pipeline_functions'
    pred_df['smoothed_pred'] = get_event_smoothed_pred(smooth_pred(pred_df['pred'].values))
    
    pred_df = pd.concat([pred_df, prob_df.iloc[:, -1]], axis=1)
    pred_df.to_csv(out_file)

# =============================================================================
# PART 3: Metric Calculation (from calc_metrics.py and get_metrics.py)
# =============================================================================

# --- Functions from calc_metrics.py ---
def extract_seiz_ranges(true_data):
    diff_data = np.diff(np.concatenate([[0], np.squeeze(true_data), [0]]))
    starts = np.where(diff_data == 1)[0]
    stops = np.where(diff_data == -1)[0]
    return list(zip(starts, stops))

def compute_metrics(true, pred, prob, stride=2):
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    seiz_ranges = extract_seiz_ranges(true)
    pred_seiz_ranges = extract_seiz_ranges(pred)

    metrics = {}
    metrics['total_dura'] = len(true) * stride / 60
    metrics['tn'] = tn / np.sum(true == 0) if np.sum(true == 0) > 0 else np.nan
    
    if np.any(true == 1):
        metrics['total_sz_dura'] = np.sum(true) * stride / 60
        metrics['avg_sz_dura'] = np.mean([(end - start) * stride / 60 for start, end in seiz_ranges])
        metrics['num_sz'] = len(seiz_ranges)
        sz_detected = np.array([np.sum(pred[start:end]) >= min(0.2 * (end - start), 10) for start, end in seiz_ranges])
        recall = np.sum(sz_detected) / len(sz_detected) if len(sz_detected) > 0 else np.nan
        metrics['auprc_sample'] = average_precision_score(true, prob)
        metrics['auroc_sample'] = roc_auc_score(true, prob)
        metrics['recall_event'] = recall
    else:
        for key in ['total_sz_dura', 'avg_sz_dura', 'num_sz', 'recall_event', 'auprc_sample', 'auroc_sample']:
            metrics[key] = np.nan
    
    # Calculate FP/hr based on non-seizure duration
    non_sz_dura_hr = (len(true) - np.sum(true)) * stride / 3600
    num_fp_events = len(pred_seiz_ranges) - np.sum([np.any(true[start:end]) for start, end in pred_seiz_ranges])
    metrics['fp'] = num_fp_events / non_sz_dura_hr if non_sz_dura_hr > 0 else np.nan
    
    metrics['balanced_acc'] = np.nanmean([metrics.get('recall_event', np.nan), metrics.get('tn', np.nan)])
    return metrics

# --- Functions from get_metrics.py ---
feat_setting_metrics = {
    'stride': int(2)
}

def patient_metrics(pred_file_df):
    all_metrics = []
    for key, group in pred_file_df.groupby('patient_id'):
        if group['is_sz'].sum() == 0:
            continue
        
        segment_metrics = []
        auc_prob = []
        auc_label = []
        auc_pred = []
        
        for _, row in group.iterrows():
            try:
                pred_df = pd.read_csv(row['pred_file'], index_col=0)
            except Exception as e:
                print(f"Error reading pred file {row['pred_file']}: {e}")
                continue
                
            label = pred_df.iloc[:, -1].values
            prob = pred_df['sz_prob'].values
            pred = pred_df['smoothed_pred'].values
            event_id = os.path.basename(row['pred_file'])[:-4]
            
            metrics = compute_metrics(label, pred, prob, stride=feat_setting_metrics['stride'])
            metric_row = pd.DataFrame([metrics], index=[event_id])
            segment_metrics.append(metric_row)
            
            auc_prob.extend(prob)
            auc_label.extend(label)
            auc_pred.extend(pred)
        
        if not segment_metrics:
            print(f"No segments processed for patient {key}")
            continue

        segment_metrics = pd.concat(segment_metrics, axis=0).sort_index()
        
        # Calculate patient-level metrics from concatenated samples
        patient_metrics_agg = compute_metrics(auc_label, auc_pred, auc_prob, stride=feat_setting_metrics['stride'])
        
        # Overwrite with averaged segment metrics as per script logic
        patient_metrics_agg['avg_sz_dura'] = np.nanmean(segment_metrics['avg_sz_dura'].values)
        patient_metrics_agg['num_sz'] = np.nansum(segment_metrics['num_sz'].values)
        patient_metrics_agg['recall_event'] = np.nanmean(segment_metrics['recall_event'].values)
        patient_metrics_agg['balanced_acc'] = np.nanmean(segment_metrics['balanced_acc'].values)
        patient_metrics_agg['fp'] = np.nanmean(segment_metrics['fp'].values)
        
        all_metrics.append(pd.DataFrame([patient_metrics_agg], index=[key]))
    
    if not all_metrics:
        return pd.DataFrame()
        
    all_metrics = pd.concat(all_metrics, axis=0).sort_index()
    return all_metrics


# =============================================================================
# PART 4: Plotting & Statistics (from plot_metrics.py)
# =============================================================================

metric_labels_plot = {
    'total_dura': 'Total Duration, min',
    'tn': 'Specificity',
    'total_sz_dura': 'Total Seizure Duration, min',
    'avg_sz_dura': 'Average Seizure Duration, min',
    'num_sz': 'Number of Seizure',
    'auroc_sample': 'AUROC',
    'auprc_sample': 'AUPRC',
    'recall_event': 'Recall',
    'balanced_acc': 'Balanced Accuracy',
    'fp': 'False Alarm'
}

plot_vars_plot = ['auroc_sample', 'auprc_sample', 'recall_event', 'tn', 'balanced_acc', 'fp']
plot_labels_plot = [metric_labels_plot.get(k, k) for k in plot_vars_plot]

multi_comp_plot = {
    'uneeg_left': ['full', 'uneeg_left_front', 'uneeg_left_back', 'uneeg_left', 'uneeg_vert_left', 'uneeg_diag_left_front', 'uneeg_diag_left_back'],
    'uneeg_right': ['full', 'uneeg_right_front', 'uneeg_right_back', 'uneeg_right', 'uneeg_vert_right', 'uneeg_diag_right_front', 'uneeg_diag_right_back'],
    'uneeg_bilateral': ['full', 'uneeg_bilateral_front2', 'uneeg_bilateral_back2', 'uneeg_bilateral4', 'uneeg_vert_bilateral', 'uneeg_diag_bilateral_front', 'uneeg_diag_bilateral_back'],
    'uneeg_horz_front': ['full', 'uneeg_left_front', 'uneeg_right_front', 'uneeg_bilateral_front2'],
    'uneeg_horz_back': ['full', 'uneeg_left_back', 'uneeg_right_back', 'uneeg_bilateral_back2'],
    'uneeg_horz': ['full', 'uneeg_left', 'uneeg_right', 'uneeg_bilateral4'],
    'uneeg_vert': ['full', 'uneeg_vert_left', 'uneeg_vert_right', 'uneeg_vert_bilateral'],
    'uneeg_diag': ['full', 'uneeg_diag_left_front', 'uneeg_diag_left_back', 'uneeg_diag_right_front', 'uneeg_diag_right_back', 'uneeg_diag_bilateral_front', 'uneeg_diag_bilateral_back'],
    'epiminder': ['full', 'epiminder_2', 'epiminder_4']
}

def flatten_tableone(df):
    try:
        new_col_names = {k: f'{k}' for k, v in df.loc['n'].to_dict(orient='records')[0].items() if v}
    except KeyError:
        new_col_names = {} # Handle case where 'n' might not be a multi-index
        
    if ('n', '') in df.index:
        df = df.drop(('n', ''), axis=0)
        
    new_rows = []
    for group in df.index.get_level_values(0).unique():
        if group == 'n':
            continue
        block = df.xs(group, level=0)
        if 'mean' in group or 'median' in group:
            block.index = [group]
            new_rows.append(block)
        else:
            label_row_data = [[""] * (df.shape[1] - 1) + [block.iloc[0, -1]]]
            label_row = pd.DataFrame(label_row_data, columns=df.columns, index=[group])
            block.index = ['    ' + str(idx) for idx in block.index]
            block.iloc[0, -1] = ''
            new_block = pd.concat([label_row, block])
            new_rows.append(new_block)
            
    if not new_rows:
        return pd.DataFrame(columns=df.columns) # Return empty if no data
        
    flat_df = pd.concat(new_rows)
    flat_df.index.name = None
    flat_df = flat_df.rename(new_col_names, axis=1)
    return flat_df

def plotting(long_df, fig_path):
    n_montage = len(long_df['montage'].unique())
    fig, ax = plt.subplots(figsize=(8 + n_montage, 6))
    sns.stripplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric'].isin(plot_vars_plot[:-1])],
        order=plot_vars_plot[:-1],
        size=3, jitter=0.2,
        dodge=True, alpha=.5, legend=False, zorder=0,
        palette=sns.color_palette(n_colors=n_montage),
        ax=ax
    )
    sns.pointplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric'].isin(plot_vars_plot[:-1])],
        order=plot_vars_plot[:-1],
        estimator='mean',
        dodge=0.4 + (n_montage - 2) * 0.1, linestyle="none", errorbar=("ci", 95),
        marker="_", markersize=15, markeredgewidth=3, zorder=1, errwidth=1, color='black',
        ax=ax
    )

    ax2 = ax.twinx()
    sns.stripplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric'] == plot_vars_plot[-1]],
        size=3, jitter=0.2, dodge=True, alpha=.5, legend=False, zorder=0,
        palette=sns.color_palette(n_colors=n_montage),
        ax=ax2
    )
    sns.pointplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric'] == plot_vars_plot[-1]],
        estimator='mean', dodge=0.4 + (n_montage - 2) * 0.1, linestyle="none",
        errorbar=("ci", 95), marker="_", markersize=15, markeredgewidth=3,
        zorder=1, errwidth=1, color='black',
        ax=ax2
    )

    ax.legend().remove()
    ax2.legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('Metric', fontsize=12)
    ax2.set_ylabel('False Alarm/h', fontsize=12)
    ax.set_xticks(ticks=list(range(len(plot_labels_plot))), labels=plot_labels_plot, fontsize=12, rotation=25)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def box_plotting(long_df, fig_path):
    n_montage = len(long_df['montage'].unique())
    fig, ax = plt.subplots(figsize=(6 + n_montage, 6))
    sns.boxplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric'].isin(plot_vars_plot[:-1])],
        order=plot_vars_plot[:-1],
        boxprops=dict(alpha=0.7),
        gap=0.1, width=0.7,
        dodge=True, legend=False, zorder=0,
        palette=sns.color_palette(n_colors=n_montage),
        ax=ax
    )
    ax2 = ax.twinx()
    sns.boxplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric'] == plot_vars_plot[-1]],
        boxprops=dict(alpha=0.7),
        gap=0.1, width=0.7,
        dodge=True, zorder=0,
        palette=sns.color_palette(n_colors=n_montage),
        ax=ax2
    )
    ax.legend().remove()
    handles, labels = ax2.get_legend_handles_labels()
    # Only show unique labels
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax2.legend(*zip(*unique), loc='upper right')
    
    ax.set_xlabel('')
    ax.set_ylabel('Metric', fontsize=12)
    ax2.set_ylabel('False Alarm/h', fontsize=12)
    ax.set_xticks(ticks=list(range(len(plot_labels_plot))), labels=plot_labels_plot, fontsize=12, rotation=25)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Run Full SPaRCNet Pipeline: Probs -> Preds -> Metrics -> Plots"
    )

    # --- Define Arguments ---
    parser.add_argument("-d", "--data_folder", type=str, default='../../haoershi/emu_dataset', help="Path to the emu_dataset folder (containing seizure/ and interictal/)")
    parser.add_argument("-o", "--output_folder", type=str, default='sparcnet_results', help="Main output folder for probs, preds, metrics, etc.")
    parser.add_argument("-p", "--patient_info", type=str, default='../../haoershi/emu_dataset/emu_patient_info.csv', help="Path to emu_patient_info.csv")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Comma-separated list of montages (or 'all')")
    parser.add_argument("--force", action='store_true', help="Force re-running all steps")
    parser.add_argument("-s", "--setting", type=str, default='', help="Threshold setting ('optimal' or leave blank for fixed)")
    parser.add_argument("-t", "--thres", type=float, default=0.5, help="Fixed threshold to use if 'optimal' is not set")
    parser.add_argument("--do_plot", action='store_true', help="Generate and save plots")
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
    
    if setting_val == 'optimal':
        setting_folder_name = 'thres_optimal'
    else:
        setting_folder_name = f"thres{thres_val:.1f}"
        if setting_val:
            setting_folder_name += f"_{setting_val}"
            
    print(f"Using setting: {setting_folder_name}")

    # =================================================================
    # STEP 1: Generate Probabilities (from run_sparcnet.py)
    # =================================================================
    print("\n--- STEP 1: Generating Probabilities ---")
    prob_folder = os.path.join(base_output_folder, 'prob')
    os.makedirs(prob_folder, exist_ok=True)
    
    for m in montage_keys:
        os.makedirs(os.path.join(prob_folder, m), exist_ok=True)

    sz_folder = os.path.join(base_data_folder, 'seizure')
    iic_folder = os.path.join(base_data_folder, 'interictal')
    
    try:
        sz_files = sorted(glob.glob(os.path.join(sz_folder, '*.edf')))
        iic_files = sorted(glob.glob(os.path.join(iic_folder, '*.edf')))
        all_files = sz_files + iic_files
        if not all_files:
            print(f"Warning: No .edf files found in {sz_folder} or {iic_folder}")
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
    # STEP 2: Generate Predictions (from sparcnet_pred.py)
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
            
        if setting_val == 'optimal':
            print("  Calculating optimal threshold...")
            current_thres = get_optimal_thres(prob_files)
            print(f"  Optimal threshold for {m}: {current_thres:.4f}")
        else:
            current_thres = thres_val
            
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
    
    # Check for patient map file
    if not os.path.exists(patient_map_file):
        print(f"Error: Patient info file not found at {patient_map_file}. Cannot proceed with metrics.")
    else:
        patient_map = pd.read_csv(patient_map_file, dtype={'patient_id': str})
        
        # --- Get eligible IDs from 'full' montage ---
        eligible_ids = []
        if 'full' in montage_keys:
            full_metric_file = os.path.join(metric_folder, 'full', 'segment_metrics.csv')
            if not os.path.exists(full_metric_file) or force:
                print("  Generating segment metrics for 'full' montage...")
                m = 'full'
                pred_files_full = glob.glob(os.path.join(pred_folder_setting, m, '*.csv'))
                os.makedirs(os.path.join(metric_folder, m), exist_ok=True)
                
                full_metrics_list = []
                for f in tqdm(pred_files_full, desc="    Calc 'full' seg_metrics"):
                    try:
                        pred_df = pd.read_csv(f, index_col=0)
                    except Exception as e:
                        print(f"Error reading {f}: {e}")
                        continue
                    label = pred_df.iloc[:, -1].values
                    prob = pred_df['sz_prob'].values
                    pred = pred_df['smoothed_pred'].values
                    event_id = os.path.basename(f)[:-4]
                    metrics = compute_metrics(label, pred, prob, stride=feat_setting_metrics['stride'])
                    metric_row = pd.DataFrame([metrics], index=[event_id])
                    full_metrics_list.append(metric_row)
                
                if full_metrics_list:
                    full_metrics = pd.concat(full_metrics_list, axis=0).sort_index()
                    full_metrics.to_csv(full_metric_file)
                else:
                    full_metrics = pd.DataFrame()
            else:
                print("  Loading existing segment metrics for 'full' montage...")
                full_metrics = pd.read_csv(full_metric_file, index_col=0)
            
            if not full_metrics.empty:
                eligible_ids = full_metrics[full_metrics['recall_event'] != 0].index.to_list()
            print(f"  Found {len(eligible_ids)} eligible seizure events detected by 'full' montage.")

        else:
            print("Warning: 'full' montage not in list. Cannot filter based on its detections.")
            
        # --- Calculate patient metrics for all montages ---
        for m in tqdm(montage_keys, desc='Step 3/4: Calculating Metrics'):
            os.makedirs(os.path.join(metric_folder, m), exist_ok=True)
            out_file = os.path.join(metric_folder, m, 'patient_metrics.csv')
            
            if not force and os.path.exists(out_file) and os.path.exists(out_file.replace('.csv', '_filtered.csv')):
                print(f"  Metrics for {m} already exist. Skipping.")
                continue

            pred_files = glob.glob(os.path.join(pred_folder_setting, m, '*.csv'))
            if not pred_files:
                print(f"  No prediction files found for {m}. Skipping.")
                continue

            pred_file_df = pd.DataFrame(pred_files, columns=['pred_file'])
            pred_file_df['emu_id'] = pred_file_df['pred_file'].apply(lambda x: os.path.basename(x).split('_')[0])
            pred_file_df['event_id'] = pred_file_df['pred_file'].apply(lambda x: os.path.basename(x)[:-4])
            pred_file_df['is_sz'] = pred_file_df['event_id'].apply(lambda x: 'seizure' in x or 'event' in x.lower()) # Broader check
            
            pred_file_df = pred_file_df.merge(patient_map, on='emu_id', how='left')
            pred_file_df['patient_id'] = pred_file_df['patient_id'].fillna(pred_file_df['emu_id'])
            pred_file_df['is_detected'] = pred_file_df['event_id'].apply(lambda x: x in eligible_ids)

            # Calculate metrics for all files
            all_p_metrics = patient_metrics(pred_file_df)
            if not all_p_metrics.empty:
                all_p_metrics.to_csv(out_file)

            # Calculate metrics for filtered files
            if eligible_ids:
                filtered_p_metrics = patient_metrics(pred_file_df[pred_file_df['is_detected']])
                if not filtered_p_metrics.empty:
                    filtered_p_metrics.to_csv(out_file.replace('.csv', '_filtered.csv'))
            else:
                print(f"  No eligible IDs for filtered metrics for {m}.")

    # =================================================================
    # STEP 4: Generate Stats and Plots (from plot_metrics.py)
    # =================================================================
    print("\n--- STEP 4: Generating Stats and Plots ---")
    figure_folder = os.path.join(base_output_folder, 'figures', setting_folder_name)
    stats_folder = os.path.join(base_output_folder, 'stats', setting_folder_name)
    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)

    file_names = {'': 'patient_metrics.csv', '_filtered': 'patient_metrics_filtered.csv'}
    
    if not os.path.exists(patient_map_file):
        print(f"Error: Patient info file not found at {patient_map_file}. Cannot generate stats by type.")
    else:
        patient_map = pd.read_csv(patient_map_file, dtype={'patient_id': str})

        for plot_setting, file_name in file_names.items():
            print(f"\n  Processing stats for: {plot_setting or 'all_events'}")
            
            all_metrics_list = []
            for m in montage_keys:
                metric_file = os.path.join(metric_folder, m, file_name)
                if not os.path.exists(metric_file):
                    print(f"    Metric file not found, skipping: {metric_file}")
                    continue
                metrics = pd.read_csv(metric_file, index_col=0)
                metrics['montage'] = m
                all_metrics_list.append(metrics)
            
            if not all_metrics_list:
                print(f"    No metrics found for {plot_setting}. Skipping stats/plots.")
                continue

            all_metrics = pd.concat(all_metrics_list, axis=0).reset_index().rename(columns={'index': 'patient_id'})
            all_metrics['patient_id'] = all_metrics['patient_id'].astype(str)

            # --- Generate Plots ---
            if params['do_plot']:
                print("    Generating plots...")
                for m in tqdm(montage_keys, desc='    Plotting montages'):
                    if m == 'full':
                        continue
                    fig_path = os.path.join(figure_folder, f"{m}{plot_setting}.png")
                    tmp_metrics = all_metrics[all_metrics['montage'].isin(['full', m])]
                    if tmp_metrics.empty: continue
                    long_df = pd.melt(tmp_metrics, id_vars=['montage'], var_name='metric')
                    plotting(long_df, fig_path)

                # Multi comps
                for multi, multi_montages in tqdm(multi_comp_plot.items(), desc='    Plotting multi-comp'):
                    long_df = pd.melt(all_metrics[all_metrics['montage'].isin(multi_montages)], id_vars=['montage'], var_name='metric')
                    if long_df.empty: continue
                    box_plotting(long_df, os.path.join(figure_folder, f"multi_{multi}{plot_setting}.png"))

            # --- Generate Stats Tables ---
            print("    Generating stats tables...")
            try:
                # Overall comparison
                table1 = TableOne(all_metrics, columns=plot_vars_plot,
                                  groupby='montage', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels_plot)
                table1_df = table1.tableone.get('Grouped by montage')
                if table1_df is not None:
                    flatten_tableone(table1_df).T.to_csv(os.path.join(stats_folder, f'comparison{plot_setting}.csv'))

                # By epilepsy type
                metrics_with_info = all_metrics.merge(patient_map, on='patient_id', how='left')
                metrics_with_info = metrics_with_info[~metrics_with_info['epilepsy_type'].isna()]
                metrics_with_info['tmp_group'] = metrics_with_info['epilepsy_type'] + '_' + metrics_with_info['montage']
                table_type = TableOne(metrics_with_info, columns=plot_vars_plot,
                                      groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels_plot)
                table_type_df = table_type.tableone.get('Grouped by tmp_group')
                if table_type_df is not None:
                    flatten_tableone(table_type_df).T.to_csv(os.path.join(stats_folder, f'comparison_by_type{plot_setting}.csv'))

                # By laterality
                metrics_with_info = all_metrics.merge(patient_map, on='patient_id', how='left')
                metrics_with_info = metrics_with_info[~metrics_with_info['laterality'].isna()]
                metrics_with_info['tmp_group'] = metrics_with_info['laterality'] + '_' + metrics_with_info['montage']
                table_lat = TableOne(metrics_with_info, columns=plot_vars_plot,
                                     groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels_plot)
                table_lat_df = table_lat.tableone.get('Grouped by tmp_group')
                if table_lat_df is not None:
                    flatten_tableone(table_lat_df).T.to_csv(os.path.join(stats_folder, f'comparison_by_laterality{plot_setting}.csv'))

                # By location
                metrics_with_info = all_metrics.merge(patient_map, on='patient_id', how='left')
                metrics_with_info = metrics_with_info[~metrics_with_info['location'].isna()]
                metrics_with_info['tmp_group'] = metrics_with_info['location'] + '_' + metrics_with_info['montage']
                table_loc = TableOne(metrics_with_info, columns=plot_vars_plot,
                                     groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels_plot)
                table_loc_df = table_loc.tableone.get('Grouped by tmp_group')
                if table_loc_df is not None:
                    flatten_tableone(table_loc_df).T.to_csv(os.path.join(stats_folder, f'comparison_by_location{plot_setting}.csv'))

            except Exception as e:
                print(f"    Error generating TableOne stats: {e}")

    print("\n--- Pipeline Complete ---")
