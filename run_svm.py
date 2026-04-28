# run_svm_detection.py - REVISED for Patient-Specific Baseline Training
# Now includes Youden's J Optimal Thresholding

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
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score
from tableone import TableOne
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
# MNE logging level to avoid clutter
mne.set_log_level('ERROR')

# --- 1. SETUP PIPELINE PATHS ---
WORKING_DIR = os.path.abspath("")
PIPELINE_FUNCS_PATH = os.path.join(WORKING_DIR, 'pipeline_functions')

sys.path.append(WORKING_DIR)
sys.path.append(PIPELINE_FUNCS_PATH)

warnings.filterwarnings('ignore')
os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"
os.makedirs(".matplotlib_cache", exist_ok=True)

# --- 2. CORE UTILITY IMPORTS ---
try:
    from utils import Preprocessor 
    from utils_baseline import (
        extract_features, train_one_class_svm, compute_novelty_scores,
        estimate_outlier_fraction, detect_seizure, apply_persistence
    )
except ImportError as e:
    print(f"FATAL ERROR: Could not import required pipeline components. Missing: {e}")
    sys.exit(1)


# --- 3. HELPER FUNCTIONS (LOCAL DEFINITIONS) ---

def load_edf_file(file_name, crop_max=None):
    raw = mne.io.read_raw_edf(file_name, preload=True, verbose=0)
    
    if crop_max is not None and crop_max < raw.times[-1]:
        raw.crop(tmax=crop_max)
        
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
            mask = (times >= sz_onset) & (times <= sz_end)
            if mask.any():
                label[mask] = 1
                
    label_df = pd.DataFrame({'time': times, 'labels': label})
    return raw, df, label_df, fs

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
    
    data = pd.DataFrame(index=df.index)
    data['C3-P3'] = cp5
    data['C4-P4'] = cp6
    return data

def get_optimal_thres(prob_files):
    """Calculates optimal threshold using Youden's J statistic (TPR - FPR).
    Only uses seizure files — IIC files have all-zero labels and would
    degenerate the ROC curve, pushing the threshold to 1.
    """
    all_prob = []
    all_label = []
    for f in prob_files:
        # Skip non-seizure files
        basename = os.path.basename(f).lower()
        if 'seizure' not in basename and 'event' not in basename:
            continue
        try:
            prob_df = pd.read_csv(f, index_col=0)
            all_prob.extend(prob_df['sz_prob'].values)
            all_label.extend(prob_df['label'].values)
        except Exception:
            continue
            
    if not all_label or not np.any(np.array(all_label) == 1):
        print("    Warning: No positive labels found in seizure files. Using fallback threshold 0.5.")
        return 0.5
        
    fpr, tpr, thres = roc_curve(all_label, all_prob)
    # Youden's J statistic
    opt_thres = thres[np.argmax(tpr - fpr)]
    return opt_thres

# --- 4. CONSTANTS ---

MONTAGE_DICT = {
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
FS_FEAT = 2.0
TRAIN_DURATION_SEC = 60

# --- 5. METRIC & PLOTTING FUNCTIONS ---

def extract_seiz_ranges(true_data):
    diff_data = np.diff(np.concatenate([[0], np.squeeze(true_data), [0]]))
    starts = np.where(diff_data == 1)[0]
    stops = np.where(diff_data == -1)[0]
    return list(zip(starts, stops))

def compute_metrics(true, pred, prob, stride=2):
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    tn = np.sum((pred == 0) & (true == 0))
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
    
    non_sz_dura_hr = (len(true) - np.sum(true)) * stride / 3600
    num_fp_events = len(pred_seiz_ranges) - np.sum([np.any(true[start:end]) for start, end in pred_seiz_ranges])
    metrics['fp'] = num_fp_events / non_sz_dura_hr if non_sz_dura_hr > 0 else np.nan
    
    metrics['balanced_acc'] = np.nanmean([metrics.get('recall_event', np.nan), metrics.get('tn', np.nan)])
    return metrics

feat_setting_metrics = {'stride': int(2)}

def patient_metrics(pred_file_df):
    all_metrics = []
    for patient_id, group in pred_file_df.groupby('patient_id'):
        segment_metrics = []
        auc_prob, auc_label, auc_pred = [], [], []
        if group['is_sz'].sum() == 0: continue
        
        for _, row in group.iterrows():
            try:
                pred_df = pd.read_csv(row['pred_file'], index_col=0)
            except Exception: continue
            label = pred_df['label'].values
            prob = pred_df['sz_prob'].values
            pred = pred_df['smoothed_pred'].values
            metrics = compute_metrics(label, pred, prob, stride=feat_setting_metrics['stride'])
            metric_row = pd.DataFrame([metrics], index=[row['event_id']])
            segment_metrics.append(metric_row)
            auc_prob.extend(prob)
            auc_label.extend(label)
            auc_pred.extend(pred)
        
        if not segment_metrics: continue
        segment_metrics = pd.concat(segment_metrics, axis=0).sort_index()
        patient_metrics_agg = compute_metrics(auc_label, auc_pred, auc_prob, stride=feat_setting_metrics['stride'])
        patient_metrics_agg['avg_sz_dura'] = np.nanmean(segment_metrics['avg_sz_dura'].values)
        patient_metrics_agg['num_sz'] = np.nansum(segment_metrics['num_sz'].values)
        patient_metrics_agg['recall_event'] = np.nanmean(segment_metrics['recall_event'].values) 
        patient_metrics_agg['balanced_acc'] = np.nanmean(segment_metrics['balanced_acc'].values)
        patient_metrics_agg['fp'] = np.nanmean(segment_metrics['fp'].values)
        all_metrics.append(pd.DataFrame([patient_metrics_agg], index=[patient_id]))
    
    if not all_metrics: return pd.DataFrame()
    all_metrics = pd.concat(all_metrics, axis=0).sort_index()
    all_metrics.index.name = 'patient_id' 
    return all_metrics

# --- Plotting Dictionaries ---
multi_comp = {
    'uneeg_left': ['full', 'uneeg_left_front', 'uneeg_left_back', 'uneeg_left', 'uneeg_vert_left', 'uneeg_diag_left_front', 'uneeg_diag_left_back'],
    'uneeg_right': ['full', 'uneeg_right_front', 'uneeg_right_back', 'uneeg_right', 'uneeg_vert_right', 'uneeg_diag_right_front', 'uneeg_diag_right_back'],
    'uneeg_bilateral': ['full', 'uneeg_bilateral_front2', 'uneeg_bilateral_back2', 'uneeg_bilateral4', 'uneeg_vert_bilateral', 'uneeg_diag_bilateral_front', 'uneeg_diag_bilateral_back'],
    'epiminder': ['full', 'epiminder_2', 'epiminder_4']
}

metric_labels_plot = {
    'total_dura': 'Total Duration, min', 'tn': 'Specificity', 'total_sz_dura': 'Total Seizure Duration, min',
    'avg_sz_dura': 'Average Seizure Duration, min', 'num_sz': 'Number of Seizure', 'auroc_sample': 'AUROC',
    'auprc_sample': 'AUPRC', 'recall_event': 'Recall', 'balanced_acc': 'Balanced Accuracy', 'fp': 'False Alarm'
}
plot_vars_plot = ['auroc_sample', 'auprc_sample', 'recall_event', 'tn', 'balanced_acc', 'fp']
plot_labels_plot = [metric_labels_plot.get(k, k) for k in plot_vars_plot]

def flatten_tableone(df):
    try:
        new_col_names = {k: f'{k}' for k, v in df.loc['n'].to_dict(orient='records')[0].items() if v}
    except KeyError:
        new_col_names = {}
    if ('n', '') in df.index: df = df.drop(('n', ''), axis=0)
    new_rows = []
    for group in df.index.get_level_values(0).unique():
        if group == 'n': continue
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
    if not new_rows: return pd.DataFrame(columns=df.columns)
    flat_df = pd.concat(new_rows)
    flat_df.index.name = None
    flat_df = flat_df.rename(new_col_names, axis=1)
    return flat_df

def plotting(long_df, fig_path):
    n_montage = len(long_df['montage'].unique())
    if n_montage < 2: return 
    
    fig, ax = plt.subplots(figsize=(8 + n_montage, 6))
    sns.stripplot(x='metric', y='value', hue='montage', data=long_df[long_df['metric'].isin(plot_vars_plot[:-1])], order=plot_vars_plot[:-1], size=3, jitter=0.2, dodge=True, alpha=.5, legend=False, zorder=0, palette=sns.color_palette(n_colors=n_montage), ax=ax)
    sns.pointplot(x='metric', y='value', hue='montage', data=long_df[long_df['metric'].isin(plot_vars_plot[:-1])], order=plot_vars_plot[:-1], estimator='mean', dodge=0.4 + (n_montage - 2) * 0.1, linestyle="none", errorbar=("ci", 95), marker="_", markersize=15, markeredgewidth=3, zorder=1, errwidth=1, color='black', ax=ax)
    ax2 = ax.twinx()
    sns.stripplot(x='metric', y='value', hue='montage', data=long_df[long_df['metric'] == plot_vars_plot[-1]], size=3, jitter=0.2, dodge=True, alpha=.5, legend=False, zorder=0, palette=sns.color_palette(n_colors=n_montage), ax=ax2)
    sns.pointplot(x='metric', y='value', hue='montage', data=long_df[long_df['metric'] == plot_vars_plot[-1]], estimator='mean', dodge=0.4 + (n_montage - 2) * 0.1, linestyle="none", errorbar=("ci", 95), marker="_", markersize=15, markeredgewidth=3, zorder=1, errwidth=1, color='black', ax=ax2)
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
    if n_montage < 2: return 
    
    fig, ax = plt.subplots(figsize=(6 + n_montage, 6))
    sns.boxplot(x='metric', y='value', hue='montage', data=long_df[long_df['metric'].isin(plot_vars_plot[:-1])], order=plot_vars_plot[:-1], boxprops=dict(alpha=0.7), gap=0.1, width=0.7, dodge=True, legend=False, zorder=0, palette=sns.color_palette(n_colors=n_montage), ax=ax)
    ax2 = ax.twinx()
    sns.boxplot(x='metric', y='value', hue='montage', data=long_df[long_df['metric'] == plot_vars_plot[-1]], boxprops=dict(alpha=0.7), gap=0.1, width=0.7, dodge=True, zorder=0, palette=sns.color_palette(n_colors=n_montage), ax=ax2)
    ax.legend().remove()
    handles, labels = ax2.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax2.legend(*zip(*unique), loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('Metric', fontsize=12)
    ax2.set_ylabel('False Alarm/h', fontsize=12)
    ax.set_xticks(ticks=list(range(len(plot_labels_plot))), labels=plot_labels_plot, fontsize=12, rotation=25)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# --- 6. SVM DETECTION CORE LOGIC (REFACTORED) ---

def _get_montage_data(df, raw, montage_key):
    montage_processor = MONTAGE_DICT[montage_key]
    
    fs = raw.info['sfreq']
    prepro = Preprocessor()
    prepro.fit({'samplingFreq':fs, 'samplingFreqRaw':fs, 'channelNames':df.columns, 'studyType':'eeg', 'numberOfChannels':df.shape[1]})
    preprocessed = prepro.preprocess(df)

    if isinstance(montage_processor, list):
        data_df = preprocessed['BIPOLAR'].copy()
        valid_cols = [c for c in montage_processor if c in data_df.columns]
        data_df = data_df[valid_cols].copy()
    elif 'simulate' in montage_key:
        data_df = epiminder_simulate(raw)
    else:
        data_df = montage_processor(preprocessed)
        
    return data_df

def process_svm_event_core(file_name, montage_key, output_prob_folder, force_rerun, calibration_file=None):
    warnings.filterwarnings('ignore')
    
    output_file = os.path.join(output_prob_folder, montage_key, os.path.basename(file_name).replace('.edf', '.csv'))
    if os.path.exists(output_file) and not force_rerun:
        return

    use_external_baseline = (calibration_file is not None) and (os.path.exists(calibration_file)) and (calibration_file != file_name)
    train_file_path = calibration_file if use_external_baseline else file_name

    try:
        raw_train, df_train, _, fs = load_edf_file(train_file_path, crop_max=TRAIN_DURATION_SEC + 10)
        train_df_montage = _get_montage_data(df_train, raw_train, montage_key)
        
        eeg_data_train = train_df_montage.values
        train_samples = int(TRAIN_DURATION_SEC * fs)
        
        # --- FIX: Check if file is too short OR has 0 valid channels ---
        if eeg_data_train.shape[0] < train_samples or eeg_data_train.shape[1] == 0: return

        X_train_list = [extract_features(eeg_data_train[:train_samples, i], fs=fs) for i in range(eeg_data_train.shape[1])]
        X_train = np.nan_to_num(np.hstack(X_train_list))
        
        if X_train.size == 0: return
        clf = train_one_class_svm(X_train)

    except Exception as e:
        print(f"Error training on {train_file_path}: {e}")
        return

    try:
        raw_test, df_test, label_df, fs_test = load_edf_file(file_name) 
        test_df_montage = _get_montage_data(df_test, raw_test, montage_key)
        eeg_data_test = test_df_montage.values
        
        # --- FIX: Check if test file has 0 valid channels ---
        if eeg_data_test.shape[1] == 0: return

        X_test_list = [extract_features(eeg_data_test[:, i], fs=fs_test) for i in range(eeg_data_test.shape[1])]
        X_test_full = np.nan_to_num(np.hstack(X_test_list))

        if X_test_full.size == 0: return

        if not use_external_baseline:
            feat_idx_start = int(TRAIN_DURATION_SEC * FS_FEAT)
            if feat_idx_start >= X_test_full.shape[0]: return 
            X_test_to_predict = X_test_full[feat_idx_start:]
            time_offset_sec = TRAIN_DURATION_SEC
        else:
            X_test_to_predict = X_test_full
            time_offset_sec = 0

        y_pred = compute_novelty_scores(clf, X_test_to_predict)
        nu_hat = estimate_outlier_fraction(y_pred, n=10)
        
        smoothing_sigma = 2 * int(len(nu_hat) / 1000 + 1)
        nu_filt = np.round(gaussian_filter1d(nu_hat, smoothing_sigma), 100)
        
        start_time_s = df_test.index.min() if not df_test.empty else 0
        time_vals = start_time_s + time_offset_sec + 0.5 + np.arange(0, len(nu_filt)) * 0.5
        
        label_time = label_df.set_index('time')['labels']
        label = label_time.reindex(time_vals, method='nearest').values

        pred_df_final = pd.DataFrame({
            'SZ': nu_filt, 'LPD': 0, 'GPD': 0, 'LRDA': 0, 'GRDA': 0, 'OTHER': 0, 
            'sz_prob': nu_filt,
            'pred': 0, 
            'smoothed_pred': 0, 
            'label': label              
        }, index=time_vals)
        
        pred_df_final.index = pd.to_datetime(pred_df_final.index, unit='s')
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pred_df_final.to_csv(output_file)

    except Exception as e:
        print(f"Error predicting on {file_name}: {e}")
        return


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SVM Baseline Pipeline (Patient-Specific).")
    parser.add_argument("-d", "--data_folder", type=str, default='../../haoershi/emu_dataset', help="Path to emu_dataset")
    parser.add_argument("-o", "--output_folder", type=str, default='svm_baseline_results', help="Output folder")
    parser.add_argument("-p", "--patient_info", type=str, default='../../haoershi/emu_dataset/dataset_admission_info.csv', help="Patient info CSV")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Montages")
    parser.add_argument("-s", "--setting", type=str, default='optimal', help="Setting for threshold ('optimal' or 'fixed')")
    parser.add_argument("-t", "--thres", type=float, default=0.9, help="Fixed threshold to use if setting is not optimal")
    parser.add_argument("--n_jobs", type=int, default=40, help="Parallel jobs")
    parser.add_argument("--do_plot", action='store_true', help="Generate plots")
    parser.add_argument("--force", action='store_true', help="Force rerun")
    
    params = vars(parser.parse_args())

    base_data_folder = params['data_folder']
    base_output_folder = params['output_folder']
    patient_map_file = params['patient_info']
    n_jobs = params['n_jobs']
    
    # Establish dynamic settings folder name based on Youden's optimization or fixed thres
    if params['setting'] == 'optimal':
        setting_folder_name = 'thres_optimal'
    else:
        setting_folder_name = f"thres{params['thres']:.1f}"

    prob_folder = os.path.join(base_output_folder, 'prob')
    pred_folder_setting = os.path.join(base_output_folder, 'pred', setting_folder_name)
    metric_folder_setting = os.path.join(base_output_folder, 'metrics', setting_folder_name)
    figure_folder_setting = os.path.join(base_output_folder, 'figures', setting_folder_name)
    stats_folder_setting = os.path.join(base_output_folder, 'stats', setting_folder_name)
    
    if params['montage'] == 'all':
        montages_to_run = list(MONTAGE_DICT.keys())
    else:
        montages_to_run = [m.strip() for m in params['montage'].split(',') if m.strip() in MONTAGE_DICT]

    sz_folder = os.path.join(base_data_folder, 'seizure')
    iic_folder = os.path.join(base_data_folder, 'interictal')
    
    sz_files = sorted(glob.glob(os.path.join(sz_folder, '*.edf')))
    iic_files = sorted(glob.glob(os.path.join(iic_folder, '*.edf')))
    all_files = sz_files + iic_files
    
    if not all_files:
        print(f"FATAL ERROR: No EDF files found in {sz_folder} or {iic_folder}.")
        sys.exit(1)

    print("Building patient calibration map...")
    patient_calibration_map = {}
    
    for f in sorted(iic_files):
        filename = os.path.basename(f)
        patient_id = filename.split('_')[0]
        
        if patient_id not in patient_calibration_map:
            patient_calibration_map[patient_id] = f
            
    print(f"Found calibration files for {len(patient_calibration_map)} patients.")

    print("\n--- STEP 1: Generating Probabilities (Patient-Specific SVM) ---")
    
    all_tasks = []
    for m in montages_to_run:
        os.makedirs(os.path.join(prob_folder, m), exist_ok=True)
        
        for file_name in all_files:
            curr_fname = os.path.basename(file_name)
            curr_pid = curr_fname.split('_')[0]
            
            calib_file = patient_calibration_map.get(curr_pid, None)
            
            all_tasks.append(delayed(process_svm_event_core)(
                file_name, m, prob_folder, params['force'], calib_file
            ))

    with tqdm(total=len(all_tasks), desc='Step 1/4: Processing SVM Events') as pbar:
        Parallel(n_jobs=n_jobs)(all_tasks)
        pbar.update(len(all_tasks))
        
    print(f"Probabilities generated in {prob_folder}.")

    print("\n--- STEP 2: Generating Predictions ---")
    
    def pred_wrapper_svm(file_name, montage_key, pred_folder_out, thres_val):
        warnings.filterwarnings('ignore')
        out_file = os.path.join(pred_folder_out, montage_key, os.path.basename(file_name))
        if not params['force'] and os.path.exists(out_file): return
        try:
            prob_df = pd.read_csv(file_name, index_col=0)
        except Exception: return

        sz_prob = prob_df['sz_prob'].values
        # Using SVM specific thresholding and persistence logic
        pred = detect_seizure(sz_prob, threshold=thres_val)
        
        pred_df = pd.DataFrame({'sz_prob': sz_prob, 'pred': pred}, index=prob_df.index)
        pred_df['smoothed_pred'] = apply_persistence(pred) 
        pred_df['label'] = prob_df['label'].values 
        
        pred_df.to_csv(out_file)

    for m in montages_to_run:
        prob_files_m = glob.glob(os.path.join(prob_folder, m, '*.csv'))
        if not prob_files_m: continue
        
        if params['setting'] == 'optimal':
            print(f"  Calculating optimal Youden's J threshold for {m}...")
            current_thres = get_optimal_thres(prob_files_m)
            print(f"  Optimal threshold for {m}: {current_thres:.4f}")
        else:
            current_thres = params['thres']

        os.makedirs(os.path.join(pred_folder_setting, m), exist_ok=True)
        with tqdm(total=len(prob_files_m), desc=f'Step 2/4: Predicting [{m}]') as pbar:
            tasks = [delayed(pred_wrapper_svm)(f, m, pred_folder_setting, current_thres) for f in prob_files_m]
            Parallel(n_jobs=n_jobs)(tasks)
            pbar.update(len(prob_files_m))

    print("\n--- STEP 3: Calculating Metrics ---")

    patient_map = pd.DataFrame()
    if os.path.exists(patient_map_file):
        try:
            patient_map = pd.read_csv(patient_map_file, dtype={'patient_id': str, 'admission_id': str})
            if 'admission_id' in patient_map.columns:
                patient_map['emu_id'] = patient_map['admission_id']
            elif 'patient_id' in patient_map.columns and 'emu_id' not in patient_map.columns:
                patient_map['emu_id'] = patient_map['patient_id']
            patient_map['emu_id'] = patient_map['emu_id'].astype(str)
        except Exception: pass
    
    eligible_ids = []
    if 'full' in montages_to_run:
        full_pred_files = glob.glob(os.path.join(pred_folder_setting, 'full', '*.csv'))
        full_metrics_list = []
        for f in tqdm(full_pred_files, desc="  Calc 'full' seg_metrics for filtering"):
            try:
                pred_df = pd.read_csv(f, index_col=0)
                label = pred_df['label'].values
                prob = pred_df['sz_prob'].values
                pred = pred_df['smoothed_pred'].values
                metrics = compute_metrics(label, pred, prob, stride=feat_setting_metrics['stride'])
                full_metrics_list.append(pd.DataFrame([metrics], index=[os.path.basename(f)[:-4]]))
            except Exception: continue
            
        if full_metrics_list:
            full_metrics = pd.concat(full_metrics_list, axis=0).sort_index()
            is_sz_file = full_metrics.index.to_series().str.contains('seizure|event', case=False)
            eligible_ids = full_metrics[is_sz_file & (full_metrics['recall_event'] != 0)].index.to_list()
        
    for m in tqdm(montages_to_run, desc='Step 3/4: Calculating Patient Metrics'):
        os.makedirs(os.path.join(metric_folder_setting, m), exist_ok=True)
        out_file = os.path.join(metric_folder_setting, m, 'patient_metrics.csv')
        pred_files = glob.glob(os.path.join(pred_folder_setting, m, '*.csv'))
        if not pred_files: continue

        pred_file_df = pd.DataFrame(pred_files, columns=['pred_file'])
        pred_file_df['event_id'] = pred_file_df['pred_file'].apply(lambda x: os.path.basename(x)[:-4])
        pred_file_df['emu_id'] = pred_file_df['event_id'].apply(lambda x: x.split('_')[0])
        
        if not patient_map.empty:
            pred_file_df = pred_file_df.merge(patient_map[['emu_id', 'patient_id']], on='emu_id', how='left')
            pred_file_df['patient_id'] = pred_file_df['patient_id'].fillna(pred_file_df['emu_id'])
        else:
            pred_file_df['patient_id'] = pred_file_df['emu_id'] 

        pred_file_df['is_detected'] = pred_file_df['event_id'].apply(lambda x: x in eligible_ids)
        pred_file_df['is_sz'] = pred_file_df['event_id'].apply(lambda x: 'seizure' in x or 'event' in x.lower())
        
        all_p_metrics = patient_metrics(pred_file_df)
        if not all_p_metrics.empty:
            all_p_metrics.to_csv(out_file)

        if eligible_ids:
            filtered_p_metrics = patient_metrics(pred_file_df[pred_file_df['is_detected']])
            if not filtered_p_metrics.empty:
                filtered_p_metrics.to_csv(out_file.replace('.csv', '_filtered.csv'))

    if params['do_plot']:
        print("\n--- STEP 4: Generating Stats and Plots ---")
        os.makedirs(figure_folder_setting, exist_ok=True)
        os.makedirs(stats_folder_setting, exist_ok=True)
        file_names = {'': 'patient_metrics.csv', '_filtered': 'patient_metrics_filtered.csv'}
        
        patient_map = pd.DataFrame()
        if os.path.exists(patient_map_file):
            try:
                patient_map = pd.read_csv(patient_map_file, dtype={'patient_id': str, 'admission_id': str})
                if 'admission_id' in patient_map.columns:
                    patient_map['emu_id'] = patient_map['admission_id']
            except Exception: pass
            
        for plot_setting, file_name in file_names.items():
            all_metrics_list = []
            for m in montages_to_run:
                metric_file = os.path.join(metric_folder_setting, m, file_name)
                if os.path.exists(metric_file):
                    metrics = pd.read_csv(metric_file, index_col=0)
                    metrics['montage'] = m
                    all_metrics_list.append(metrics)
            if not all_metrics_list: continue
            all_metrics = pd.concat(all_metrics_list, axis=0).reset_index().rename(columns={'index': 'patient_id'})
            all_metrics['patient_id'] = all_metrics['patient_id'].astype(str)
            
            for m in tqdm(montages_to_run, desc='    Plotting montages'):
                if m == 'full': continue
                fig_path = os.path.join(figure_folder_setting, f"{m}{plot_setting}.png")
                tmp_metrics = all_metrics[all_metrics['montage'].isin(['full', m])]
                if tmp_metrics.empty or tmp_metrics['montage'].nunique() < 2: continue 
                long_df = pd.melt(tmp_metrics, id_vars=['montage'], var_name='metric')
                plotting(long_df, fig_path)
                
            for multi, multi_montages in tqdm(multi_comp.items(), desc='    Plotting multi-comp'):
                valid_montages = [m for m in multi_montages if m in montages_to_run]
                if len(valid_montages) < 2: continue
                long_df = pd.melt(all_metrics[all_metrics['montage'].isin(valid_montages)], id_vars=['montage'], var_name='metric')
                if long_df.empty or long_df['montage'].nunique() < 2: continue 
                box_plotting(long_df, os.path.join(figure_folder_setting, f"multi_{multi}{plot_setting}.png"))
                
            try:
                table1 = TableOne(all_metrics, columns=plot_vars_plot, groupby='montage', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels_plot)
                table1_df = table1.tableone.get('Grouped by montage')
                if table1_df is not None:
                    flatten_tableone(table1_df).T.to_csv(os.path.join(stats_folder_setting, f'comparison{plot_setting}.csv'))
                if not patient_map.empty and 'epilepsy_type' in patient_map.columns and 'laterality' in patient_map.columns and 'location' in patient_map.columns:
                    metrics_with_info = all_metrics.merge(patient_map, on='patient_id', how='left')
                    metrics_type = metrics_with_info[~metrics_with_info['epilepsy_type'].isna()]
                    if not metrics_type.empty:
                        metrics_type['tmp_group'] = metrics_type['epilepsy_type'] + '_' + metrics_type['montage']
                        table_type = TableOne(metrics_type, columns=plot_vars_plot, groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels_plot)
                        table_type_df = table_type.tableone.get('Grouped by tmp_group')
                        if table_type_df is not None: flatten_tableone(table_type_df).T.to_csv(os.path.join(stats_folder_setting, f'comparison_by_type{plot_setting}.csv'))
                    metrics_lat = metrics_with_info[~metrics_with_info['laterality'].isna()]
                    if not metrics_lat.empty:
                        metrics_lat['tmp_group'] = metrics_lat['laterality'] + '_' + metrics_lat['montage']
                        table_lat = TableOne(metrics_lat, columns=plot_vars_plot, groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels_plot)
                        table_lat_df = table_lat.tableone.get('Grouped by tmp_group')
                        if table_lat_df is not None: flatten_tableone(table_lat_df).T.to_csv(os.path.join(stats_folder_setting, f'comparison_by_laterality{plot_setting}.csv'))
                    metrics_loc = metrics_with_info[~metrics_with_info['location'].isna()]
                    if not metrics_loc.empty:
                        metrics_loc['tmp_group'] = metrics_loc['location'] + '_' + metrics_loc['montage']
                        table_loc = TableOne(metrics_loc, columns=plot_vars_plot, groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels_plot)
                        table_loc_df = table_loc.tableone.get('Grouped by tmp_group')
                        if table_loc_df is not None: flatten_tableone(table_loc_df).T.to_csv(os.path.join(stats_folder_setting, f'comparison_by_location{plot_setting}.csv'))
            except Exception as e: print(f"Error generating TableOne stats: {e}")
        
    print("\n--- SVM Baseline Pipeline Complete ---")
