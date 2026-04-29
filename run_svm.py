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
    from utils import * 
    from feat_funcs import *
    from calc_metrics import *
except ImportError as e:
    print(f"FATAL ERROR: Could not import required pipeline components. Missing: {e}")
    sys.exit(1)

# --- CONSTANTS ---

MONTAGE_DICT = {
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

FS_FEAT = 2.0
TRAIN_DURATION_SEC = 60

feat_setting_svm = {
    'win':1,
    'stride':0.5,
    'reref':'BIPOLAR',
    'lowcut':1,
    'highcut':40
}

plot_vars = ['auroc_sample', 'auprc_sample', 'recall_event', 'precision_event', 'f1_event', 'fp']
plot_labels = [metric_labels.get(k, k) for k in plot_vars]


# --- 3. HELPER FUNCTIONS (LOCAL DEFINITIONS) ---
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


def process_pat(pat, group):
    warnings.filterwarnings('ignore')
    group = group.sort_values('file')
    iic_file = group[group['type']=='iic'].iloc[0]['file']
    raw, df, label_df, fs = load_edf_file(iic_file)
    prepro = Preprocessor()
    prepro.fit({'samplingFreq':fs, 'samplingFreqRaw':fs, 'channelNames':df.columns, 'studyType':'eeg', 'numberOfChannels':df.shape[1]})
    preprocessed = prepro.preprocess(df)
    for m in montages_to_run:
        montage_processor = MONTAGE_DICT[m]
        if isinstance(montage_processor,list):
            data_df = preprocessed['BIPOLAR']
            data_df = data_df[MONTAGE_DICT[m]]
        else:
            data_df = montage_processor(preprocessed)
        train_data = data_df.iloc[:int(fs*60),:].values
        clf_list = []
        for i in range(train_data.shape[1]):
            X_train = extract_features(train_data[:, i], fs=fs)
            X_train = np.nan_to_num(X_train)
            clf = train_one_class_svm(X_train)
            clf_list.append(clf)

        for _, row in group.iterrows():
            file_name = row['file']
            raw, df, label_df, fs = load_edf_file(file_name)
            prepro = Preprocessor()
            prepro.fit({'samplingFreq':fs, 'samplingFreqRaw':fs, 'channelNames':df.columns, 'studyType':'eeg', 'numberOfChannels':df.shape[1]})
            preprocessed = prepro.preprocess(df)
            if isinstance(montage_processor,list):
                data_df = preprocessed['BIPOLAR']
                data_df = data_df[MONTAGE_DICT[m]]
            else:
                data_df = montage_processor(preprocessed)

            prob_path = os.path.join(prob_folder,m,file_name.split('/')[-1].replace('.edf','.csv'))
            if os.path.exists(prob_path) and not force:
                continue
            
            len_feat = extract_features(data_df.iloc[:, 0].values, fs=fs)
            start_time_s = data_df.index.min()
            time_vals = start_time_s + feat_setting_svm['stride'] + np.arange(0, len(len_feat)) * feat_setting_svm['stride']
            feat_labels = [label_df.loc[(data_df.index >= time_vals[i]-feat_setting_svm['win']) & (data_df.index < time_vals[i]),'labels'].any().astype(int) for i in range(len(time_vals))]
            pred_df_final = pd.DataFrame(index=time_vals)
            pred_df_final['label'] = feat_labels
            for i in range(data_df.shape[1]):
                X_test = extract_features(data_df.iloc[:, i].values, fs=fs)
                X_test = np.nan_to_num(X_test)

                y_pred = compute_novelty_scores(clf_list[i], X_test)
                nu_hat = estimate_outlier_fraction(y_pred, n=20)
                
                smoothing_sigma = 2 * int(len(nu_hat) / 1000 + 1)
                nu_filt = np.round(gaussian_filter1d(nu_hat, smoothing_sigma), 100)
                pred_df_final['nu_hat_'+data_df.columns[i]] = nu_hat
                pred_df_final['sz_prob_'+data_df.columns[i]] = nu_filt

            
            pred_df_final.index = pd.to_datetime(pred_df_final.index, unit='s')
            pred_df_final.to_csv(prob_path)


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

def get_optimal_thres_f1(prob_files, stride):
    pass

def process_file_svm(file_name, thres=0.99, avg = True):
    warnings.filterwarnings('ignore')
    out_file = os.path.join(pred_folder_setting,m,file_name.split('/')[-1])
    if not force and os.path.exists(out_file):
        return
    prob_df = pd.read_csv(file_name,index_col=0)
    prob_mat = prob_df[[c for c in prob_df.columns if c.startswith('nu_hat')]].values
    sz_prob = prob_mat.mean(axis=1)
    if avg:
        pred = detect_seizure(sz_prob, threshold=thres)
    else:
        pred_mat = (prob_mat >= thres).astype(int)
        # perc_chan = pred_mat.sum(axis=1)/pred_mat.shape[1]
        # pred = perc_chan >= 1
        pred = pred_mat.sum(axis=1) >= min(2,pred_mat.shape[1])
    pred = get_event_smoothed_pred(smooth_pred(pred), gap_num=int(4/feat_setting_svm['stride']), min_event_num=int(20/feat_setting_svm['stride'])) #int(4/feat_setting['stride'])
    pred = apply_persistence(pred) 
    pred_df = pd.DataFrame(np.vstack([sz_prob, pred]).T,columns=['sz_prob','pred'], index=prob_df.index)
    pred_df = pd.concat([pred_df, prob_df[['label']]],axis=1)
    pred_df.to_csv(out_file)

# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SVM Baseline Pipeline (Patient-Specific).")
    parser.add_argument("-d", "--data_folder", type=str, default='emu_dataset', help="Path to emu_dataset")
    parser.add_argument("-o", "--output_folder", type=str, default='svm_results', help="Output folder")
    parser.add_argument("-p", "--patient_info", type=str, default='emu_dataset/dataset_admission_info.csv', help="Patient info CSV")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Montages")
    parser.add_argument("-s", "--setting", type=str, default='', help="Setting for threshold ('optimal' or 'fixed')")
    parser.add_argument("-t", "--thres", type=float, default=0.9, help="Fixed threshold to use if setting is not optimal")
    parser.add_argument("--n_jobs", type=int, default=40, help="Parallel jobs")
    parser.add_argument("--thres_file", type=str, default='threses_all.csv', help="File containing thresholds for each montage")
    parser.add_argument("--force", action='store_true', help="Force rerun")
    
    params = vars(parser.parse_args())

    base_data_folder = params['data_folder']
    base_output_folder = params['output_folder']
    patient_map_file = params['patient_info']
    n_jobs = params['n_jobs']
    force = params['force']
    
    # Establish dynamic settings folder name based on Youden's optimization or fixed thres
    thres_val = params['thres']
    setting_val = params['setting']
    
    if setting_val:
        setting_folder_name = setting_val
    else:
        setting_folder_name = f"thres{thres_val:.1f}"

    prob_folder = os.path.join(base_output_folder, 'prob')
    pred_folder_setting = os.path.join(base_output_folder, 'pred', setting_folder_name)
    metric_folder_setting = os.path.join(base_output_folder, 'metrics', setting_folder_name)
    stats_folder_setting = os.path.join(base_output_folder, 'stats', setting_folder_name)
    
    if params['montage'] == 'all':
        montages_to_run = list(MONTAGE_DICT.keys())
    else:
        montages_to_run = [m.strip() for m in params['montage'].split(',') if m.strip() in MONTAGE_DICT]
    for m in montages_to_run:
        os.makedirs(os.path.join(prob_folder,m),exist_ok=True)

    try:
        all_files = glob.glob(f"{base_data_folder}/**/*.edf", recursive=True)
        if not all_files:
            print(f"Warning: No .edf files found in {base_data_folder}")
    except Exception as e:
        print(f"Error finding EDF files: {e}")
        all_files = []
    
    file_df = pd.DataFrame({'file':all_files})
    file_df['patient'] = file_df['file'].apply(lambda x: x.split('/')[-1].split('_')[0])
    file_df['type'] = file_df['file'].apply(lambda x: 'seizure' if 'seizure' in x else 'iic')
    n_pat = len(file_df['patient'].unique())
    
    print("\n--- STEP 1: Generating Probabilities (Patient-Specific SVM) ---")
    
    with tqdm(total=n_pat,desc = 'Processing block'):
        results = Parallel(n_jobs=20)(delayed(process_pat)(pat, group) for pat, group in file_df.groupby('patient'))
        
    print(f"Probabilities generated in {prob_folder}.")

    print("\n--- STEP 2: Generating Predictions ---")

    for m in montages_to_run:
        prob_files = glob.glob(os.path.join(prob_folder, m, '*.csv'))
        if not prob_files: continue
        
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

        os.makedirs(os.path.join(pred_folder_setting, m), exist_ok=True)
        with tqdm(total=len(prob_files),desc = 'Processing file'):
            results = Parallel(n_jobs=40)(delayed(process_file_svm)(file_name, current_thres) for file_name in prob_files)

    # =================================================================
    # STEP 3: Calculate Metrics (from calc_metrics.py and get_metrics.py)
    # =================================================================
    print("\n--- STEP 3: Calculating Metrics ---")
    
    for m in tqdm(montages_to_run, desc='Step 3/4: Calculating Metrics'):
        os.makedirs(os.path.join(metric_folder_setting, m), exist_ok=True)
        pred_files = glob.glob(os.path.join(pred_folder_setting, m, '*.csv'))
        if not pred_files:
            print(f"  No prediction files found for {m}. Skipping.")
            continue

        # segment metrics
        out_file = os.path.join(metric_folder_setting,m,'segment_metrics.csv')
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
                metrics = compute_metrics(label, pred, prob, stride=feat_setting_svm['stride'])
                metric_row = pd.DataFrame([metrics],index=[event_id])
                full_metrics.append(metric_row)
            full_metrics = pd.concat(full_metrics,axis=0).sort_index()
            full_metrics[['precision_event','f1_event']] = full_metrics[['precision_event','f1_event']].fillna(0.0)
            full_metrics.to_csv(out_file)

        # patient metrics
        out_file = os.path.join(metric_folder_setting, m, 'patient_metrics.csv')
        if not force and os.path.exists(out_file):
            continue
        else:
            pred_file_df = pd.DataFrame(pred_files, columns=['pred_file'])
            pred_file_df['admission_id'] = pred_file_df['pred_file'].apply(lambda x: x.split('/')[-1].split('_')[0])
            pred_file_df['event_id'] = pred_file_df['pred_file'].apply(lambda x: x.split('/')[-1][:-4])
            pred_file_df['is_sz'] = pred_file_df['event_id'].apply(lambda x: 'seizure' in x)

            # Calculate metrics for all files
            all_p_metrics = patient_metrics(pred_file_df, feat_setting_svm['stride'])
            if not all_p_metrics.empty:
                all_p_metrics.to_csv(out_file)

    # =================================================================
    # STEP 4: Generate Stats and Plots (from plot_metrics.py)
    # =================================================================
    print("\n--- STEP 4: Generating Stats and Plots ---")
    os.makedirs(stats_folder_setting, exist_ok=True)

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
            for m in montages_to_run:
                metric_file = os.path.join(metric_folder_setting, m, file_name)
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
                flatten_tableone(table1_df).T.to_csv(os.path.join(stats_folder_setting, f'comparison.csv'))

            # By epilepsy type
            metrics_with_info = all_metrics.merge(patient_map, on='admission_id', how='left')
            metrics_with_info = metrics_with_info[~metrics_with_info['epilepsy_type'].isna()]
            metrics_with_info['tmp_group'] = metrics_with_info['epilepsy_type'] + '_' + metrics_with_info['montage']
            table_type = TableOne(metrics_with_info, columns=plot_vars,
                                    groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels)
            table_type_df = table_type.tableone.get('Grouped by tmp_group')
            if table_type_df is not None:
                flatten_tableone(table_type_df).T.to_csv(os.path.join(stats_folder_setting, f'comparison_by_type.csv'))

            # By laterality
            metrics_with_info = all_metrics.merge(patient_map, on='admission_id', how='left')
            metrics_with_info = metrics_with_info[~metrics_with_info['laterality'].isna()]
            metrics_with_info['tmp_group'] = metrics_with_info['laterality'] + '_' + metrics_with_info['montage']
            table_lat = TableOne(metrics_with_info, columns=plot_vars,
                                    groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels)
            table_lat_df = table_lat.tableone.get('Grouped by tmp_group')
            if table_lat_df is not None:
                flatten_tableone(table_lat_df).T.to_csv(os.path.join(stats_folder_setting, f'comparison_by_laterality.csv'))

            # By location
            metrics_with_info = all_metrics.merge(patient_map, on='admission_id', how='left')
            metrics_with_info = metrics_with_info[~metrics_with_info['location'].isna()]
            metrics_with_info['tmp_group'] = metrics_with_info['location'] + '_' + metrics_with_info['montage']
            table_loc = TableOne(metrics_with_info, columns=plot_vars,
                                    groupby='tmp_group', missing=False, overall=False, pval=False, decimals=3, labels=plot_labels)
            table_loc_df = table_loc.tableone.get('Grouped by tmp_group')
            if table_loc_df is not None:
                flatten_tableone(table_loc_df).T.to_csv(os.path.join(stats_folder_setting, f'comparison_by_location.csv'))

        except Exception as e:
            print(f"    Error generating TableOne stats: {e}")

    print("\n--- SVM Baseline Pipeline Complete ---")
