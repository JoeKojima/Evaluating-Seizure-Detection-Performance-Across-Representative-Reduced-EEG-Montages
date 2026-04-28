# Functions and imports
import argparse
import numpy as np
import pandas as pd
import os, sys
import scipy
import mne
from mne.filter import filter_data, notch_filter
from tqdm import tqdm
import pickle
import json
import warnings
import glob
from joblib import Parallel, delayed
os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"
os.makedirs(".matplotlib_cache",exist_ok=True)
warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# paths
working_dir = os.path.abspath("")
sys.path.append(working_dir)
sys.path.append(os.path.join(working_dir,'funcs'))
from utils import *
from feat_funcs import *
from calc_metrics import *

feat_setting = {'name':'sparcnet',
                'win':int(10), 'stride':int(2),
                'reref':'BIPOLAR', 'resample':200,
                'lowcut':1, 'highcut':40} # in seconds

montage_list = ['full',
                'uneeg_left_front',
                'uneeg_left_back',
                'uneeg_right_front',
                'uneeg_right_back',
                'uneeg_right',
                'uneeg_left',
                'uneeg_bilateral4',
                'uneeg_bilateral_back2',
                'uneeg_bilateral_front2',
                'uneeg_vert_left',
                'uneeg_vert_right',
                'uneeg_diag_left_front',
                'uneeg_diag_left_back',
                'uneeg_diag_right_front',
                'uneeg_diag_right_back',
                'uneeg_diag_bilateral_front',
                'uneeg_diag_bilateral_back',
                'uneeg_vert_bilateral',
                'epiminder_2',
                'epiminder_4',
                # 'epiminder_simulate',
                'zero']


# def average_metric(df):
#     summary = {}
#     for col in df.columns:
#         if col in ['total_dura','tn_min','fp_min','total_sz_dura']:
#             summary[col] = df[col].sum(skipna=True)
#         elif col == 'tn':
#             summary[col] = (df[col]*df['tn_min']).sum(skipna=True) / df['tn_min'].sum(skipna=True)
#         elif col in ['avg_sz_dura','auroc_sample','auprc_sample','recall_event','precision_event', 'precision_sample','fp', 'f1_event','f1_sample']:
#             summary[col] = df[col].mean(skipna=True)
#         elif col in ['recall_sample']:
#             summary[col] = (df[col]*df['avg_sz_dura']).sum(skipna=True) / df['avg_sz_dura'].sum(skipna=True)
#     summary_df = pd.DataFrame([summary])
#     return summary_df

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
            pred_df = pd.read_csv(row['pred_file'],index_col=0)
            label = pred_df.iloc[:,-1].values
            prob = pred_df['sz_prob'].values
            pred = pred_df['smoothed_pred'].values
            event_id = row['pred_file'].split('/')[-1][:-4]
            metrics = compute_metrics(label, pred, prob, stride=feat_setting['stride'])
            metric_row = pd.DataFrame([metrics],index=[event_id])
            segment_metrics.append(metric_row)
            auc_prob.extend(prob)
            auc_label.extend(label)
            auc_pred.extend(pred)
        segment_metrics = pd.concat(segment_metrics,axis=0).sort_index()
        patient_metrics = compute_metrics(auc_label, auc_pred, auc_prob, stride=feat_setting['stride'])
        patient_metrics['avg_sz_dura'] = np.nanmean(segment_metrics['avg_sz_dura'].values)
        patient_metrics['num_sz'] = np.nansum(segment_metrics['num_sz'].values)
        patient_metrics['recall_event'] = np.nanmean(segment_metrics['recall_event'].values)
        patient_metrics['balanced_acc'] = np.nanmean(segment_metrics['balanced_acc'].values)
        patient_metrics['fp'] = np.nanmean(segment_metrics['fp'].values)
        all_metrics.append(pd.DataFrame([patient_metrics],index=[key]))
    all_metrics = pd.concat(all_metrics,axis=0).sort_index() # this is per patient metrics
    return all_metrics

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Run SPaRCNet prediction on folder of edf files, store probabilities"
    )

    # Define arguments
    parser.add_argument("-o", "--output", type=str, default='sparcnet/metrics', help="Patient data to fine-tune on")
    parser.add_argument("--pred_folder", type=str, default='sparcnet/pred', help="Patient data to fine-tune on")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Patient data to fine-tune on")
    parser.add_argument("--force", action='store_true', help="Force re-running")
    parser.add_argument("-t", "--thres", type=float, default=0.5, help="Patient data to fine-tune on")
    parser.add_argument("-s", "--setting", type=str, default='', help="Patient data to fine-tune on")
    
    # Parse the arguments
    params = vars(parser.parse_args())
    metric_folder = params['output']
    pred_folder = params['pred_folder']
    force = params['force']
    thres = params['thres']

    setting_folder = f"thres{thres:.1f}"
    if params['setting']:
        setting_folder += f"_{params['setting']}"
        if params['setting'] == 'optimal':
            setting_folder = 'thres_optimal'
    metric_folder = os.path.join(metric_folder, setting_folder)
    pred_folder = os.path.join(pred_folder, setting_folder)
    
    if params['montage'] == 'all':
        montage = montage_list
    else:
        montage = params['montage'].split(',')

    # filter out seizure events that are not detected by full montage, but calculate patient-wise metrics for other montages
    if os.path.exists(os.path.join(metric_folder, 'full', 'segment_metrics.csv')):
        full_metrics = pd.read_csv(os.path.join(metric_folder, 'full', 'segment_metrics.csv'), index_col=0)
    else:
        m = 'full'
        pred_files = glob.glob(os.path.join(pred_folder, m,  '*.csv'))
        os.makedirs(os.path.join(metric_folder,m),exist_ok=True)
        out_file = os.path.join(metric_folder,m,'segment_metrics.csv')
        full_metrics = []
        for f in pred_files:
            pred_df = pd.read_csv(f,index_col=0)
            label = pred_df.iloc[:,-1].values
            prob = pred_df['sz_prob'].values
            pred = pred_df['smoothed_pred'].values
            event_id = f.split('/')[-1][:-4]
            metrics = compute_metrics(label, pred, prob, stride=feat_setting['stride'])
            metric_row = pd.DataFrame([metrics],index=[event_id])
            full_metrics.append(metric_row)
        full_metrics = pd.concat(full_metrics,axis=0).sort_index()
        full_metrics.to_csv(out_file)
    eligible_ids = full_metrics[full_metrics['recall_event'] != 0].index.to_list() # this is for each setting specifically


    for m in montage:
        # metric folder for specific setting and montage
        os.makedirs(os.path.join(metric_folder,m),exist_ok=True)
        out_file = os.path.join(metric_folder,m,'patient_metrics.csv')
        if not force and os.path.exists(out_file):
            continue
        pred_files = glob.glob(os.path.join(pred_folder, m, '*.csv'))
        pred_file_df = pd.DataFrame(pred_files, columns=['pred_file'])
        pred_file_df['emu_id'] = pred_file_df['pred_file'].apply(lambda x: x.split('/')[-1].split('_')[0])
        pred_file_df['event_id'] = pred_file_df['pred_file'].apply(lambda x: x.split('/')[-1][:-4])
        pred_file_df['is_sz'] = pred_file_df['event_id'].apply(lambda x: 'seizure' in x)
        patient_map = pd.read_csv('../emu_dataset/emu_patient_info.csv', dtype={'patient_id':str})
        pred_file_df = pred_file_df.merge(patient_map,on='emu_id',how='left')
        pred_file_df['patient_id'] = pred_file_df['patient_id'].fillna(pred_file_df['emu_id'])
        pred_file_df['is_detected'] = pred_file_df['event_id'].apply(lambda x: x in eligible_ids)

        all_metrics = patient_metrics(pred_file_df)
        all_metrics.to_csv(out_file)
        
        filtered_metrics = patient_metrics(pred_file_df[pred_file_df['is_detected']])
        filtered_metrics.to_csv(out_file.replace('.csv','_filtered.csv'))
        

        
