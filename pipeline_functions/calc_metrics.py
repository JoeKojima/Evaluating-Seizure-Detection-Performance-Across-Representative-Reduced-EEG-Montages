import numpy as np
import pandas as pd
import os, sys, json
import argparse
from sklearn.metrics import balanced_accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score
from timescoring.annotations import Annotation
from timescoring import scoring 

metric_labels = {'total_dura':'Total Duration, min',
           'tn':'Specificity, %', # or specificity, percent true negative data detected as negative
           'total_sz_dura':'Total Seizure Duration, min',
           'avg_sz_dura':'Average Seizure Duration, min',
           'num_sz': 'Number of Seizure',
           'num_pred': 'Number of Predicted Events',
           'auroc_sample': 'AUROC',
           'auprc_sample': 'AUPRC',
           'recall_event':'Seizure Event Detected, %', 
           'precision_event':'Precision, %',
           'f1_event':'Event-wise F1 score',
           'fp':'False Alarm per hour'}

def extract_seiz_ranges(true_data):
    diff_data = np.diff(np.concatenate([[0],np.squeeze(true_data),[0]]))
    starts = np.where(diff_data == 1)[0]
    stops = np.where(diff_data == -1)[0]
    return list(zip(starts,stops))

def compute_metrics(true, pred, prob, stride = 2):
    """Compute metrics for seizure detection based on true labels and predicted labels."""
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    seiz_ranges = extract_seiz_ranges(true)
    pred_seiz_ranges = extract_seiz_ranges(pred)

    metrics = {}
    metrics['total_dura']=len(true) * stride / 60 # in minute
    metrics['tn']=tn / np.sum(true == 0)
    metrics['num_pred'] = len(pred_seiz_ranges)
    metrics['num_sz'] = len(seiz_ranges)
        
    labels = Annotation(true, 1/stride)
    preds = Annotation(pred, 1/stride)
    param = scoring.EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=5 * 60,
        minDurationBetweenEvents=90)
    scores = scoring.EventScoring(labels, preds, param)

    # metrics['fp_min']= fp * stride / 60
    if np.any(true == 1):
        metrics['total_sz_dura']=np.sum(true) * stride / 60
        metrics['avg_sz_dura']=np.mean([(end-start) * stride / 60 for start, end in seiz_ranges])
        metrics['auprc_sample'] = average_precision_score(true, prob)
        metrics['auroc_sample'] = roc_auc_score(true, prob)
    else:
        for key in ['total_sz_dura','avg_sz_dura','auprc_sample','auroc_sample']:#,'precision_event','precision_sample','f1_event','f1_sample']:
            metrics[key]=np.nan
    metrics['recall_event'] = scores.sensitivity
    metrics['fp'] = scores.fpRate/24
    metrics['precision_event'] = scores.precision
    metrics['f1_event'] = scores.f1
 
    return metrics


def patient_metrics(pred_file_df, stride):
    all_metrics = []
    for key, group in pred_file_df.groupby('admission_id'):
        if group['is_sz'].sum() == 0:
            continue
        segment_metrics = []
        auc_prob = []
        auc_label = []
        auc_pred = []
        for _, row in group.iterrows():
            try:
                pred_df = pd.read_csv(row['pred_file'],index_col=0)
                if len(pred_df) == 0:
                    print(row['pred_file'].split('/')[-1][:-4])
                    continue
            except:
                continue
            label = pred_df.iloc[:,-1].values
            prob = pred_df['sz_prob'].values
            pred = pred_df['pred'].values
            event_id = row['pred_file'].split('/')[-1][:-4]
            metrics = compute_metrics(label, pred, prob, stride=stride)
            metric_row = pd.DataFrame([metrics],index=[event_id])
            segment_metrics.append(metric_row)
            auc_prob.extend(prob)
            auc_label.extend(label)
            auc_pred.extend(pred)
        segment_metrics = pd.concat(segment_metrics,axis=0).sort_index()
        patient_metrics = compute_metrics(auc_label, auc_pred, auc_prob, stride=stride)
        patient_metrics['avg_sz_dura'] = np.nansum(segment_metrics['total_sz_dura'].values)/np.nansum(segment_metrics['num_sz'].values)
        patient_metrics['num_sz'] = np.nansum(segment_metrics['num_sz'].values)
        patient_metrics['num_pred'] = np.nansum(segment_metrics['num_pred'].values)
        patient_metrics['recall_event'] = np.nanmean(segment_metrics['recall_event'].values)
        patient_metrics['fp'] = np.nanmean(segment_metrics['fp'].values)
        patient_metrics['precision_event'] = np.nansum(segment_metrics['precision_event'] * segment_metrics['num_pred']) / np.nansum(segment_metrics['num_pred'])
        patient_metrics['f1_event'] = 2*patient_metrics['recall_event']*patient_metrics['precision_event']/(patient_metrics['recall_event']+patient_metrics['precision_event'])
        all_metrics.append(pd.DataFrame([patient_metrics],index=[key]))
    all_metrics = pd.concat(all_metrics,axis=0).sort_index() # this is per patient metrics
    all_metrics[['precision_event','f1_event']] = all_metrics[['precision_event','f1_event']].fillna(0.0)
    return all_metrics