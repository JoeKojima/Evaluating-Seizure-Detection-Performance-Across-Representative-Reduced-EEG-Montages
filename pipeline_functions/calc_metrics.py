import numpy as np
import pandas as pd
import os, sys, json
import argparse
from sklearn.metrics import balanced_accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score

metric_labels = {'total_dura':'Total Duration, min',
        #    'tn_min':'Data Reduction, min', 
           'tn':'Specificity, %', # or specificity, percent true negative data detected as negative
        #    'fp_min':'False Positives, min', 
           'total_sz_dura':'Total Seizure Duration, min',
           'avg_sz_dura':'Average Seizure Duration, min',
           'num_sz': 'Number of Seizure',
           'auroc_sample': 'AUROC',
           'auprc_sample': 'AUPRC',
           'recall_event':'Seizure Event Detected, %', #at least 10 samples (~20 seconds) or 20% of samples detected
           'balanced_acc':'Balanced Accuracy',
        #    'recall_sample':'Seizure Sample Detected, %', # sample-wise detection
        #    'precision_event':'Precision Event, %',
        #    'precision_sample':'Precision Sample, %',
           'fp':'False Alarm per hour'}
        #    'f1_event':'F1 Event',
        #    'f1_sample':'F1 Sample'} 


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
    metrics['total_dura']=len(true) * stride / 60
    # metrics['tn_min']=tn * stride / 60
    metrics['tn']=tn / np.sum(true == 0)
    # metrics['fp_min']= fp * stride / 60
    if np.any(true == 1):
        metrics['total_sz_dura']=np.sum(true) * stride / 60
        metrics['avg_sz_dura']=np.mean([(end-start) * stride / 60 for start, end in seiz_ranges])
        metrics['num_sz'] = len(seiz_ranges)
        sz_detected = np.array([np.sum(pred[start:end]) >= min(0.2*(end-start),10) for start, end in seiz_ranges])
        recall = np.sum(sz_detected) / len(sz_detected)
        metrics['auprc_sample'] = average_precision_score(true, prob)
        metrics['auroc_sample'] = roc_auc_score(true, prob)
        metrics['recall_event']=recall
        # metrics['recall_sample'] = tp/np.sum(true)
        # if np.any(pred==1):
        #     pred_is_sz = np.array([np.any(true[start:end] == 1) for start, end in pred_seiz_ranges])
        #     metrics['precision_event'] = np.sum(pred_is_sz) / len(pred_seiz_ranges)
        #     metrics['fp']= (len(pred_seiz_ranges)-np.sum(pred_is_sz))/metrics['total_dura']*60 # false alarm rate/hour
        #     metrics['precision_sample'] = tp/np.sum(pred)
        #     try:
        #         metrics['f1_event'] = 2*metrics['precision_event']*metrics['recall_event']/(metrics['precision_event']+metrics['recall_event'])
        #         metrics['f1_sample'] = 2*metrics['precision_sample']*metrics['recall_sample']/(metrics['precision_sample']+metrics['recall_sample'])
        #     except:
        #         metrics['f1_event'] = np.nan
        #         metrics['f1_sample'] = np.nan
        # else:
        #     metrics['precision_event'] = np.nan
        #     metrics['precision_sample'] = np.nan
        #     metrics['fp'] = 0
        #     metrics['f1_event'] = np.nan   
        #     metrics['f1_sample'] = np.nan
    else:
        for key in ['total_sz_dura','avg_sz_dura','num_sz','recall_event', 'auprc_sample','auroc_sample']:#'recall_sample','precision_event','precision_sample','f1_event','f1_sample']:
            metrics[key]=np.nan
        metrics['fp'] = len(pred_seiz_ranges)/metrics['total_dura']*60
    metrics['balanced_acc'] = np.nanmean([metrics['recall_event'],metrics['tn']])
    return metrics
