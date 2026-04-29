import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from timescoring.annotations import Annotation
from timescoring import scoring

metric_labels = {
    "total_dura": "Total Duration, min",
    "tn": "Specificity, %",  # or specificity, percent true negative data detected as negative
    "total_sz_dura": "Total Seizure Duration, min",
    "avg_sz_dura": "Average Seizure Duration, min",
    "num_sz": "Number of Seizure",
    "num_pred": "Number of Predicted Events",
    "auroc_sample": "AUROC",
    "auprc_sample": "AUPRC",
    "recall_event": "Seizure Event Detected, %",
    "precision_event": "Precision, %",
    "f1_event": "Event-wise F1 score",
    "fp": "False Alarm per hour",
}


def extract_seiz_ranges(true_data):
    diff_data = np.diff(np.concatenate([[0], np.squeeze(true_data), [0]]))
    starts = np.where(diff_data == 1)[0]
    stops = np.where(diff_data == -1)[0]
    return list(zip(starts, stops))


def compute_metrics(true, pred, prob, stride=2):
    """Compute metrics for seizure detection based on true labels and predicted labels."""
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    seiz_ranges = extract_seiz_ranges(true)
    pred_seiz_ranges = extract_seiz_ranges(pred)

    metrics = {}
    metrics["total_dura"] = len(true) * stride / 60  # in minute
    metrics["tn"] = tn / np.sum(true == 0)
    metrics["num_pred"] = len(pred_seiz_ranges)
    metrics["num_sz"] = len(seiz_ranges)

    labels = Annotation(true, 1 / stride)
    preds = Annotation(pred, 1 / stride)
    param = scoring.EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=5 * 60,
        minDurationBetweenEvents=90,
    )
    scores = scoring.EventScoring(labels, preds, param)

    # metrics['fp_min']= fp * stride / 60
    if np.any(true == 1):
        metrics["total_sz_dura"] = np.sum(true) * stride / 60
        metrics["avg_sz_dura"] = np.mean(
            [(end - start) * stride / 60 for start, end in seiz_ranges]
        )
        metrics["auprc_sample"] = average_precision_score(true, prob)
        metrics["auroc_sample"] = roc_auc_score(true, prob)
    else:
        for key in [
            "total_sz_dura",
            "avg_sz_dura",
            "auprc_sample",
            "auroc_sample",
        ]:  # ,'precision_event','precision_sample','f1_event','f1_sample']:
            metrics[key] = np.nan
    metrics["recall_event"] = scores.sensitivity
    metrics["fp"] = scores.fpRate / 24
    metrics["precision_event"] = scores.precision
    metrics["f1_event"] = scores.f1

    return metrics


