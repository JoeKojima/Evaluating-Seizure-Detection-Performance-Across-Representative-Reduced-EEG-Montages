import os
import glob
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from tableone import TableOne
from tqdm import tqdm

from calc_metrics import compute_metrics, metric_labels
from utils import flatten_tableone
from feat_funcs import apply_persistence, get_event_smoothed_pred, smooth_pred
from sklearn.metrics import roc_curve
from timescoring import scoring
from timescoring.annotations import Annotation

metric_use = [
    "auroc_sample",
    "auprc_sample",
    "recall_event",
    "precision_event",
    "f1_event",
    "fp",
]
metric_use_labels = [metric_labels.get(k, k) for k in metric_use]
model_label_map = {
    "sparcnet": "SPaRCNet",
    "ndd": "NDD",
    "svm": "SVM",
}
model_stride = {
    "sparcnet": 2,
    "ndd": 0.5,
    "svm": 0.5,
}


def compute_eventwise_scores(model_data, t, model):
    import warnings

    warnings.filterwarnings("ignore")
    stride = model_stride[model]
    all_scores = []
    for _, group in model_data.groupby("event_id"):
        true = group["label"].values
        prob = group["sz_prob"].values
        pred = (prob >= t).astype(int)
        pred = get_event_smoothed_pred(
            smooth_pred(pred), gap_num=int(4 / stride), min_event_num=int(20 / stride)
        )
        if model == "SVM":
            pred = apply_persistence(pred)
        metrics = compute_metrics(true, pred, prob, stride)
        metrics["patient"] = group["patient"].iloc[0]
        all_scores.append(metrics)
    all_scores = pd.DataFrame(all_scores)
    agg_scores = []
    for name, group in all_scores.groupby(["patient"]):
        recall = np.nanmean(group["recall_event"].values)
        fp = np.nanmean(group["fp"].values)
        perc = np.nansum(group["precision_event"] * group["num_pred"]) / np.nansum(
            group["num_pred"]
        )
        f1 = 2 * recall * perc / (recall + perc)
        agg_scores.append([recall, fp, perc, f1])
    agg_scores = np.array(agg_scores)
    agg_scores[np.isnan(agg_scores)] = 0.0
    agg_scores[np.isinf(agg_scores)] = 0.0
    agg_scores = np.nanmean(np.array(agg_scores), axis=0)
    return agg_scores


def get_optimal_thres(prob_files, prob_func, thres_file, method="f1"):
    # this only runs with prob_files from one model and one montage
    montage = prob_files[0].split("/")[-2]
    model = prob_files[0].split("/")[-3].split("_")[0]
    model_label = model_label_map[model]
    if thres_file is not None:
        if os.path.exists(thres_file):
            threses = pd.read_csv(thres_file)
            thres_row = threses[
                (threses["model"] == model_label) & (threses["montage"] == montage)
            ]
            if len(thres_row) > 0:
                return thres_row["thres_" + method].iloc[-1]
            else:
                print(f"No threshold found for {model_label} {montage} in {thres_file}")
        else:
            print("No threshold file found\nCalculating optimal threshold...")
    print("No threshold file provided\nCalculating optimal threshold...")

    all_data = []
    for f in prob_files:
        try:
            prob_df = pd.read_csv(f, index_col=0)
            sz_prob = prob_func(prob_df)  # Assuming SZ is column 1 (0-indexed)
            label = prob_df["label"].values
        except Exception as e:
            print(f"Error reading prob file {f}: {e}")
            continue
        tmp = pd.DataFrame({"sz_prob": sz_prob, "label": label})
        tmp["event_id"] = f[:-4]
        all_data.append(tmp)
    all_data = pd.concat(all_data)
    all_data["patient"] = all_data["event_id"].apply(lambda x: x.split("_")[0])

    fpr, tpr, thres = roc_curve(all_data["sz_prob"].values, all_data["label"].values)
    opt_thres_yodenj = thres[np.argmax(tpr - fpr)]

    N = 200
    if len(thres) > N:
        idx = np.linspace(0, len(thres) - 1, N).astype(int)
        thres = thres[idx]
    results = Parallel(n_jobs=40)(
        delayed(compute_eventwise_scores)(all_data, t, model) for t in thres
    )
    sens, far, perc, f1 = zip(*results)
    sens = np.array(list(sens))
    far = np.array(list(far))
    perc = np.array(list(perc))
    f1 = np.array(list(f1))
    f1[0] = 0.0
    f1[-1] = 0.0
    opt_thres_f1 = thres[np.argmax(f1)]
    thres_row = pd.DataFrame(
        [
            {
                "model": model_label,
                "montage": montage,
                "thres_yodenj": opt_thres_yodenj,
                "thres_f1": opt_thres_f1,
            }
        ]
    )
    if thres_file is not None:
        if os.path.exists(thres_file):
            thres_row.to_csv(thres_file, mode="a", header=False, index=False)
        else:
            thres_row.to_csv(thres_file, index=False)
    if method == "yodenj":
        return opt_thres_yodenj
    elif method == "f1":
        return opt_thres_f1
    else:
        raise ValueError(f"Invalid method: {method}")


def patient_metrics(pred_file_df, stride):
    all_metrics = []
    for key, group in pred_file_df.groupby("admission_id"):
        if group["is_sz"].sum() == 0:
            continue
        segment_metrics = []
        auc_prob = []
        auc_label = []
        auc_pred = []
        for _, row in group.iterrows():
            try:
                pred_df = pd.read_csv(row["pred_file"], index_col=0)
                if len(pred_df) == 0:
                    print(row["pred_file"].split("/")[-1][:-4])
                    continue
            except:
                continue
            label = pred_df.iloc[:, -1].values
            prob = pred_df["sz_prob"].values
            pred = pred_df["pred"].values
            event_id = row["pred_file"].split("/")[-1][:-4]
            metrics = compute_metrics(label, pred, prob, stride=stride)
            metric_row = pd.DataFrame([metrics], index=[event_id])
            segment_metrics.append(metric_row)
            auc_prob.extend(prob)
            auc_label.extend(label)
            auc_pred.extend(pred)
        segment_metrics = pd.concat(segment_metrics, axis=0).sort_index()
        patient_metrics = compute_metrics(auc_label, auc_pred, auc_prob, stride=stride)
        patient_metrics["avg_sz_dura"] = np.nansum(
            segment_metrics["total_sz_dura"].values
        ) / np.nansum(segment_metrics["num_sz"].values)
        patient_metrics["num_sz"] = np.nansum(segment_metrics["num_sz"].values)
        patient_metrics["num_pred"] = np.nansum(segment_metrics["num_pred"].values)
        patient_metrics["recall_event"] = np.nanmean(
            segment_metrics["recall_event"].values
        )
        patient_metrics["fp"] = np.nanmean(segment_metrics["fp"].values)
        patient_metrics["precision_event"] = np.nansum(
            segment_metrics["precision_event"] * segment_metrics["num_pred"]
        ) / np.nansum(segment_metrics["num_pred"])
        patient_metrics["f1_event"] = (
            2
            * patient_metrics["recall_event"]
            * patient_metrics["precision_event"]
            / (patient_metrics["recall_event"] + patient_metrics["precision_event"])
        )
        all_metrics.append(pd.DataFrame([patient_metrics], index=[key]))
    all_metrics = pd.concat(
        all_metrics, axis=0
    ).sort_index()  # this is per patient metrics
    all_metrics[["precision_event", "f1_event"]] = all_metrics[
        ["precision_event", "f1_event"]
    ].fillna(0.0)
    return all_metrics


def calculate_metrics_for_montages(
    montage_keys, pred_folder_setting, metric_folder, stride, force
):
    for montage in tqdm(montage_keys, desc="Step 3/4: Calculating Metrics"):
        os.makedirs(os.path.join(metric_folder, montage), exist_ok=True)
        pred_files = glob.glob(os.path.join(pred_folder_setting, montage, "*.csv"))
        if not pred_files:
            print(f"  No prediction files found for {montage}. Skipping.")
            continue

        segment_out_file = os.path.join(metric_folder, montage, "segment_metrics.csv")
        if force or not os.path.exists(segment_out_file):
            full_metrics = []
            for pred_file in pred_files:
                pred_df = pd.read_csv(pred_file, index_col=0)
                label = pred_df["label"].values
                prob = pred_df["sz_prob"].values
                pred = pred_df["pred"].values
                event_id = os.path.basename(pred_file)[:-4]
                metrics = compute_metrics(label, pred, prob, stride=stride)
                full_metrics.append(pd.DataFrame([metrics], index=[event_id]))

            if full_metrics:
                full_metrics = pd.concat(full_metrics, axis=0).sort_index()
                full_metrics[["precision_event", "f1_event"]] = full_metrics[
                    ["precision_event", "f1_event"]
                ].fillna(0.0)
                full_metrics.to_csv(segment_out_file)

        patient_out_file = os.path.join(metric_folder, montage, "patient_metrics.csv")
        if force or not os.path.exists(patient_out_file):
            pred_file_df = pd.DataFrame(pred_files, columns=["pred_file"])
            pred_file_df["admission_id"] = pred_file_df["pred_file"].apply(
                lambda x: os.path.basename(x).split("_")[0]
            )
            pred_file_df["event_id"] = pred_file_df["pred_file"].apply(
                lambda x: os.path.basename(x)[:-4]
            )
            pred_file_df["is_sz"] = pred_file_df["event_id"].apply(
                lambda x: "seizure" in x
            )
            all_patient_metrics = patient_metrics(pred_file_df, stride)
            if not all_patient_metrics.empty:
                all_patient_metrics.to_csv(patient_out_file)


def generate_stats_tables(
    montage_keys,
    metric_folder,
    stats_folder,
    patient_map_file,
):
    os.makedirs(stats_folder, exist_ok=True)
    if not os.path.exists(patient_map_file):
        print(
            f"Error: Patient info file not found at {patient_map_file}. Cannot generate stats by type."
        )
        return

    patient_map = pd.read_csv(
        patient_map_file, dtype={"patient_id": str, "admission_id": str}
    )
    print("    Generating stats tables...")

    try:
        all_metrics = []
        for montage in montage_keys:
            metric_file = os.path.join(metric_folder, montage, "patient_metrics.csv")
            metrics = pd.read_csv(metric_file, index_col=0)
            metrics["montage"] = montage
            all_metrics.append(metrics)

        all_metrics = (
            pd.concat(all_metrics, axis=0)
            .reset_index()
            .rename(columns={"index": "admission_id"})
        )
        all_metrics["admission_id"] = all_metrics["admission_id"].astype(str)

        table_all = TableOne(
            all_metrics,
            columns=metric_use,
            groupby="montage",
            missing=False,
            overall=False,
            pval=False,
            decimals=3,
            labels=metric_use_labels,
        )
        table_all_df = table_all.tableone.get("Grouped by montage")
        if table_all_df is not None:
            flatten_tableone(table_all_df).T.to_csv(
                os.path.join(stats_folder, "comparison.csv")
            )

        for col, out_name in [
            ("epilepsy_type", "comparison_by_type.csv"),
            ("laterality", "comparison_by_laterality.csv"),
            ("location", "comparison_by_location.csv"),
        ]:
            grouped = all_metrics.merge(patient_map, on="admission_id", how="left")
            grouped = grouped[~grouped[col].isna()]
            grouped["tmp_group"] = grouped[col] + "_" + grouped["montage"]
            table = TableOne(
                grouped,
                columns=metric_use,
                groupby="tmp_group",
                missing=False,
                overall=False,
                pval=False,
                decimals=3,
                labels=metric_use_labels,
            )
            table_df = table.tableone.get("Grouped by tmp_group")
            if table_df is not None:
                flatten_tableone(table_df).T.to_csv(
                    os.path.join(stats_folder, out_name)
                )
    except Exception as exc:
        print(f"    Error generating TableOne stats: {exc}")
