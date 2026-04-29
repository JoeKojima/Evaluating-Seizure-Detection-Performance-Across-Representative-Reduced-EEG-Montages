import os
import glob

import numpy as np
import pandas as pd
from tableone import TableOne
from tqdm import tqdm

from calc_metrics import compute_metrics, metric_labels
from utils import flatten_tableone

metric_use = [
    "auroc_sample",
    "auprc_sample",
    "recall_event",
    "precision_event",
    "f1_event",
    "fp",
]
metric_use_labels = [metric_labels.get(k, k) for k in metric_use]


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
