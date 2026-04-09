# Functions and imports
import argparse
import numpy as np
import pandas as pd
import os, sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Non-GUI backend
from tqdm import tqdm
import warnings
import glob
os.environ["MPLCONFIGDIR"] = ".matplotlib_cache"
os.makedirs(".matplotlib_cache",exist_ok=True)
warnings.filterwarnings("ignore")

# paths
working_dir = os.path.abspath("")
sys.path.append(working_dir)
sys.path.append(os.path.join(working_dir,'funcs'))
from utils import *
from feat_funcs import *
from calc_metrics import *

from tableone import TableOne
from scipy.stats import wilcoxon
import seaborn as sns
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.weight'] = 'light'
full_palete = ["#5F6C7B", "#368899", "#E03F67", "#F47C6D",'#3B9B63','#A0E03F',"#9C6FCF"]


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

multi_comp = {'uneeg_left':['full','uneeg_left_front','uneeg_left_back','uneeg_left','uneeg_vert_left','uneeg_diag_left_front','uneeg_diag_left_back'],
              'uneeg_right':['full','uneeg_right_front','uneeg_right_back','uneeg_right','uneeg_vert_right','uneeg_diag_right_front','uneeg_diag_right_back'],
              'uneeg_bilateral':['full','uneeg_bilateral_front2','uneeg_bilateral_back2','uneeg_bilateral4','uneeg_vert_bilateral','uneeg_diag_bilateral_front','uneeg_diag_bilateral_back'],
              'uneeg_horz_front':['full','uneeg_left_front','uneeg_right_front','uneeg_bilateral_front2'],
              'uneeg_horz_back':['full','uneeg_left_back','uneeg_right_back','uneeg_bilateral_back2'],
              'uneeg_horz':['full','uneeg_left','uneeg_right','uneeg_bilateral4'],
              'uneeg_vert':['full','uneeg_vert_left', 'uneeg_vert_right','uneeg_vert_bilateral'],
              'uneeg_diag':['full', 'uneeg_diag_left_front', 'uneeg_diag_left_back','uneeg_diag_right_front', 'uneeg_diag_right_back','uneeg_diag_bilateral_front','uneeg_diag_bilateral_back'],
              'epiminder':['full','epiminder_2','epiminder_4']}

metric_labels = {'total_dura':'Total Duration, min',
        'tn':'Specificity', # or specificity, percent true negative data detected as negative
        'total_sz_dura':'Total Seizure Duration, min',
        'avg_sz_dura':'Average Seizure Duration, min',
        'num_sz': 'Number of Seizure',
        'auroc_sample': 'AUROC',
        'auprc_sample': 'AUPRC',
        'recall_event':'Recall', #at least 10 samples (~20 seconds) or 20% of samples detected
        'balanced_acc':'Balanced Accuracy',
        'fp':'False Alarm'}

plot_vars = ['auroc_sample','auprc_sample','recall_event','tn', 'balanced_acc', 'fp']
plot_labels = [metric_labels[k] for k in plot_vars]

def flatten_tableone(df):
    new_col_names = {k:f'{k}' for k,v in df.loc['n'].to_dict(orient='records')[0].items() if v} 
    df = df.drop(('n',''),axis=0)
    new_rows = []
    for group in df.index.get_level_values(0).unique():
        if group == 'n':
            continue
        block = df.xs(group, level=0)
        # Insert a blank row with group label
        if 'mean' in group or 'median' in group:
            block.index = [group]
            new_rows.append(block)
        else:
            label_row = pd.DataFrame([[""] * (df.shape[1]-1)+[block.iloc[0,-1]]], columns=df.columns, index=[group])
            block.index = ['    ' + idx for idx in block.index]
            block.iloc[0,-1] = ''
            new_block = pd.concat([label_row, block])
            new_rows.append(new_block)
    flat_df = pd.concat(new_rows)
    flat_df.index.name = None
    flat_df = flat_df.rename(new_col_names,axis=1)
    return flat_df

def plotting(long_df, fig_path):
    n_montage = len(long_df['montage'].unique())
    fig, ax = plt.subplots(figsize=(8+n_montage, 6))
    sns.stripplot(
        x='metric',
        y='value',
        hue='montage',
        # dodge=True,
        data=long_df[long_df['metric'].isin(plot_vars[:-1])],
        order=plot_vars[:-1],
        size=3,jitter=0.2,
        dodge=True, alpha=.5, legend=False,zorder=0,
        palette=full_palete[:n_montage],
        ax = ax
    )
    sns.pointplot(
        x='metric',
        y='value',
        hue='montage',
        # dodge=True,
        data=long_df[long_df['metric'].isin(plot_vars[:-1])],
        order=plot_vars[:-1],
        estimator='mean',# median
        dodge=0.4+(n_montage-2)*0.1, linestyle="none", errorbar=("ci",95),#("pi", 50),
        marker="_", markersize=15, markeredgewidth=3,zorder=1,errwidth=1,color='black',
        ax=ax
    )

    ax2 = ax.twinx()
    sns.stripplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric']==plot_vars[-1]],
        size=3, jitter=0.2, dodge=True, alpha=.5, legend=False, zorder=0,
        palette=full_palete[:n_montage],
        ax=ax2
    )
    sns.pointplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric']==plot_vars[-1]],
        estimator='mean', dodge=0.4+(n_montage-2)*0.1, linestyle="none",
        errorbar=("ci", 95), marker="_", markersize=15, markeredgewidth=3,
        zorder=1, errwidth=1, color='black',
        ax=ax2
    )

    ax.legend().remove()
    ax2.legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_ylabel('Metric', fontsize=12)
    ax2.set_ylabel('False Alarm/h', fontsize=12)
    ax.set_xticks(ticks=list(range(len(plot_labels))), labels=plot_labels,fontsize=12, rotation=25)
    fig.savefig(fig_path,dpi=300)


def box_plotting(long_df, fig_path):
    n_montage = len(long_df['montage'].unique())
    fig, ax = plt.subplots(figsize=(6+n_montage, 6))
    sns.boxplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric'].isin(plot_vars[:-1])],
        order=plot_vars[:-1],
        boxprops=dict(alpha=0.7),
        gap=0.1, width=0.7,
        dodge=True, legend=False, zorder=0,
        palette=full_palete[:n_montage],
        ax = ax
    )
    # sns.pointplot(
    #     x='metric',
    #     y='value',
    #     hue='montage',
    #     # dodge=True,
    #     data=long_df[long_df['metric'].isin(plot_vars[:-1])],
    #     order=plot_vars[:-1],
    #     estimator='mean',# median
    #     dodge=0.4+(n_montage-2)*0.1, linestyle="none", errorbar=("ci",95),#("pi", 50),
    #     marker="_", markersize=15, markeredgewidth=3,zorder=1,errwidth=1,color='black',
    #     ax=ax
    # )

    ax2 = ax.twinx()
    sns.boxplot(
        x='metric',
        y='value',
        hue='montage',
        data=long_df[long_df['metric']==plot_vars[-1]],
        boxprops=dict(alpha=0.7),
        gap=0.1, width=0.7,
        dodge=True, zorder=0,
        palette=full_palete[:n_montage],
        ax=ax2
    )
    ax.legend().remove()
    ax2.legend(loc='upper right')
    # ax2.legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_ylabel('Metric', fontsize=12)
    ax2.set_ylabel('False Alarm/h', fontsize=12)
    ax.set_xticks(ticks=list(range(len(plot_labels))), labels=plot_labels,fontsize=12, rotation=25)
    fig.savefig(fig_path,dpi=300)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Run SPaRCNet prediction on folder of edf files, store probabilities"
    )

    # Define arguments
    parser.add_argument("-o", "--output", type=str, default='sparcnet/figures', help="Patient data to fine-tune on")
    parser.add_argument("--metric_folder", type=str, default='sparcnet/metrics', help="Patient data to fine-tune on")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Patient data to fine-tune on")
    parser.add_argument("--force", action='store_true', help="Force re-running")
    parser.add_argument("-t", "--thres", type=float, default=0.5, help="Patient data to fine-tune on")
    parser.add_argument("-s", "--setting", type=str, default='', help="Patient data to fine-tune on")
    # parser.add_argument("--multi_comp",  action='store_true',  help="Patient data to fine-tune on")
    parser.add_argument("--do_plot",  action='store_true',  help="Patient data to fine-tune on")
    
    # Parse the arguments
    params = vars(parser.parse_args())
    figure_folder = params['output']
    metric_folder = params['metric_folder']
    force = params['force']
    thres = params['thres']

    setting_folder = f"thres{thres:.1f}"
    if params['setting']:
        setting_folder += f"_{params['setting']}"
        if params['setting'] == 'optimal':
            setting_folder = 'thres_optimal'

    metric_folder = os.path.join(metric_folder, setting_folder)
    figure_folder = os.path.join(figure_folder, setting_folder)
    stats_folder = os.path.join('sparcnet/stats', setting_folder)
    os.makedirs(figure_folder,exist_ok=True)
    os.makedirs(stats_folder,exist_ok=True)
    
    if params['montage'] == 'all':
        montage = montage_list
    else:
        montage = params['montage'].split(',')

    
    file_names = {'':'patient_metrics.csv',
                  '_filtered':'patient_metrics_filtered.csv'}

    patient_map = pd.read_csv('../emu_dataset/emu_patient_info.csv', dtype={'patient_id':str})
                
    for plot_setting, file_name in file_names.items():
        # full_metric = pd.read_csv(os.path.join(metric_folder, 'full', file_name))
        all_metrics = []
        for m in montage:
            metric_file = os.path.join(metric_folder, m, file_name)
            metrics = pd.read_csv(metric_file, index_col=0)
            metrics['montage'] = m
            all_metrics.append(metrics)
        all_metrics = pd.concat(all_metrics,axis=0).reset_index().rename(columns={'index':'patient_id'})
        all_metrics['patient_id'] = all_metrics['patient_id'].astype(str)

        if params['do_plot']:
            for m in montage:
                if m == 'full':
                    continue
                fig_path = os.path.join(figure_folder, m+plot_setting+'.png')
                stats_file = os.path.join(stats_folder, m+plot_setting+'.csv')
                tmp_metrics = all_metrics[all_metrics['montage'].isin(['full',m])]
                long_df = pd.melt(tmp_metrics, id_vars=['montage'], var_name='metric')
                plotting(long_df, fig_path)
                stats = []
                for score in plot_vars:
                    group1 = long_df[(long_df['metric'] == 'full') & (long_df['metric'] == score)]['value'].to_list()
                    group2 = long_df[(long_df['metric'] == m) & (long_df['metric'] == score)]['value'].to_list()
                    stat, p = wilcoxon(group1, group2, alternative='greater')
                    stats.append([score, np.mean(group1), np.mean(group2), stat, p])
                stats = pd.DataFrame(stats, columns = ['metric', 'mean1', 'mean2','stat', 'p'])
                stats.to_csv(stats_file, index=False)

            # multi comps
            for multi, multi_montages in multi_comp.items():
                long_df = pd.melt(all_metrics[all_metrics['montage'].isin(multi_montages)], id_vars=['montage'], var_name='metric')    
                box_plotting(long_df, os.path.join(figure_folder, 'multi_'+multi+plot_setting+'.png'))

        # comparison, and comparison by diagnosis/laterality/location
        table1 = TableOne(all_metrics, columns=plot_vars, #nonnormal=plot_vars, 
                          groupby='montage', missing=False, overall=False, pval=False, decimals = 3, labels=plot_labels)
        table1 = table1.tableone['Grouped by montage']
        flatten_tableone(table1)[montage_list].T.to_csv(os.path.join(stats_folder, 'comparison'+plot_setting+'.csv'))
        
        # by epilepsy type
        metrics_with_info = all_metrics.merge(patient_map, on='patient_id', how='left')
        metrics_with_missing_type = metrics_with_info[(metrics_with_info['epilepsy_type'].isna()) & (metrics_with_info['montage']=='full')]
        print(f"{len(metrics_with_missing_type)} of {len(metrics_with_info[metrics_with_info['montage']=='full'])} is dropped due to missing epilepsy type")
        metrics_with_info = metrics_with_info[~metrics_with_info['epilepsy_type'].isna()]
        metrics_with_info['tmp_group'] = metrics_with_info['epilepsy_type'] + '_' + metrics_with_info['montage']
        table1 = TableOne(metrics_with_info, columns=plot_vars, #nonnormal=plot_vars, 
                          groupby='tmp_group', missing=False, overall=False, pval=False, decimals = 3, labels=plot_labels)
        table1 = table1.tableone['Grouped by tmp_group']
        flatten_tableone(table1).T.to_csv(os.path.join(stats_folder, 'comparison_by_type'+plot_setting+'.csv'))

        # by laterality
        metrics_with_info = all_metrics.merge(patient_map, on='patient_id', how='left')
        metrics_with_missing_type = metrics_with_info[(metrics_with_info['laterality'].isna()) & (metrics_with_info['montage']=='full')]
        print(f"{len(metrics_with_missing_type)} of {len(metrics_with_info[metrics_with_info['montage']=='full'])} is dropped due to missing laterality")
        metrics_with_info = metrics_with_info[~metrics_with_info['laterality'].isna()]
        metrics_with_info['tmp_group'] = metrics_with_info['laterality'] + '_' + metrics_with_info['montage']
        table1 = TableOne(metrics_with_info, columns=plot_vars, #nonnormal=plot_vars, 
                          groupby='tmp_group', missing=False, overall=False, pval=False, decimals = 3, labels=plot_labels)
        table1 = table1.tableone['Grouped by tmp_group']
        flatten_tableone(table1).T.to_csv(os.path.join(stats_folder, 'comparison_by_laterality'+plot_setting+'.csv'))

        # by location
        metrics_with_info = all_metrics.merge(patient_map, on='patient_id', how='left')
        metrics_with_missing_type = metrics_with_info[(metrics_with_info['location'].isna()) & (metrics_with_info['montage']=='full')]
        print(f"{len(metrics_with_missing_type)} of {len(metrics_with_info[metrics_with_info['montage']=='full'])} is dropped due to missing laterality")
        metrics_with_info = metrics_with_info[~metrics_with_info['location'].isna()]
        metrics_with_info['tmp_group'] = metrics_with_info['location'] + '_' + metrics_with_info['montage']
        table1 = TableOne(metrics_with_info, columns=plot_vars, #nonnormal=plot_vars, 
                          groupby='tmp_group', missing=False, overall=False, pval=False, decimals = 3, labels=plot_labels)
        table1 = table1.tableone['Grouped by tmp_group']
        flatten_tableone(table1).T.to_csv(os.path.join(stats_folder, 'comparison_by_location'+plot_setting+'.csv'))
        

