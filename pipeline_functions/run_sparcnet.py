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

def load_edf_file(file_name):
    # suppose f is an interictal file
    raw = mne.io.read_raw_edf(file_name, preload=True, verbose = 0)
    fs = raw.info['sfreq']
    df = raw.to_data_frame().set_index('time')
    times = raw.times
    annotations = raw.annotations
    label = np.zeros(len(times)).astype(int)
    if annotations:
        for anno in annotations:
            sz_onset = anno['onset']
            sz_dura = anno['duration']
            sz_end = sz_onset+sz_dura
            label[(times >= sz_onset)&(times <= sz_end)] = 1
    label_df = pd.DataFrame({'time':times,'labels':label})
    return raw, df, label_df, fs

import torch
import torch.nn as nn
import torch.nn.functional as F

sparcnet_path = os.path.join(working_dir,'SPaRCNet')
sys.path.append(sparcnet_path)
from DenseNetClassifier import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cnn = torch.load(sparcnet_path+"/model_1130.pt", map_location=torch.device(device), weights_only=False)
model_cnn.eval()

feat_setting = {'name':'sparcnet',
                'win':int(10), 'stride':int(2),
                'reref':'BIPOLAR', 'resample':200,
                'lowcut':1, 'highcut':40} # in seconds

def sparcnet_single(data, fs):
    """Do seizure prediciton on a 10-second clip.
    Data should be a pd dataframe

    Args:
        data (_type_): _description_
        fs (_type_): _description_
    """
    if 'Fz-Cz' in data.columns:
        data = data.drop(columns=['Fz-Cz'])
    if 'Cz-Pz' in data.columns:
        data = data.drop(columns=['Cz-Pz'])
    data = data.values
    data = bandpass_filter(data, fs, lo = feat_setting['lowcut'], hi = feat_setting['highcut'])
    data = downsample(data, fs, feat_setting['resample'])
    data = np.where(data<=500, data, 500)
    data = np.where(data>=-500, data, -500)
    data = torch.from_numpy(data).float()
    data = data.T.unsqueeze(0)
    data = data.to(device)
    output, _ = model_cnn(data)
    sz_prob = F.softmax(output,1).detach().cpu().numpy().flatten()
    return sz_prob

def custom_bipolar(df,pairs):
    filtered = df['filtered']
    columns = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
               'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
    data = df['BIPOLAR'][columns]
    data.loc[:,columns] = 0
    for p in pairs:
        ch1, ch2 = p.split('-')
        # try:
        ch_data = filtered[ch1]-filtered[ch2]
        data.loc[:,[c for c in columns if c.startswith(ch1)][0]] = ch_data
        # except:
        #     pass
    return data

def epiminder_simulate(raw):
    # simulate CP5-CP1 CP2-CP6, fill in C3-P3, C4-P4
    # Example: you already have a raw object
    fs = raw.info['sfreq']
    n_times = raw.n_times
    new_ch_names = ['CP5', 'CP6', 'CP1', 'CP2']    # any names you want
    new_ch_types = ['eeg'] * len(new_ch_names)
    new_data = np.zeros((len(new_ch_names), n_times))
    new_info = mne.create_info(new_ch_names, fs, new_ch_types)
    new_raw = mne.io.RawArray(new_data, new_info)
    raw.add_channels([new_raw], force_update_info=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    raw.info['bads'] = ['CP5', 'CP6', 'CP1', 'CP2']  # mark as bad so they get interpolated
    raw.interpolate_bads(reset_bads=True)
    
    df = raw.to_data_frame().set_index('time')
    new_prepro = Preprocessor()
    new_prepro.fit({'samplingFreq':fs, 'samplingFreqRaw':fs, 'channelNames':df.columns, 'studyType':'eeg', 'numberOfChannels':df.shape[1]})
    df = new_prepro.preprocess(df)
    filtered = df['filtered']
    cp5 = filtered['CP5']-filtered['CP1']
    cp6 = filtered['CP6']-filtered['CP2']
    columns = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
    data = df['BIPOLAR'][columns]
    data.loc[:,columns] = 0
    data.loc[:,'C3-P3'] = cp5
    data.loc[:,'C4-P4'] = cp6
    return data

montage_dict = {'full':['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                        'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],
                'uneeg_left_front':['F7-T3'],
                'uneeg_left_back':['T3-T5'],
                'uneeg_right_front':['F8-T4'],
                'uneeg_right_back':['T4-T6'],
                'uneeg_right':['F8-T4','T4-T6'],
                'uneeg_left':['F7-T3','T3-T5'],
                'uneeg_bilateral4':['F7-T3','T3-T5','F8-T4', 'T4-T6'],
                'uneeg_bilateral_back2':['T3-T5','T4-T6'],
                'uneeg_bilateral_front2':['F7-T3','F8-T4'],
                'uneeg_vert_left':lambda df: custom_bipolar(df,['C3-T3']),
                'uneeg_vert_right':lambda df: custom_bipolar(df,['C4-T4']),
                'uneeg_diag_left_front':lambda df: custom_bipolar(df,['F3-T3']),
                'uneeg_diag_left_back':lambda df: custom_bipolar(df,['P3-T3']),
                'uneeg_diag_right_front':lambda df: custom_bipolar(df,['F4-T4']),
                'uneeg_diag_right_back':lambda df: custom_bipolar(df,['P4-T4']),
                'uneeg_diag_bilateral_front':lambda df: custom_bipolar(df,['F3-T3','F4-T4']),
                'uneeg_diag_bilateral_back':lambda df: custom_bipolar(df,['P3-T3','P4-T4']),
                'uneeg_vert_bilateral':lambda df: custom_bipolar(df,['C3-T3','C4-T4']),
                'epiminder_2':['C3-P3','C4-P4'],
                'epiminder_4':['C3-P3','C4-P4','T3-T5','T4-T6'],
                # 'epiminder_simulate':lambda x: epiminder_simulate(x),
                'zero':[]}

def process_file(file_name):
    warnings.filterwarnings('ignore')
    raw, df, label_df, fs = load_edf_file(file_name)
    prepro = Preprocessor()
    prepro.fit({'samplingFreq':fs, 'samplingFreqRaw':fs, 'channelNames':df.columns, 'studyType':'eeg', 'numberOfChannels':df.shape[1]})
    preprocessed = prepro.preprocess(df)
    window_starts = np.arange(0,df.index[-1]-feat_setting['win']+1,feat_setting['stride'])
    for m in montage:
        prob_path = os.path.join(prob_folder,m,file_name.split('/')[-1].replace('.edf','.csv'))
        if os.path.exists(prob_path) and not force:
            continue
        montage_processor = montage_dict[m]
        if isinstance(montage_processor,list):
            data_df = preprocessed['BIPOLAR']
            data_df = data_df[montage_dict['full']]
            data_df.loc[:,~data_df.columns.isin(montage_processor)] = 0
        elif 'simulate' in m:
            data_df = montage_processor(raw)
        else:
            data_df = montage_processor(preprocessed)
        sz_prob_df = []
        for win_start in window_starts:
            clip = data_df.loc[(data_df.index >= win_start) & (data_df.index < win_start + feat_setting['win'])]
            feat_index = clip.index[-1]
            feat_label = label_df.loc[(data_df.index >= win_start) & (data_df.index < win_start + feat_setting['win']),'labels'].any().astype(int)
            sz_prob = sparcnet_single(clip, fs)
            sz_prob_df.append(pd.DataFrame([np.r_[sz_prob,[feat_label]]],columns = ['SZ','LPD','GPD','LRDA','GRDA','OTHER','label'], index=[feat_index]))
        sz_prob_df = pd.concat(sz_prob_df)
        sz_prob_df.to_csv(prob_path)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description="Run SPaRCNet prediction on folder of edf files, store probabilities"
    )

    # Define arguments
    parser.add_argument("-f", "--folder", type=str, default='', help="files in folder to process")
    parser.add_argument("-o", "--output", type=str, default='sparcnet/prob', help="Patient data to fine-tune on")
    parser.add_argument("-m", "--montage", type=str, default='all', help="Patient data to fine-tune on")
    parser.add_argument("--force", action='store_true', help="Force re-running")
    
    # Parse the arguments
    params = vars(parser.parse_args())

    if not params['folder']:
        sz_folder = '/mnt/sauce/littlab/users/haoershi/emu_dataset/seizure'
        sz_files = sorted(glob.glob(sz_folder+'/*.edf'))
        iic_folder = '/mnt/sauce/littlab/users/haoershi/emu_dataset/interictal'
        iic_files = sorted(glob.glob(iic_folder+'/*.edf'))
        all_files = sz_files+iic_files
    else:
        all_files = glob.glob(params['folder'])

    prob_folder = params['output']
    os.makedirs(prob_folder,exist_ok=True)

    if params['montage'] == 'all':
        montage = list(montage_dict.keys())
    else:
        montage = params['montage'].split(',')
    for m in montage:
        os.makedirs(os.path.join(prob_folder,m),exist_ok=True)
    force = params['force']
    with tqdm(total=len(all_files),desc = 'Processing block'):
        # for file_name in all_files:
        #     process_file(file_name)
        results = Parallel(n_jobs=40)(delayed(process_file)(file_name) for file_name in all_files)
    # for file_name in tqdm(all_files,total=len(all_files)):
        
