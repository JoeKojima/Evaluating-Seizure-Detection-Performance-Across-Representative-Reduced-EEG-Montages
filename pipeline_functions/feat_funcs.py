import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.signal import welch
from scipy.integrate import simpson
from scipy.ndimage import binary_opening, binary_closing
from sklearn.svm import OneClassSVM

# ————————general————————


def bandpower(
    data,
    fs,
    band,
    win_size=None,
    relative=False,
):
    """Adapted from https://raphaelvallat.com/bandpower.html
    Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array or 2d-array
        Input signal in the time-domain. (time by channels)
    fs : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    win_size : float
        Length of each window in seconds.
        If None, win_size = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : np.ndarray
        Absolute or relative band power. channels by bands
    """
    band = np.asarray(band)
    assert len(band) == 2, "CNTtools:invalidBandRange"
    assert band[0] < band[1], "CNTtools:invalidBandRange"
    if np.ndim(data) == 1:
        data = data[:, np.newaxis]
    nchan = data.shape[1]
    bp = np.nan * np.zeros(nchan)
    low, high = band

    # Define window length
    # if win_size is not None:
    #     nperseg = int(win_size * fs)
    # else:
    #     nperseg = int((2 / low) * fs)

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data.T, fs)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    if psd.ndim == 2:
        bp = simpson(psd[:, idx_band], dx=freq_res)
    elif psd.ndim == 1:
        bp = simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simpson(psd, dx=freq_res)

    return bp


# ————————————sparcnet————————————


def extract_seiz_ranges(true_data):
    # the extracted start and stop can should be used like start:stop to extract data
    diff_data = np.diff(np.concatenate([[0], np.ravel(true_data), [0]]))
    starts = np.where(diff_data == 1)[0].tolist()
    stops = np.where(diff_data == -1)[0].tolist()
    return list(zip(starts, stops))


def nan_aware_uniform_filter1d(arr, size):
    arr = np.asarray(arr, dtype=float)
    half = size // 2
    padded = np.pad(arr, pad_width=half, mode="edge")  # nearest-like padding
    result = np.full_like(arr, np.nan)

    for i in range(len(arr)):
        window = padded[i : i + size]
        if np.isfinite(window).any():
            result[i] = np.nanmean(window)  # mean ignoring NaNs
    return result


def smooth_pred(pred):
    pred = np.array(pred)
    if len(pred) == 1:
        return pred.astype(int)
    smoothed = pred.copy()
    for win in range(2, 4):
        smoothed = binary_opening(smoothed, structure=np.ones(win))  # remove short 1s
        smoothed = binary_closing(smoothed, structure=np.ones(win))  # remove short 0s
        smoothed[0] = smoothed[1]
        smoothed[-1] = smoothed[-2]

    return smoothed.astype(int)


def get_events(pred, gap_num=2, min_event_num=10, return_pred=False):
    smoothed_pred = smooth_pred(pred)
    new_pred = np.zeros_like(smoothed_pred)
    sz_events = np.array(extract_seiz_ranges(smoothed_pred))

    if sz_events.shape[0] == 0:
        if return_pred:
            return np.array([]), new_pred
        else:
            return np.array([])

    # Merge events that are close together
    start_times = sz_events[:, 0]
    end_times = sz_events[:, 1]

    merged_start = [start_times[0]]
    merged_end = [end_times[0]]

    for i in range(1, len(start_times)):
        if start_times[i] - merged_end[-1] <= gap_num:
            # Merge the events
            merged_end[-1] = end_times[i]
        else:
            # Start a new event
            merged_start.append(start_times[i])
            merged_end.append(end_times[i])

    sz_events = np.array(list(zip(merged_start, merged_end)))

    # Filter short events
    durations = sz_events[:, 1] - sz_events[:, 0]
    sz_events = sz_events[durations >= min_event_num]

    if sz_events.shape[0] == 0:
        if return_pred:
            return np.array([]), new_pred
        else:
            return np.array([])

    # Merge events that are close together
    start_times = sz_events[:, 0]
    end_times = sz_events[:, 1]

    merged_start = [start_times[0]]
    merged_end = [end_times[0]]

    for i in range(1, len(start_times)):
        if start_times[i] - merged_end[-1] <= 4:
            # Merge the events
            merged_end[-1] = end_times[i]
        else:
            # Start a new event
            merged_start.append(start_times[i])
            merged_end.append(end_times[i])

    sz_events = np.array(list(zip(merged_start, merged_end)))

    for start, end in sz_events:
        new_pred[start:end] = 1
    if return_pred:
        return sz_events, new_pred
    else:
        return sz_events


def get_event_smoothed_pred(smoothed_pred, gap_num=2, min_event_num=10):
    new_pred = np.zeros_like(smoothed_pred)
    sz_events = np.array(extract_seiz_ranges(smoothed_pred))

    if sz_events.shape[0] == 0:
        return new_pred

    # Merge events that are close together
    start_times = sz_events[:, 0]
    end_times = sz_events[:, 1]

    merged_start = [start_times[0]]
    merged_end = [end_times[0]]

    for i in range(1, len(start_times)):
        if start_times[i] - merged_end[-1] <= gap_num:
            # Merge the events
            merged_end[-1] = end_times[i]
        else:
            # Start a new event
            merged_start.append(start_times[i])
            merged_end.append(end_times[i])

    sz_events = np.array(list(zip(merged_start, merged_end)))

    # Filter short events
    durations = sz_events[:, 1] - sz_events[:, 0]
    sz_events = sz_events[durations >= min_event_num]
    for start, end in sz_events:
        new_pred[start:end] = 1
    return new_pred


# ————————————svm————————————


#### feature extraction
def teager_operator(x):
    """
    Computes the Teager Energy Operator for a 1D signal.
    The output will be 2 samples shorter than the input.
    """
    return x[1:-1] ** 2 - x[:-2] * x[2:]


def mean_curve_length(x):
    return np.log1p(np.mean(np.abs(np.diff(x))))


def mean_energy(x):
    return np.log1p(np.mean(x**2))


def mean_teager_energy(x):
    te = teager_operator(x)
    return np.log1p(np.mean(te))


def extract_features(eeg, fs=200, win_len=1.0, step_size=0.5):
    """
    epoch: 1D np.array EEG signal
    fs: sampling frequency (e.g. 200 Hz)
    win_len: window length in seconds
    step_size: window overlap step in seconds
    """
    N = int(win_len * fs)
    step = int(step_size * fs)
    features = []

    for start in range(0, len(eeg) - N + 1, step):
        x = eeg[start : start + N]
        cl = mean_curve_length(x)
        e = mean_energy(x)
        te = mean_teager_energy(x)
        features.append([cl, e, te])
    return np.array(features)


#### classifier
def train_one_class_svm(X_train, nu=0.1, gamma=1.0):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X_train)
    return model


def compute_novelty_scores(model, X_test):
    preds = model.predict(X_test)  # +1 or -1
    return preds


def estimate_outlier_fraction(y_pred, n=10):
    y = (y_pred == -1).astype(float)
    cumsum = np.cumsum(y)
    out = cumsum.copy()
    out[n:] = cumsum[n:] - cumsum[:-n]
    # variable window size at start
    counts = np.minimum(np.arange(1, len(y) + 1), n)

    return out / counts


# === Seizure Detection Based on ν̂ Hypothesis Test ===
def detect_seizure(nu_hat, threshold=0.8):
    return (nu_hat >= threshold).astype(int)


# === Apply Persistence Filter ===
def extract_seiz_ranges(true_data):
    # the extracted start and stop can should be used like start:stop to extract data
    diff_data = np.diff(np.concatenate([[0], np.ravel(true_data), [0]]))
    starts = np.where(diff_data == 1)[0].tolist()
    stops = np.where(diff_data == -1)[0].tolist()
    return list(zip(starts, stops))


def apply_persistence(z, refractory_sec=180, step_sec=0.5):
    refractory_steps = int(refractory_sec / step_sec)
    pred_events = extract_seiz_ranges(z)
    detection = np.zeros_like(z)
    if len(pred_events) == 0:
        return detection
    starts = [pred_events[0][0]]
    ends = [pred_events[0][1]]
    if len(pred_events) > 1:
        for i in range(1, len(pred_events)):
            if pred_events[i][0] - ends[-1] < refractory_steps:
                pass
            else:
                starts.append(pred_events[i][0])
                ends.append(pred_events[i][1])
    for start, end in zip(starts, ends):
        detection[start:end] = 1
    return detection


# seizure onset detection
def get_onset_and_spread(
    sz_prob,
    threshold=None,
    ret_smooth_mat=True,  # True
    filter_w=5,  # seconds
    rwin_size=5,  # seconds #10
    rwin_req=4,  # seconds #9
    w_size=1,
    w_stride=0.5,
):

    sz_clf = (sz_prob > threshold).reset_index(drop=True)
    filter_w_idx = np.floor((filter_w - w_size) / w_stride).astype(int) + 1
    sz_clf = pd.DataFrame(
        sig.ndimage.median_filter(sz_clf, size=filter_w_idx, mode="nearest", origin=0),
        columns=sz_prob.columns,
    )
    seized_idxs = np.any(sz_clf, axis=1)
    rwin_size_idx = np.floor((rwin_size - w_size) / w_stride).astype(int) + 1
    rwin_req_idx = np.floor((rwin_req - w_size) / w_stride).astype(int) + 1
    sz_spread_idxs_all = (
        sz_clf.rolling(window=rwin_size_idx, center=False)
        .apply(lambda x: (x == 1).sum() > rwin_req_idx)
        .dropna()
        .reset_index(drop=True)
    )
    sz_spread_idxs = sz_spread_idxs_all.loc[seized_idxs]
    extended_seized_idxs = np.any(sz_spread_idxs, axis=1)
    if sum(extended_seized_idxs) > 0:
        # Get indices into the sz_prob matrix and times since start of matrix that the seizure started
        first_sz_idxs = sz_spread_idxs.loc[extended_seized_idxs].idxmax(axis=0)
        sz_idxs_arr = np.array(first_sz_idxs)
        sz_order = np.argsort(first_sz_idxs)
        sz_idxs_arr = first_sz_idxs.iloc[sz_order].to_numpy()
        sz_ch_arr = first_sz_idxs.index[sz_order].to_numpy()
        # sz_times_arr = self.get_win_times(len(sz_clf))[sz_idxs_arr]
        # sz_times_arr -= np.min(sz_times_arr)
        # sz_ch_arr = np.array([s.split("-")[0] for s in sz_ch_arr]).flatten()
    else:
        sz_ch_arr = []
        sz_idxs_arr = np.array([])
    sz_idxs_df = pd.DataFrame(sz_idxs_arr.reshape(1, -1), columns=sz_ch_arr)
    sz_idxs_df.drop(
        columns=sz_idxs_df.columns[(sz_idxs_df == 0).all()]
    )  # zhiyu: drop those start at 0
    if ret_smooth_mat:
        return sz_idxs_df, sz_spread_idxs_all
    else:
        """sz_idx_df is the onset time of each channel"""
        return sz_idxs_df
