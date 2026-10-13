#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:02:46 2025

MXPol data processing raw --> L0, converting .dat-files to .nc with:
- elevation & azimuth correction
- dropped data filtering: removing spurious bins/time steps from data, including new 
variable 'Mask' that is False for timesteps that have been removed through 
autovariance-based outlier detection
- omitting (not converting) files where frequency tracking was lost

@author: clerx
"""
#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import re
import gc
import pyart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask
import multiprocessing
import time

from datetime import datetime, timedelta
from pyjacopo import parse_config, read_raw_data
from pyjacopo import process_dataset, write_cfradial
from scipy import ndimage
from scipy.stats import zscore

dask.config.set({
    "distributed.worker.memory.target": 0.5,
    "distributed.worker.memory.spill": 0.7,
    "distributed.worker.memory.pause": 0.8,
    "distributed.worker.memory.terminate": 0.9,
    "distributed.worker.multiprocessing-method": "spawn",
    "distributed.worker.use-filelock": False
})

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


#%%

corr_el = -2.3
corr_az = 190
base_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MXPol'
config = parse_config(os.path.join(base_dir, 'config_spec.yml'))

pltConfig_MXPol = {
    'xlim_RHI': [-20, 20],
    'ylim_RHI': [0, 15],
    'xlim_sector': [-4, 20],
    'ylim_sector': [-4, 20],
    'ar': 1,
    'vdop_min': -11,
    'vdop_max': 5
}

limits_Zdr = {
    'SNRh': ('exclude_below', 5),
    'SNRv': ('exclude_below', 5),
    'Rhohv': ('exclude_below', 0.95),
    # 'ml_idx': ('exclude_above', 0.1)
    }

limits_SNR = {
    'SNRh': ('exclude_below', -5),
    'SNRv': ('exclude_below', -5),
    }

limits_clutter = {
    'SNRh': ('exclude_below', 3),
    'SNRv': ('exclude_below', 3),
    }

scantimes = {
    'Zdr': 123,
    'RHI': 49,
    'sector': 43,
    }

plotvars = {'Zh': 'Reflectivity', 
            'Zdr': 'Diff. reflectivity', 
            'SNRh': 'SNR at hor. pol.', 
            'SNRv': 'SNR at vert. pol.'
    }

plotvals = {'Zh': (-20, 30, 'Spectral_r'),
            'Zdr': (-0.5, 4, 'plasma'),
            'SNRh': (-5, 5, 'YlOrRd'),
            'SNRv': (-5, 5, 'YlOrRd')}

pattern = re.compile(r"(XPOL-\d{8}-\d{6})")
pattern_rhi = re.compile("(RHI)")
pattern_sector = re.compile("(sector)")

""" list of file names (strings) to be excluded due to lost frequency tracking.
Second string is name of first OK scan, list potentially omits some Zdr-scans 
that could already be useless since this list was based on quicklooks (Zdr was 
at the end of scanning sequence so starting file is always the first valid one).
"""
lost_frequency_times = [
     ('20241129-093758', '20241129-111859'),
     ('20241209-061913', '20241209-062252'),
     ('20241210-045727', '20241210-065155'),
     ('20241211-005731', '20241211-131323'),
     ('20241212-065545', '20241212-065937'),
     ('20241213-090121', '20241214-220406'),
     ('20241215-205044', '20241216-084431'),
     ('20241217-014915', '20241220-094708'),
     ('20241222-081854', '20241222-083141'),
     ('20241224-123639', '20241224-123914'), #check if elevation correction goes OK
     ('20250103-072540', '20250103-074247'),
     ('20250104-021318', '20250106-062100'), # from here onwards maybe some files already OK  (or bad) before the mentioned files
     ('20250107-010046', '20250107-062514'),
     ('20250111-013936', '20250114-063819'), 
     ('20250115-194048', '20250118-144705'), # plenty of files not recorded properly (0 bytes)
     ('20250120-071620', '20250121-120002', )
 ]

#%%

def generate_date_range(start_date, end_date):
    """
    Generates a list of dates between start_date and end_date (inclusive).
    """
    start = datetime.strptime(start_date, "%Y-%m-%d") if type(start_date) is str else start_date
    end = datetime.strptime(end_date, "%Y-%m-%d") if type(end_date) is str else end_date
    delta = timedelta(days=1)

    dates = []
    while start <= end:
        dates.append(start)
        start += delta
    return dates


def check_time(time, intervals=lost_frequency_times):
    for start_str, end_str in intervals:
        start_time = datetime.strptime(start_str, '%Y%m%d-%H%M%S')
        end_time = datetime.strptime(end_str, '%Y%m%d-%H%M%S')
        if start_time <= time <= end_time:
            return True
    return False


def elevation_correction(elevation_data, elevation_correction=corr_el):
    elevation_data = elevation_data + elevation_correction
    elevation_data = np.where(elevation_data > 0, 
                              elevation_data - 180, 
                              elevation_data + 180)    
    return elevation_data


def azimuth_correction(azimuth_data, azimuth_correction=corr_el):
    azimuth_data = azimuth_data + azimuth_correction
    azimuth_data = np.where(azimuth_data > 360,
                            azimuth_data - 360,
                            azimuth_data)
    return azimuth_data    


def mean_std(input_array, axis):
    if axis == 'time':
        ax = 0
    elif axis == 'range':
        ax = 1
    else:
        print(f"Error: axis {ax} should be specifiied ('time' or 'range').")
    mean = np.nanmean(input_array, ax)
    std = np.nanstd(input_array, ax)
    
    return (mean, std)


def normalise(input_array, lower_bound, upper_bound):
    """
    Normalise values in input_array that lie within [lower_bound, upper_bound] to the range [0, 1].
    Parameters:
    - input_array: np.ndarray or np.ma.MaskedArray
    - lower_bound: float, minimum value to consider
    - upper_bound: float, maximum value to consider
    
    Returns:
    - np.ma.MaskedArray with values normalized to [0, 1]
    """
    if lower_bound == upper_bound:
        raise ValueError("Lower and upper bounds cannot be the same")
        
    masked = np.ma.masked_where((input_array < lower_bound) | (input_array > upper_bound), input_array)
    normalised = (masked - lower_bound) / (upper_bound - lower_bound)
    
    return normalised


def apply_filter(radar_obj, conditions):
    """
    Create gate filter for pyart object based on dictionary containing conditions.
    conditions-dictionary in the shape
    {[variable name]: (type of limit, threshold value)}
    
    returns gate filter object
    """
    gate_filter =  pyart.filters.GateFilter(radar_obj)
    
    for field, (method, threshold) in conditions.items():
        if method == "exclude_below":
            gate_filter.exclude_below(field, threshold)
        elif method == "exclude_above":
            gate_filter.exclude_above(field, threshold)
        elif method == "exclude_equal":
            gate_filter.exclude_equal(field, threshold)
        elif method == "exclude_inside":
            gate_filter.exclude_inside(field, *threshold)  # Expecting (min, max) tuple
        elif method == "exclude_outside":
            gate_filter.exclude_outside(field, *threshold)  # Expecting (min, max) tuple
    
    return gate_filter
        

def create_snr_mask(radar, limits_dict=limits_SNR):
    """
    Create boolean mask from limits dictionary.
    True = gate should be masked.
    """
    combined_mask = None

    for field, (method, threshold) in limits_dict.items():
        data = radar.fields[field]['data']
        data = np.ma.getdata(data)

        if method == 'exclude_below':
            mask = data < threshold
        elif method == 'exclude_above':
            mask = data > threshold
        elif method == 'exclude_equal':
            mask = data == threshold
        elif method == 'exclude_inside':
            mask = (data > threshold[0]) & (data < threshold[1])
        elif method == 'exclude_outside':
            mask = (data < threshold[0]) | (data > threshold[1])
        else:
            continue

        combined_mask = mask if combined_mask is None else (combined_mask | mask)

    return combined_mask


def rolling_mean(np_array, window, axis):
    """
    Compute rolling mean along specified axis (time or range) while preserving shape.
    For 1D arrays: computes rolling mean along the single axis.
    For 2D arrays: computes rolling mean along the last axis (range dimension)
    """    
    assert window %2 == 1 # window size must be odd
    
    if axis == 'time':
        axis = 0
    elif axis == 'range':
        axis = 1
    else:
        print(f"axis {axis} should be 'time' or 'range'")
    
    # Ensure input is at least 2D (if 1D, make it (1, N))
    np_array = np.atleast_2d(np_array)
    
    return ndimage.uniform_filter1d(np_array, size=window, axis=axis, mode='nearest')


def rolling_mean_std(np_array, window, axis='time'):
    """
    Compute rolling standard deviation along a specified axis.
    """
    assert window % 2 == 1
    
    if axis == 'time':
        axis = 0
    elif axis == 'range':
        axis = 1
    else:
        print(f"axis {axis} should be 'time' or 'range'")
        
    mean_sq = ndimage.uniform_filter1d(np_array**2, window, axis, mode='nearest')
    mean = ndimage.uniform_filter1d(np_array, window, axis, mode='nearest')
    
    return np.sqrt(mean_sq - mean**2)
    

def compute_autovariance(data, lag=1):
    """
    Compute the autocovariance of a time series for a given lag.
    
    Parameters:
        data (array): input time series
        lag (int): lag at which to compute the autovariance
        
    Returns:
        float: the autocovariance at the given lag
    """
    mean = np.nanmean(data)
    n = len(data)
    
    autocov = np.sum((data[:n-lag] - mean) * (data[lag:] - mean)) / (n - 1)
    return autocov


def compute_autovariance_2d(data, lag=1, axis='range'):
    """
    Compute the autocovariance of a time series per range gate.
    
    Parameters:
        data (array): input time series (axis 0 = range, axis 1 = time)
        lag (int): lag at which to compute the autovariance
        
    Returns:
        float: the autocovariance at the given lag
    """
    ts, rgs = data.shape     # number of timesteps & rangegates
    if axis == 'range': 
        ax = rgs
        step = ts
    elif axis == 'time':
        ax = ts
        step = rgs
    else:
        print(f"Incorrectly specified axis {axis}, should be 'time' or 'range'")
        
    # calculate average and autocovariance along the specififi
    autocov = np.zeros(ax)
    for i in range(ax):
        series = data[:, i] if axis == 'range' else data[i, :]
        
        if np.isnan(series).all():
            continue
        
        if series.mask.all() == True:
            continue
        
        mean = np.nanmean(series)
        autocov[i] = np.nansum((series[:step - lag] - mean) * (series[lag:] - mean)) / (step - 1)
    
    return autocov


def detect_outliers_autovariance(data, window=3, lag=1, threshold=2.5, axis='time'):
    """
    Detect outliers based on autovariance.
    
    Parameters:
    data (array-like): The input time series.
    lag (int): The lag at which to compute autovariance.
    threshold (float): Z-score threshold for outlier detection.
    
    Returns:
    list: Indices of detected outliers.
    """    
    
    # Compute autovariance for each window
    autovars = np.array([compute_autovariance(np.array(data[i:i+window], copy=True), lag) for i in range(len(data) - window + 1)])
    valid_mask = ~np.isnan(autovars)
    z_scores = np.full_like(autovars, np.nan)
    z_scores[valid_mask] = zscore(autovars[valid_mask])
    
    # Find outliers based on threshold
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    centered_outliers = outlier_indices + window // 2
    centered_outliers = centered_outliers[(centered_outliers >= window // 2) & (centered_outliers < len(data))]
    
    return centered_outliers


def detect_outliers_autovariance_2d(data, window=3, lag=1, threshold=2.5, fraction=0.4):
    """
    Detect outliers based on autovariance array.
    
    Parameters:
    data (2D array): The input radar field with shape (time, range).
    window (int): Window size for computing local autovariance.
    lag (int): The lag at which to compute autovariance.
    threshold (float): Z-score threshold for outlier detection.
    fraction (float): if > than this % of data missing, mask whole ray
    
    Returns:
    mask_2d (2D boolean array): Mask where True indicates an outlier.
    """
    timesteps, ranges = data.shape
    mask_2d = np.full((data.shape), False)
    
    autovariance_range = compute_autovariance_2d(data, lag=1, axis='range')
            
    norm_data = np.zeros_like(data)
    z_scores = np.full_like(data, np.nan) 
    for r in range(ranges):  # Loop over range gates
        valid_mask = ~np.isnan(data[:, r])  # Ignore NaNs
        if np.sum(valid_mask) > 0:  # Ensure at least some valid data
            norm_data[:, r] = data[:, r] / autovariance_range[r]
            z_scores[valid_mask, r] = zscore(norm_data[valid_mask, r])  
    
    mask_2d = np.abs(z_scores) > threshold 
    idx = np.where((mask_2d.sum(axis=1)) > int(fraction * ranges))[0]
    
    return idx, mask_2d


def detect_outliers_rolling_autovariance(data, window=3, lag=1, threshold=2.5, axis='time'):
    """
    Detect outliers based on rolling mean and autovariance.
    
    Parameters:
    data (array-like): The input time series.
    window (int): Rolling mean window size (should be odd).
    lag (int): The lag at which to compute autovariance.
    threshold (float): Z-score threshold for outlier detection.
    
    Returns:
    list: Indices of detected outliers.
    """
    assert window % 2 == 1, "Window size must be odd"
    
    # Compute rolling mean using provided function
    rolling_avg = rolling_mean(data, window, axis)
    
    # Compute deviation from rolling mean
    detrended_data = data[:len(rolling_avg)] - rolling_avg
    
    # Compute autovariance for each rolling window
    autovars = np.array([compute_autovariance(detrended_data[i:i+window], lag) for i in range(len(detrended_data) - window + 1)])
    # autovars = np.array([compute_autovariance(data[i:i+window], lag) for i in range(len(data) - window + 1)])
    
    # Compute Z-score of autovariances
    z_scores = zscore(autovars)
    
    # Find outliers based on threshold
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    centered_outliers = outlier_indices + window // 2
    
    centered_outliers = centered_outliers[(centered_outliers >= window // 2) & (centered_outliers < len(data))]
    
    return centered_outliers


def create_mask(radar_object, SNRhname='SNRh', SNRvname='SNRv', threshold=2.5):
    """
    Create radar variable 'Mask' that masks time steps identified as outliers
    using autovariance of SNRh & SNRVh.
    
    Parameters:
    radar_object: pyart radar object with axes 'range' and 'time'
    SNRhname: name of variable containing horizontal signal-to-noise ratio (str)
    SNRvname: name of variable containing vertical signal-to-noise ratio (str)
    
    Returns:
    mask_2d: 2D-data array of shape (radar_object.time, radar_object.range) 
    containing mask (boolean)
    """
    SNRh = radar_object.fields[SNRhname]['data']
    SNRv = radar_object.fields[SNRvname]['data']
    
    # drop_h = detect_outliers_autovariance(SNRh)
    # drop_v = detect_outliers_autovariance(SNRv)
    drop_h, mask_h = detect_outliers_autovariance_2d(SNRh, threshold=threshold)
    drop_v, mask_v = detect_outliers_autovariance_2d(SNRv, threshold=threshold)
    drop_noise = np.unique(np.concatenate((drop_h, drop_v)))
    
    mask_2d = np.ones(SNRh.shape, dtype=bool)
    mask_2d[drop_noise, :] = False # drop timesteps with > 40% noise 
    # mask_2d &= ~mask_h & ~mask_v
    
    return drop_noise, ~mask_2d


def mask_remove(radar_object, maskname):
    """ 
    Removes data from variables if mask == True if variable is of the same 
    dimensions as the mask.
    
    radar_object: pyart radar object
    maskname: name of boolean mask-variable (str)
    
    returns:
    radar_object with modified fields (removing data in places where mask == True)
    """
    mask = radar_object.fields[maskname]['data']
    for field_name in radar_object.fields:
        if field_name == maskname:
            continue 
        
        field_data = radar_object.fields[field_name]['data']
        
        if isinstance(field_data, list) or mask.shape != field_data.shape:
            continue
        
        radar_object.fields[field_name]['data'] = np.ma.masked_where(mask, field_data)
        
    return radar_object


def mask_addvars(radar_object, maskname):
    """
    Adds masked variables to the radar object if variable is of the same 
    dimensions as the mask.
    
    radar_object: pyart radar object
    maskname: name of boolean mask-variable (str)
    
    returns:
    radar_object with masked variables added fields containing original (non-masked) data
    """
    fields = list(radar_object.fields)
    fields.remove(maskname)

    mask = radar_object.fields[maskname]['data'].astype(bool)

    for field_name in fields:
        field_data = radar_object.fields[field_name]['data']

        if isinstance(field_data, list):
            continue

        # only process fields that match first two dims
        if field_data.ndim < 2:
            continue

        if field_data.shape[:2] != mask.shape:
            continue

        field_full = field_name + '_full'

        if field_full not in radar_object.fields:
            full_dic = {
                **radar_object.fields[field_name],
                'data': np.ma.array(
                    field_data.data.copy(),
                    mask=np.ma.getmaskarray(field_data).copy())
            }
            # 'valid_min'/'valid_max' on the source field reflect the
            # *thresholded* range (e.g. post-censoring SNR >= 0). The
            # '_full' field is explicitly the unthresholded data, which can
            # legitimately fall outside that range (e.g. raw SNR going
            # negative below the noise floor). Leaving the inherited bounds
            # in place causes CF-aware readers (netCDF4-python, xarray,
            # Py-ART) to silently mask out most or all of the '_full' field
            # on read, defeating the point of keeping it unrestricted.
            full_dic.pop('valid_min', None)
            full_dic.pop('valid_max', None)
            radar_object.fields[field_full] = full_dic

        original_mask = np.ma.getmaskarray(field_data)

        if field_data.ndim == 2:
            expanded_mask = mask
        elif field_data.ndim == 3:
            expanded_mask = np.broadcast_to(
                mask[:, :, np.newaxis],
                field_data.shape
            )
        else:
            continue

        combined_mask = np.logical_or(original_mask, expanded_mask)

        radar_object.fields[field_name]['data'] = np.ma.array(
            field_data.data,
            mask=combined_mask
        )

        radar_object.fields[field_name]['long_name'] = \
            f"{radar_object.fields[field_name].get('long_name', field_name)} (masked)"

    return radar_object


def file_lists(date, ql_dir, raw_dir):
    """
    Returns lists of files per scan type (sector scan, RHI, Zdr/birdbath-PPI) 
    and writes these list into the raw data folder (if file already exists 
    checks whether it's up to date).
    """
    year, month, day = date.year, date.month, date.day
    files = sorted(glob.glob(f"{ql_dir}/*.png"))
    dat_files = sorted(glob.glob(f"{raw_dir}/*.dat"))
    
    rhis = []
    sectors = []
    zdrs = []
    
    for f in files:
        if pattern_rhi.search(f):
            match = pattern.search(f)
            if match:
                fn = match.group(1)
                rhis.append(f"{raw_dir}/{fn}.dat")
        elif pattern_sector.search(f):
            match = pattern.search(f)
            if match:
                fn = match.group(1)
                sectors.append(f"{raw_dir}/{fn}.dat")
    
    zdrs = [fn for fn in dat_files if fn not in sectors and fn not in rhis]
    
    add = {'zdr': zdrs, 'sector': sectors, 'rhi': rhis}
    for a, file_list in add.items():
        fpath = os.path.join(raw_dir, f"{year}{month:02}{day:02}_{a}_raw_all.txt")
        file_list_set = set(file_list)
        
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                existing_files = set(line.strip() for line in f.readlines())
        else:
            existing_files = set()
            
        final_list = sorted(existing_files | file_list_set)
        
        if final_list == sorted(existing_files):
            print(f"File already exists and is up to date: {fpath}")
        else:
            with open(fpath, 'w') as f:
                for filename in final_list:
                    f.write(filename + '\n')
            print(f"Created/updated file {fpath}")
    
    return rhis, sectors, zdrs
    

def plotting(radar, scantype, outpath):
    display = pyart.graph.RadarDisplay(radar)

    #gf = apply_filter(radar_object, limits_clutter)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    if scantype == 'Zdr':
        for ax, (key, value) in zip(axs.flatten(), plotvars.items()):
            display.plot_vpt(key, 
                             vmin=plotvals[key][0], 
                             vmax=plotvals[key][1], 
                             cmap=plotvals[key][2],
                             ax=ax)
        plt.suptitle('Unfiltered for noise')
    elif scantype == 'RHI':
        for ax, (key, value) in zip(axs.flatten(), plotvars.items()):
            ttl = f"{round(radar.metadata['Azimuth_value'], 1)} deg {pd.to_datetime(radar.time['data'].min(), unit='s')}\n{value}"
            display.plot_rhi(key, 
                             vmin=plotvals[key][0],
                             vmax=plotvals[key][1], 
                             cmap=plotvals[key][2], 
                             ax=ax, title=ttl)
            ax.set_xlim(pltConfig_MXPol['xlim_RHI'])
            ax.set_ylim(pltConfig_MXPol['ylim_RHI'])
        plt.suptitle('Unfiltered for noise')
    else:
        for ax, (key, value) in zip(axs.flatten(), plotvars.items()):
            display.plot_ppi(key, #gatefilter=gf,
                             vmin=plotvals[key][0], 
                             vmax=plotvals[key][1], 
                             cmap=plotvals[key][2],
                             ax=ax)
            ax.set_xlim(pltConfig_MXPol['xlim_sector'])
            ax.set_ylim(pltConfig_MXPol['ylim_sector'])
        #plt.suptitle('SNR > 5.0')
    plt.tight_layout()
    print(f"Saving {outpath.split('/')[-1]}")
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='w')
    plt.close()


def process_save(file_list, out_dir, remove_or_add, plot=False, maskname='Mask'):
    """
    Function to process files and save L0 .nc files 
    - azimuth- & elevation-correction
    - dropped data removal

    Parameters:
    file_list: list of all .dat files in directory
    out_dir: location to save .nc files
    remove_or_add: 'remove' for removing data, 'add' (str)
        
    Returns:
    
    """
    assert remove_or_add in {'add', 'remove'}, f"{remove_or_add} should be 'remove' or 'add'."

    excluded_files = []
    os.makedirs(out_dir, exist_ok=True)
    
    def log_exclusion(msg):
        excluded_files.append(msg)
        
    for f in file_list:
        try:
            name = f.split('/')[-1][:-4]
            processed_fn = os.path.join(out_dir, f'{name}*.nc')
            if glob.glob(processed_fn):
                print(f".nc file already exists for {name}, skipping.", flush=True)
                log_exclusion(f"{name}: already exists.")
                continue
                
            print(f"processing {name}...", flush=True)
            header, records = read_raw_data(f, config)
            if 'PPI' in records:
                scantype = 'Zdr'
                radar = process_dataset('PPI', header, records['PPI'], config)
                radar.azimuth['data'] = azimuth_correction(radar.azimuth['data'], corr_az)
                if not radar.metadata['AzOffset_description'] == 'Az. Offset already substracted':
                    radar.metadata['AzOffset_description'] = 'Az. Offset already substracted'
                radar.metadata['Elevation_value'] = elevation_correction(radar.metadata['Elevation_value'], corr_el)
            elif 'RHI' in records:
                scantype = 'RHI'
                radar = process_dataset('RHI', header, records['RHI'], config)
                radar.azimuth['data'] = azimuth_correction(radar.azimuth['data'], corr_az)
                if not radar.metadata['AzOffset_description'] == 'Az. Offset already substracted':
                    radar.metadata['AzOffset_description'] = 'Az. Offset already substracted'
            elif 'PROFILE' in records:
                scantype = 'PROFILE'
                radar = process_dataset('PROFILE', header, records['PROFILE'], config)
            else:
                scantype = 'sector'
                radar = process_dataset('SECTOR_SCAN', header, records['Point'], config)

            print(f"opened {name} ({scantype}), continuing processing", flush=True)
            
            # check how long scan took
            scantime = radar.time['data'].max() - radar.time['data'].min()
            if scantime <= int(0.5*scantimes[scantype]):
                error_msg = f"{name} ({scantype}): too much data missing, only {int(scantime)} of {scantimes[scantype]} sec. recorded"
                print(error_msg, flush=True)
                log_exclusion(error_msg)
                continue

            radar.elevation['data'] = elevation_correction(radar.elevation['data'], corr_el)
            radar.metadata['Elevation_correction'] = 'Yes'
            radar.metadata['AzOffset_value'] = corr_az
                
            (ids, mask) = create_mask(radar)
            snr_mask = create_snr_mask(radar, limits_SNR)
            mask = mask | snr_mask

            N = len(radar.time['data'])
            if len(ids) >= int(0.5*N):
                error_msg = f"{name} ({scantype}): too much data missing"
                print(error_msg, flush=True)
                log_exclusion(error_msg)
                continue
        
            radar.fields[maskname] = {
                'data': mask, 
                'units': 'unitless', 
                'long_name': 'Masked time steps identified as outliers - to filter out dropped data blocks', 
                'standard_name':'masked_timesteps'
                }
            
            radar = mask_remove(radar, maskname) if remove_or_add == 'remove' else mask_addvars(radar, maskname)
            radar.fields[maskname]['data'] = radar.fields[maskname]['data'].astype(np.int8)
            
            config_scantype_map = {
                'Zdr': 'PPI',
                'RHI': 'RHI',
                'PROFILE': 'PROFILE',
                'sector': 'SECTOR_SCAN'
            }
            cfg_key = config_scantype_map.get(scantype, 'RHI')
            try:
                safe_fill = config['products']['datasets'][cfg_key]['products']['NETCDF_POLAR']['nanval']
                safe_fill = np.float32(safe_fill) # Match the 32-bit array precision
            except (KeyError, TypeError):
                safe_fill = np.float32(-9999.0)

            for field in list(radar.fields.keys()):
                field_data = radar.fields[field]['data']
                if isinstance(field_data, list):
                    # print(field)
                    field_data = np.array(field_data)
                    radar.fields[field]['data'] = field_data
                
                if np.issubdtype(field_data.dtype, np.float64):
                    radar.fields[field]['data'] = field_data.astype(np.float32)
                elif np.issubdtype(field_data.dtype, np.int64):
                    radar.fields[field]['data'] = field_data.astype(np.int32)
                
                radar.fields[field]['_FillValue'] = safe_fill
                radar.fields[field]['missing_value'] = safe_fill
                
                if isinstance(radar.fields[field]['data'], np.ma.MaskedArray):
                    radar.fields[field]['data'].fill_value = safe_fill

            outname = pattern.search(f).group(1)
            outpath = os.path.join(out_dir, f"{outname}_{scantype}.nc")

            # fix to adhere to new required PyArt metadata
            if scantype.upper() == 'RHI':
                radar.scan_type = 'rhi'
                radar.metadata['primary_axis'] = 'axis_z' 
            elif scantype.upper() in ['ZDR', 'PPI']:
                radar.scan_type = 'ppi'
                radar.metadata['primary_axis'] = 'axis_z'

            if 'volume_number' not in radar.metadata:
                radar.metadata['volume_number'] = 1
            if 'platform_type' not in radar.metadata:
                radar.metadata['platform_type'] = 'fixed'
            if 'instrument_type' not in radar.metadata:
                radar.metadata['instrument_type'] = 'radar'

            try:
                write_cfradial(outpath, radar)
                print(f"Created output file {outpath}", flush=True)
                if plot:
                    plot_dir = os.path.join(os.path.dirname(out_dir), "plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    outpath_plot = os.path.join(plot_dir, f"{outname}_{scantype}.png")
                    plotting(radar, scantype, outpath_plot)
                    print(f"Saved plot {os.path.basename(outpath_plot)}", flush=True)
            except FileExistsError:
                print(f"Collision: {outname} was already finished by another worker. Skipping.", flush=True)
                continue
            except Exception as e:
                error_msg = f"Error writing file {outname}_{scantype}.nc: {str(e)}"
                print(error_msg, flush=True)
                excluded_files.append(error_msg)
            finally:
                if 'radar' in locals():
                    radar.fields.clear()
                    del radar
                if 'header' in locals():
                    del header
                if 'records' in locals():
                    del records
                gc.collect()
            
        except Exception as e:
            error_msg = f"{f}: error {str(e)}"
            print(error_msg, flush=True)
            excluded_files.append(error_msg)
        
    return excluded_files


def format_elapsed(seconds):
    c = pd.Timedelta(seconds=seconds).components
    if c.days:
        return f"{c.days}d {c.hours}h {c.minutes}m {c.seconds}s"
    elif c.hours:
        return f"{c.hours}h {c.minutes}m {c.seconds}s"
    elif c.minutes:
        return f"{c.minutes}m {c.seconds}s"
    else:
        return f"{seconds:.2f}s"


#%%

dates = generate_date_range("2024-11-26", "2025-01-24")
# dates = generate_date_range("2024-11-25", "2024-11-28") # for testing
# dates = generate_date_range('2024-11-29', '2024-12-07')
# dates = dates + [datetime(2024, 12, 9), datetime(2024, 12, 10), datetime(2024, 12, 14), datetime(2024, 12, 20), datetime(2024, 12, 22)]
# dates = dates + [datetime(2024, 12, 24), datetime(2024, 12, 25), datetime(2025, 1, 7), datetime(2025, 1, 14), datetime(2025, 1, 15), datetime(2025, 1, 21)]

dates = [datetime(2024, 11, 28)]
dates_all = generate_date_range('2024-11-26', '2025-01-24')
dates_nr = [d for d in dates_all if d not in dates]

base_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MXPol'

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    for date in dates_all:
        date_str = date.strftime('%Y-%m-%d')
        year, month, day = date.year, date.month, date.day
        raw_dir = f"{base_dir}/raw_data/{year}/{month:02}/{day:02}"
        files = sorted(glob.glob(f"{raw_dir}/*.dat"))

        if not files:
            continue
        
        outdir = os.path.join(base_dir, f"proc_data/L0_new/{year}/{month:02}/{day:02}")
        os.makedirs(outdir, exist_ok=True)

        print(f"processing {len(files)} files for {date_str}")
        t0 = time.time()

        excluded_files = []
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(process_save, [f], outdir, 'add', True): f
                for f in files
            }
            for future in as_completed(futures):
                f = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error while processing {os.path.basename(f)}: {e}")
                    continue
                if result:
                    excluded_files.extend(r for r in result if r is not None)

        if excluded_files:
            log_path = os.path.join(base_dir, 'proc_data/L0_new/excluded_files.txt')
            with open(log_path, 'a') as f_log:
                for line in excluded_files:
                    f_log.write(line + '\n')
        
        gc.collect()
        print(f"Finished processing files for {year}-{month:02}-{day:02} at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed time: {format_elapsed(time.time() - t0)}")

# old main block (pre-parallellisation)
# if __name__ == '__main__':
#     for date in dates:
#         year, month, day = date.year, date.month, date.day
#         # ql_dir = os.path.join(base_dir, f"quicklooks/{year}/{month:02}/{day:02}")
#         raw_dir = f"{base_dir}/raw_data/{year}/{month:02}/{day:02}"

#         # rhi_list, sector_list, zdr_list = file_lists(date, ql_dir, raw_dir)

#         files = sorted(glob.glob(f"{raw_dir}/*.dat"))
#         outdir = os.path.join(base_dir, f"proc_data/L0/{year}/{month:02}/{day:02}")
#         os.makedirs(outdir, exist_ok=True)

#         excluded_files = process_save(date, files, outdir, f"{base_dir}/proc_data/L0", 'add')
#         # print(excluded_files)
# %%
