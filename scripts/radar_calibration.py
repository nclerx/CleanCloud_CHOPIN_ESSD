#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:18:07 2025

Script for calibration of MXPol (X-band / 9.4 GHz), MIRA (Ka-band / 35 GHz) and
BASTA (W-band / 95 GHz) radars during the CleanCloud CHOPIN campaign (October
2024 - January 2025) following Jorquera et al. 2023 (doi:10.1175/JTECH-D-22-0087.1)
and Dias Neto et al. 2019 (doi:10.5194/essd-11-845-2019)

@author: clerx
@contributor: mweiss
"""
#%% imports
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from datetime import datetime

from src.constants_input import base_dir, att_fn, nl, frequencies, all_variables_MXPol, variables_MXPol
from src.radar_processing import safe_openmfdataset, correct_gas_attenuation, get_clean_mask, calib_stats
from src.plotting import plot_calibration, plot_calibration_timeline, plot_calibration_quicklook


frequency = frequencies['BASTA']

#%% user input
date = datetime(2025, 1, 15)

# how to determine time interval for analysis ('auto' or 'manual')
t_min_max = 'auto' 
# 'manual' uses "hour_start" and "hour_end" to determine time interval
# 'auto' determines time interval based on minimum/maximum time in used data sets
# if 't_start' and 't_end' are specified, these are used for more precise determination of time interval
if t_min_max == 'manual':
    hour_start = 17
    hour_end = 19
    t_start = t_end = None
    # t_start = '18:00:00'
    # t_end = '19:30:00'

# which radars to use
use_MXPol = True
use_MIRA = True
use_BASTA = True
if use_BASTA:
    BASTA_mode = '25m'
    # BASTA_mode = '12m5'
write_doc = True
save_plots = True

# set minimum and maximum range for data (estimated from quicklooks) in meters
r_min = 500
r_max = 10000

# remove over-sensitive MIRA-data to improve reflectivity range selection below a certain lower limit of
# shape y = -x + lower_Ze_limit - keeps spread of data points around linear correlation above this lower threshold
limit_sensitivity = True
lower_Ze_limit = -10   # to eliminate values between (x, y) with x = y use limit = 2*x (e.g. -60 for removing everything below [-30, -30])

# remove all values below a lower limit of shape y = x + intercept 
limit_gradient = False
intercept = -10

t_int = 5 # time sampling interval in seconds
t_nthres = 5 # if original time resolution is <= t_nthres * t_int, use 'nearest', otherwise interpolate data


# set in- and output directories
output_dir = os.path.join(base_dir, 'radar_calib')

assert r_max > r_min
r_res = 25 # range gate resolution

limits = (limit_sensitivity, lower_Ze_limit, limit_gradient, intercept)

#%% retrieving & manipulating data 

# delete any pre-existing variables that are defined in this cell
for var in [
    'year', 'month', 'day', 'dataset_times', 'time_res',
    'files_MXPol', 'MXPol_valid', 'MXPol_all', 'MXPol',
    'files_MIRA', 'MIRA_all', 'MIRA',
    'files_BASTA', 'BASTA_all', 'BASTA',
    'tmin', 'tmax', 'full_time', 'mask', 'mask_xr',
    'background_mask', 'background_mask_resampled',
    'common_range', 'datasets', 'active_datasets',
    'active_sizes', 'Zh_all', 'Zg_all', 'Zb_all'
]:
    if var in locals():
        del globals()[var]

year, month, day = date.year, date.month, date.day

dataset_times = []
time_res = {}
if use_MXPol:
    files_MXPol = sorted(glob.glob(os.path.join(base_dir, f"MXPol/proc_data/L0/{year}/{month:02}/{day:02}/*_Zdr.nc")))
    MXPol_valid, _ = safe_openmfdataset(files_MXPol)
    MXPol_all = xr.open_mfdataset(MXPol_valid, combine='nested', concat_dim='time', chunks={'time': 3600}, drop_variables=[var for var in all_variables_MXPol if var not in variables_MXPol])
    MXPol_all = MXPol_all.drop_dims(['nfft']) # remove unnecessary dimensions
    _, index = np.unique(MXPol_all['time'], return_index=True)
    MXPol_all = MXPol_all.isel(time=index) # remove duplicates in time axes
    dataset_times.append(MXPol_all['time'].values)
    dt_MXPol = np.median(np.diff(MXPol_all['time']))
    time_res['MXPol'] = dt_MXPol

if use_MIRA:
    files_MIRA = sorted(glob.glob(os.path.join(base_dir, f"MIRA/L1/{year}/{month:02}/{day:02}/*_merged.nc")))
    MIRA_all = xr.open_mfdataset(files_MIRA)
    MIRA_all = MIRA_all.drop_dims(['doppler']) # remove unnecessary dimensions
    dataset_times.append(MIRA_all['time'].values)
    dt_MIRA = np.median(np.diff(MIRA_all['time']))
    time_res['MIRA'] = dt_MIRA

if use_BASTA:
    assert BASTA_mode in ['12m5', '25m']
    if BASTA_mode == '12m5':
        files_BASTA = sorted(glob.glob(os.path.join(base_dir, f"BASTA/L1/12m5/BASTA_L1_12m5_{year}{month:02}{day:02}*.nc")))
    else:
        files_BASTA = sorted(glob.glob(os.path.join(base_dir, f"BASTA/L1/25m/BASTA_L1_25m_{year}{month:02}{day:02}*.nc")))
    BASTA_all = xr.open_mfdataset(files_BASTA)
    BASTA_all['reflectivity'] = BASTA_all['reflectivity'].where((BASTA_all['reflectivity'] >= -60) & (BASTA_all['reflectivity'] <= 100), np.nan)
    dataset_times.append(BASTA_all['time'].values)
    dt_BASTA = np.median(np.diff(BASTA_all['time']))
    time_res['BASTA'] = dt_BASTA

if not dataset_times:
    raise ValueError("No radar datasets selected")

if t_min_max == 'manual':
    assert hour_start is not None and hour_end is not None, 't_min and t_max must be defined when using "t_min_max = manual"'
    if t_start and t_end:
        tmin = np.datetime64(datetime(year, month, day) + pd.to_timedelta(t_start))
        tmax = np.datetime64(datetime(year, month, day) + pd.to_timedelta(t_end))
    else:
        tmin = np.datetime64(datetime(year, month, day, hour_start))
        tmax = np.datetime64(datetime(year, month, day, hour_end, 59))
else:
    tmin = max(np.min(times) for times in dataset_times)
    tmax = min(np.max(times) for times in dataset_times)

full_time = pd.date_range(tmin, tmax, freq=f"{t_int}s")

# if t_int <= 5x time resolution, use 'nearest'; otherwise use 'interpolate'
if use_MXPol:
    # check if full time interval is covered
    if (MXPol_all.time.min().values <= tmin + np.timedelta64(int(t_int * t_nthres), 's')) and (MXPol_all.time.max().values >= tmax - np.timedelta64(int(t_int * t_nthres), 's')):
        MXPol = MXPol_all.loc[dict(time=slice(tmin, tmax))]
        if t_int <= (time_res['MXPol'] * t_nthres).astype('timedelta64[s]'):
            MXPol = MXPol_all.reindex(time=full_time, method='nearest')
        else:
            # convert reflectivity values to linear for interpolation, convert back after resampling
            MXPol['Zh'].values = 10**((MXPol['Zh'].values)/10)
            MXPol = MXPol.resample(time=f"{t_int}s").mean(dim='time')
            MXPol['Zh'].values = 10*np.log10(MXPol['Zh'].values)
        mask = get_clean_mask(MXPol['Zh'])
        mask_xr = MXPol['Zh'].copy(data=mask)
        MXPol = MXPol.assign(clean_mask = mask_xr)
        del mask, mask_xr
    else:
        raise ValueError("Specified time interval not covered by MXPol data.")

if use_MIRA:
    # check if full time interval is covered
    if (MIRA_all.time.min().values <= tmin + np.timedelta64(int(t_int * t_nthres), 's')) and (MIRA_all.time.max().values >= tmax - np.timedelta64(int(t_int * t_nthres), 's')):
        MIRA = MIRA_all.loc[dict(time=slice(tmin, tmax))]
        # check if original time step is close enough to desired timestep
        if t_int <= (time_res['MIRA'] * t_nthres).astype('timedelta64[s]'):
            MIRA = MIRA_all.reindex(time=full_time, method='nearest')
        else:
            MIRA = MIRA.resample(time=f"{t_int}s").mean(dim='time')
        mask = get_clean_mask(MIRA['Z'])
        mask_xr = MIRA['Z'].copy(data=mask)
        MIRA = MIRA.assign(clean_mask = mask_xr)
        del mask, mask_xr
    else:
        raise ValueError("Specified time interval not covered by MIRA data.")

if use_BASTA:
    # check if full time interval is covered
    if (BASTA_all.time.min().values <= tmin + np.timedelta64(int(t_int * t_nthres), 's')) and (BASTA_all.time.max().values >= tmax - np.timedelta64(int(t_int * t_nthres), 's')):
        # BASTA = BASTA_all.loc[dict(time=slice(tmin, tmax))]
        BASTA = BASTA_all.sel(time=slice(tmin, tmax))
        # check if original time step is close enough to desired timestep
        if t_int <= (time_res['BASTA'] * t_nthres).astype('timedelta64[s]'):
            BASTA = BASTA_all.reindex(time=full_time, method='nearest')
        else:
            if 'background_mask' in BASTA:
                background_mask = BASTA['background_mask'].copy()
                BASTA = BASTA.drop_vars('background_mask')  # Remove from dataset
            
            # convert reflectivity values to linear for interpolation, convert back after resampling
            BASTA['reflectivity'].values = 10**((BASTA['reflectivity'].values)/10)
            BASTA = BASTA.resample(time=f"{t_int}s").mean(dim='time', keep_attrs=True)
            BASTA['reflectivity'].values = 10*np.log10(BASTA['reflectivity'].values)
            
            # add back the resampled background_mask
            if 'background_mask' in locals():
                background_mask_resampled = background_mask.resample(time=f"{t_int}s").nearest()
                BASTA = BASTA.assign(background_mask=background_mask_resampled)

        mask = get_clean_mask(BASTA['reflectivity'])
        if 'background_mask' in BASTA:
            mask = mask & BASTA['background_mask']
        mask_xr = BASTA['reflectivity'].copy(data=mask)
        BASTA = BASTA.assign(clean_mask = mask_xr)
        del mask, mask_xr
    else:
        raise ValueError("Specified time interval not covered by BASTA data.")

# resample range
common_range = np.arange(r_min, r_max, r_res)
if use_MXPol:
    MXPol = MXPol.reindex(range=common_range, method='nearest')
if use_MIRA:
    MIRA = MIRA.reindex(range=common_range, method='nearest')
if use_BASTA:
    BASTA = BASTA.reindex(range=common_range, method='nearest')

# make sure time and range order of data sets are the same
datasets = {}
if use_MXPol:
    datasets['MXPol'] = MXPol
if use_MIRA:
    datasets['MIRA'] = MIRA
if use_BASTA:
    datasets['BASTA'] = BASTA

active_datasets = {name: ds for name, ds, use in zip(datasets.keys(), datasets.values(), [use_MXPol, use_MIRA, use_BASTA]) if use}
active_sizes = [ds.sizes for ds in active_datasets.values()]

# check if all datasets have the same sizes
if not all(size == active_sizes[0] for size in active_sizes):
    print(f"Error in dataset dimensions:{nl}" + nl.join(f"{name}: {ds.sizes}" for name, ds in active_datasets.items()))

# flatten data for making histograms
if use_MXPol:
    Zh_all = MXPol['Zh'].values.flatten()
if use_MIRA:
    Zg_all = (10 * np.log10(MIRA['Zg'])).values.flatten()
if use_BASTA:
    BASTA['reflectivity'].values = correct_gas_attenuation(att_fn, BASTA, 'reflectivity', frequency, tmin, tmax)
    Zb_all = BASTA['reflectivity'].where(BASTA['background_mask']).values.flatten()

#%% plotting & storage of calibration information
if use_MXPol and use_MIRA:
    mask = mask_sensitivity = mask_gradient = None
    mask = np.ones_like(Zh_all, dtype=bool)
    if limit_sensitivity:
        mask_sensitivity = Zg_all >= (-1 * Zh_all + lower_Ze_limit)
        mask = mask & mask_sensitivity
    if limit_gradient:
        mask_gradient = Zg_all >= (Zh_all + intercept)
        mask = mask & mask_gradient
    Zh_used = Zh_all[mask]
    Zg_used = Zg_all[mask]
    # Z_limits = plot_calibration(Zh_used, Zg_used, ('MXPol', 'MIRA'), (r_min, r_max), date, (tmin, tmax, t_int), output_dir, limits, savefig=save_plots, writedoc=write_doc)
    Z_limits, fit_data = plot_calibration(Zh_used, Zg_used, ('MXPol', 'MIRA'), (r_min, r_max), date, (tmin, tmax, t_int), output_dir, limits, savefig=False, writedoc=False)

    names = ('MXPol', 'Zh', 'MIRA', 'Z')
    # plot_calibration_quicklook(MXPol, MIRA, output_dir, names, (r_min, r_max), (tmin, tmax), Z_limits, limits, save_plots)
    plot_calibration_quicklook(MXPol, MIRA, output_dir, names, (r_min, r_max), (tmin, tmax), Z_limits, limits)

# if use_MXPol and use_BASTA:
#     mask = mask_sensitivity = mask_gradient = None
#     mask = np.ones_like(Zh_all, dtype=bool)
#     if limit_sensitivity:
#         mask_sensitivity = Zb_all >= (-1 * Zh_all + lower_Ze_limit)
#         mask = mask & mask_sensitivity
#     if limit_gradient:
#         mask_gradient = Zb_all >= (Zh_all + intercept)
#         mask = mask & mask_gradient
#     Zh_used = Zh_all[mask]
#     Zb_used = Zb_all[mask]
#     Z_limits = plot_calibration(Zh_used, Zb_used, ('BASTA', 'MXPol', BASTA_mode), (r_min, r_max), date, (tmin, tmax, t_int), output_dir, limits, savefig=save_plots, writedoc=write_doc)
    
#     names = ('BASTA', 'reflectivity', BASTA_mode, 'MXPol', 'Zh')
#     plot_calibration_quicklook(BASTA.where(BASTA['background_mask']), MXPol, output_dir, names, (r_min, r_max), (tmin, tmax), Z_limits, limits, save_plots)

if use_MIRA and use_BASTA:
    mask = mask_sensitivity = mask_gradient = None
    mask = np.ones_like(Zg_all, dtype=bool)
    if limit_sensitivity:
        mask_sensitivity = Zg_all >= (-1 * Zb_all + lower_Ze_limit)
        mask = mask & mask_sensitivity
    if limit_gradient:
        mask_gradient = Zg_all >= (Zb_all + intercept)
        mask = mask & mask_gradient
    Zb_used = Zb_all[mask]
    Zg_used = Zg_all[mask]
    # Z_limits = plot_calibration(Zb_used, Zg_used, ('BASTA', 'MIRA', BASTA_mode), (r_min, r_max), date, (tmin, tmax, t_int), output_dir, limits, savefig=save_plots, writedoc=write_doc)
    Z_limits = plot_calibration(Zb_used, Zg_used, ('BASTA', 'MIRA', BASTA_mode), (r_min, r_max), date, (tmin, tmax, t_int), output_dir, limits, savefig=False, writedoc=False)

    names = ('BASTA', 'reflectivity', BASTA_mode, 'MIRA', 'Z')
    # plot_calibration_quicklook(BASTA, MIRA, output_dir, names, (r_min, r_max), (tmin, tmax), Z_limits, limits, save_plots)
    plot_calibration_quicklook(BASTA, MIRA, output_dir, names, (r_min, r_max), (tmin, tmax), Z_limits, limits)

 # %% read output file & plot calibration results
fn_MIRABASTA = os.path.join(base_dir, 'radar_calib/calibrationdata_BASTA25m_MIRA.txt')
calib_MIRABASTA = pd.read_csv(fn_MIRABASTA, sep='\t')

cal_vals, cal_vals_excl, stats = calib_stats(calib_MIRABASTA)

fig, ax = plot_calibration_timeline(calib_MIRABASTA, ('MIRA', 'BASTA'))
plotname = os.path.join(os.path.dirname(fn_MIRABASTA), "calibrationdata_BASTA25m_MIRA.png")
plt.savefig(plotname, dpi=300, bbox_inches='tight', facecolor='w')
plt.show()

fn_MXPolMIRA = os.path.join(base_dir, 'radar_calib/calibrationdata_MXPol_MIRA.txt')
calib_MXPolMIRA = pd.read_csv(fn_MXPolMIRA, sep='\t')
fig, ax = plot_calibration_timeline(calib_MXPolMIRA, ('MIRA', 'MXPol'))
plotname = os.path.join(os.path.dirname(fn_MXPolMIRA), "calibrationdata_MXPol_MIRA.png")
plt.savefig(plotname, dpi=300, bbox_inches='tight', facecolor='w')

