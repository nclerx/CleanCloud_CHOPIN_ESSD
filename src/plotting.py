#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:58:02 2025

@author: clerx
"""
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import os
import re
import time
import pyart
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import seaborn as sns
import gc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from pathlib import Path
from collections import defaultdict
from datetime import datetime
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection, PolyCollection

from src.constants_input import temps, nl, headers, radar_altitude, temps, pltConfig, calibrationvalues, variables_to_plot, variables_type_map, radiosonde_dates, MXPol_lost_frequency_times, variables_peaktree, variables_to_plot_pyart, default_variable_maps
from src.utils import fit_slope, remove_nans, create_file_with_headers, write_calibrationfile, format_list, flatten_for_densityplot, determine_nbins, determine_nhexbins, mask_to_intervals, mask_to_intervals_numpy, calculate_edges, merge_close_intervals, intervals_to_mask
from src.radar_processing import prepare_MIRAdata, prepare_MXPoldata, range_selection, determine_range_limits, density_filter, calib_stats, find_nan_intervals, znc_to_pyart, noise_threshold

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def log(self, msg):
        dt = time.perf_counter() - self.t0
        print(f"[{dt:7.2f} s] {msg}")
        self.t0 = time.perf_counter()

radiosonde_datetimes = [pd.to_datetime(d, format='%Y%m%d%H') for d in radiosonde_dates]

plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
})

grouped = defaultdict(list)
type_by_var = {var: key for key, vars_ in variables_type_map.items() for var in vars_}
type_by_var['RMSg'] = 'other'


def meters_to_km(y, _):
    return f"{y / 1000:.1f} km"


def meters_to_km_num(y, _):
    return f"{y / 1000:.0f}"


def add_isotherm_labels(ax, ERA_data, altitudes, alt_range=None, temps=None, temp_threshold=2, xlims=None, fallback_place=False, debug=False):
    """helper function to add labels to plotted isotherms"""
    # temp_threshold = 2 # max. offset°C
    if temps is None:
        return 
    
    altitudes = np.asarray(altitudes)
    alt_min, alt_max = ax.get_ylim() if alt_range is None else alt_range

    bbox_style = dict(boxstyle='round, pad=0.2', alpha=0.75,
                      facecolor='white', edgecolor='white')
    
    if xlims is not None:
        try:
            xnums  = mdates.date2num([pd.to_datetime(x) for x in xlims])
            ax.set_xlim(xlims)
            xlims = xnums
        except Exception:
            xlims = ax.get_xlim()
    else:
        xlims = ax.get_xlim()

    time_start, time_end = [pd.to_datetime(mdates.num2date(x)) for x in xlims]
    time_offset = 0.02 * (time_end - time_start) 
    label_times = [time_start + time_offset, time_end - time_offset]

    for label_time in label_times:
        for temp in temps:
            # ensure label_time is tz-naive to match ERA_data['time']
            if getattr(label_time, "tzinfo", None) is not None:
                label_time = label_time.tz_localize(None)

            try:
                temp_at_label_time = ERA_data['t'].interp(time=label_time, method='linear')
            except Exception as e:
                print(f"interp failed for {label_time} (temp {temp}): {e}")
                continue

            temp_profile = temp_at_label_time.values
            if temp_profile.ndim != 1 or temp_profile.shape[0] != altitudes.shape[0]:
                if debug:
                    print("Profile/altitudes shape mismatch:", temp_profile.shape, altitudes.shape)
                continue

            temp_diff = np.abs(temp_profile - temp)

            valid_alt_mask = (altitudes >= alt_min) & (altitudes <= alt_max)
            
            if not np.any(valid_alt_mask):
                if debug:
                    print("No altitudes in range", alt_range)
                    continue
            
            valid_indices = np.where(valid_alt_mask)[0]
            valid_temp_diff = temp_diff[valid_indices]
            best_idx_rel = np.argmin(valid_temp_diff)
            best_idx = valid_indices[best_idx_rel]
            best_diff = valid_temp_diff[best_idx_rel]
            label_alt = float(altitudes[best_idx])

            # check if label_alt is within current y-axis range
            if not (alt_min <= label_alt <= alt_max):
                if debug:
                    print(f"Skipping label for {temp} at {label_alt} (outside y-axis range {alt_min}-{alt_max})")
                continue

            if debug:
                print(f"Label time {label_time} temp {temp}: best_diff={best_diff:.2f} at alt={label_alt}")

            if best_diff <= temp_threshold: # good elevation & time match
                ax.text(label_time, label_alt, f"{temp}°C",
                        fontsize=9, ha='center', va='center',
                        bbox=bbox_style, zorder=30, color='black', transform=ax.transData)
            else:
                if fallback_place:
                    # place a faint label (so you know an isotherm exists but not exact)
                    bbox_faint = bbox_style.copy()
                    bbox_faint.update(dict(alpha=0.5, facecolor='white'))
                    ax.text(label_time, label_alt, f"{temp}°C",
                            fontsize=8, ha='center', va='center',
                            bbox=bbox_faint, zorder=25, color='gray', transform=ax.transData)
                    if debug:
                        print(f"Placed fallback label for {temp} at {label_alt} (diff {best_diff:.2f})")
                else:
                    if debug:
                        print(f"Skipped label for {temp} (diff {best_diff:.2f} > {temp_threshold})")


def annotate_temperatures(ax, temp_altitudes, temp_time, alt_range=(0, 10)):
    """helper function to annotate isotherms"""
    bbox_style = dict(boxstyle='round, pad=0.2', alpha=0.5,
                      facecolor='white', edgecolor='white')
    alt_min, alt_max = alt_range 

    for temp, altitude in temp_altitudes.items():
        alt_value = float(altitude) / 1e3
        if (alt_value >= alt_min) & (alt_value <= alt_max):
            ax.text(temp_time, alt_value, f"{temp}°C", fontsize=9, 
                    ha='center', va='center', bbox=bbox_style, zorder=10)
            

def plot_spectrogram_MIRA(flte, output_dir, spq=3, plot_Z=True, plot_LDR=True, savefig=True, overwrite=False):
    """
    plot MIRA spectrograms (reflectivity and/or LDR) from a given file
    flte: filename (string)
    output_dir: directory to store plots
    spq: one spectrogram every "spq" minutes (int) - resampled MIRA data at 5 second-resolution
    plot_Z: make reflectivity spectrograms
    plot_LDR: make LDR spectrograms
    savefig/overwrite: save/overwrite existing figures
    """
    ds = prepare_MIRAdata(flte)
    x = -1 * ds['doppler'].values
    y = ds.range.values

    if plot_Z:
        sZco_data = ds['sZco'].values
    if plot_LDR:
        sLDR_data = ds['sLDR'].values

    x_min, x_max = -5, 3
    y_min, y_max = 0, 10e3
    
    sp_int = spq * 12 # time interval for spectrogram plots (5 second-data --> 1 plot per minute)
    len_time = len(ds.time)

    valid_timestamp_mask = np.any(np.isfinite(ds['sZco'].values), axis=(1, 2))
    valid_indices = np.where(valid_timestamp_mask)[0]

    if len(valid_indices) == 0:
        print(f"No valid timestamps found in {os.path.basename(flte)}")
    
    else:
        for i in range(0, len_time, sp_int):
            closest_valid = valid_indices[np.argmin(np.abs(valid_indices - i))]
            if i > 0:
                prev_closest = valid_indices[np.argmin(np.abs(valid_indices - (i - sp_int)))]
                if closest_valid == prev_closest:
                    continue

            if plot_Z:
                outname_sZ = f"{output_dir}/{pd.to_datetime(ds.time.values[closest_valid], unit='s').strftime('%Y%m%d-%H%M%S')}_sZco.png"
                if not overwrite and os.path.exists(outname_sZ):
                    print(f"{outname_sZ} already exists, skipping")
                    continue
    
                z = sZco_data[closest_valid, :, :]
    
                # fig, ax = plt.subplots(figsize=(4, 10), tight_layout=True)
                fig, ax = plt.subplots(figsize=(6, 7), tight_layout=True)
                cax = plt.pcolormesh(x, y, z, cmap=pltConfig['sZco']['cmap'], vmin=pltConfig['sZco']['vmin'], vmax=pltConfig['sZco']['vmax'])
                            
                colorbar = fig.colorbar(cax, ax=ax, label=pltConfig['sZco']['label'], shrink=0.8, aspect=40, pad=0.05)
                ax.set_xlim([x_min, x_max])
                x_major_ticks = np.arange(-10, 6, 5)
                x_minor_ticks = np.arange(-10, 6, 1)
                ax.set_xticks(x_major_ticks, minor=False)
                ax.set_xticks(x_minor_ticks, minor=True)

                ax.set_ylim([y_min, y_max])                
                y_major_ticks = np.arange(0, y_max + 1000, 1000)
                y_minor_ticks = np.arange(0, y_max + 200, 200)
                ax.set_yticks(y_major_ticks, minor=False)
                ax.set_yticks(y_minor_ticks, minor=True)

                ax.grid(True, which='major', alpha=0.7, linestyle='-', linewidth=0.8)
                ax.grid(True, which='minor', alpha=0.6, linestyle='-', linewidth=0.5) 
                ax.axvline(x=0, linewidth=1.2, alpha=0.7, linestyle='--')
                ax.axvline(x=-10, linewidth=0.8, alpha=0.7, linestyle='-')
                ax.axvline(x=5, linewidth=0.8, alpha=0.7, linestyle='-')
                ax.axhline(y=0, linewidth=0.8, alpha=0.7, linestyle='-')
                ax.axhline(y=10000, linewidth=0.8, alpha=0.7, linestyle='-')
                
                ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
                ax.set_xlabel('Doppler velocity [m/s]')
                ax.set_ylabel('Height above radar')
                plt.suptitle(f"Spectral Ze for {pd.to_datetime(ds.time.values[closest_valid], unit='s')}", fontweight='bold')
                if savefig:
                    plt.savefig(outname_sZ, dpi=300, bbox_inches='tight', facecolor='w')
                    print(f"  Saved {os.path.basename(outname_sZ)}")
                else:
                    plt.show()
                plt.close(fig)
                del z
            
            if plot_LDR:
                outname_sLDR = f"{output_dir}/{pd.to_datetime(ds.time.values[closest_valid], unit='s').strftime('%Y%m%d-%H%M%S')}_sLDR.png"
                if not overwrite and os.path.exists(outname_sLDR):
                    print(f"{os.path.basename(outname_sLDR)} already exists, skipping")
                    continue

                ldr = sLDR_data[closest_valid, :, :]

                fig, ax = plt.subplots(figsize=(4, 10), tight_layout=True)
                cax = plt.pcolormesh(x, y, ldr, cmap=pltConfig['sLDR']['cmap'], vmin=pltConfig['sLDR']['vmin'], vmax=pltConfig['sLDR']['vmax'])
                            
                colorbar = fig.colorbar(cax, ax=ax, label=pltConfig['sLDR']['label'], shrink=0.8, aspect=40, pad=0.05)
                ax.set_xlim([x_min, x_max])
                x_major_ticks = np.arange(-10, 6, 5)
                x_minor_ticks = np.arange(-10, 6, 1)
                ax.set_xticks(x_major_ticks, minor=False)
                ax.set_xticks(x_minor_ticks, minor=True)
                
                ax.set_ylim([y_min, 10000])
                y_major_ticks = np.arange(0, 11000, 1000)
                y_minor_ticks = np.arange(0, 10200, 200)
                ax.set_yticks(y_major_ticks, minor=False)
                ax.set_yticks(y_minor_ticks, minor=True)

                ax.grid(True, which='major', alpha=0.7, linestyle='-', linewidth=0.8)
                ax.grid(True, which='minor', alpha=0.6, linestyle='-', linewidth=0.5) 
                ax.axvline(x=0, linewidth=1.2, alpha=0.7, linestyle='--')
                ax.axvline(x=-10, linewidth=0.8, alpha=0.7, linestyle='-')
                ax.axvline(x=5, linewidth=0.8, alpha=0.7, linestyle='-')
                ax.axhline(y=0, linewidth=0.8, alpha=0.7, linestyle='-')
                ax.axhline(y=10000, linewidth=0.8, alpha=0.7, linestyle='-')
                
                ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
                ax.set_xlabel('Doppler velocity [m/s]')
                ax.set_ylabel('Height above radar')
                plt.suptitle(f"Spectral LDR for {pd.to_datetime(ds.time.values[closest_valid], unit='s')}", fontweight='bold')
                if savefig:
                    plt.savefig(outname_sLDR, dpi=300, bbox_inches='tight', facecolor='w')
                    print(f"Saved {os.path.basename(outname_sLDR)}")
                else:
                    plt.show()
                plt.close(fig)
                del ldr
    ds.close()
    del ds, x, y
    if plot_Z:
        del sZco_data
    if plot_LDR:
        del sLDR_data


def plot_spectrogram_MXPol(f, output_dir, spq=1, remove_noise=True, noise_method='hildebrandsekhon', stdev_threshold=1, min_bins=3, savefig=True, overwrite=False):
    """
    flte: filename (string)
    output_dir: directory to store plots
    spq: one spectrogram every "spq" minutes (int) - native MXPol-data at 1 second-resolution
    plot_Z: make reflectivity spectrograms
    plot_LDR: make LDR spectrograms
    savefig/overwrite: save/overwrite existing figures

    only plots files when > 0.5% of the range gates contains data above a threshold of -35 dB
    applies 'noise filter' (only preserves data in patches larger than 3x3 range x doppler)
    -- this doesn't remove all 'empty' (only noise) plots but largely reduces the number of useless plots --
    """
    # check if input file not in lost frequency intervals
    MXPol_lost_frequency_intervals = [(datetime.strptime(start, '%Y%m%d-%H%M%S'), datetime.strptime(end, '%Y%m%d-%H%M%S')) for start, end in MXPol_lost_frequency_times]
    pattern = re.compile(r"(\d{8}-\d{6})")
    datestr = pattern.search(f).groups()[0]
    check_dt = datetime.strptime(datestr, '%Y%m%d-%H%M%S')
    for start, end in MXPol_lost_frequency_intervals:
        if start <= check_dt <= end:
            print(f"{os.path.basename(f)} in MXPol lost frequency interval, skipping")
            return

    ds = prepare_MXPoldata(f, remove_noise=remove_noise, noise_method=noise_method, stdev_threshold=stdev_threshold)
    x_edges = calculate_edges(ds['sVel'].values[0])
    y_edges = calculate_edges(ds['range'].values)
    sZH_data = ds['sZH'].values

    x_min, x_max = -10, 5
    y_min, y_max = ds.range.min().item(), 10e3

    sp_int = spq * 60 # time interval for spectrogram plots
    len_time = len(ds.time)

    threshold = -35         # min. dB value 
    min_fraction = 0.05     # require at least 5% valid data points per timestamp (before filtering)

    total_points = ds['sZH'].shape[1] * ds['sZH'].shape[2]
    valid_points = ((np.isfinite(ds['sZH'].values) & (ds['sZH'].values > threshold)).sum(axis=(1, 2)))

    valid_mask = valid_points >= (min_fraction * total_points)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print(f"No valid timestamps found in {os.path.basename(f)}")    
    else:
        for i in range(0, len_time, sp_int):
            closest_valid = valid_indices[np.argmin(np.abs(valid_indices - i))]
            if i > 0:
                prev_closest = valid_indices[np.argmin(np.abs(valid_indices - (i - sp_int)))]
                if closest_valid == prev_closest:
                    continue

            outname_sZ = f"{output_dir}/{pd.to_datetime(ds.time.values[closest_valid], unit='s').strftime('%Y%m%d-%H%M%S')}_sZco.png"
            if not overwrite and os.path.exists(outname_sZ):
                print(f"{outname_sZ} already exists, skipping")
                continue

            z = noise_threshold(noise_threshold(sZH_data[closest_valid, :, :], min_bins=min_bins, axis=1), min_bins=min_bins, axis=0)
                        
            if (np.isfinite(z).sum() / z.size) >=  0.005: # only plot when more than 0.5% of the points contains actual data            
                fig, ax = plt.subplots(figsize=(4, 10), tight_layout=True)
                cax = plt.pcolormesh(x_edges, y_edges, z, cmap=pltConfig['sZco']['cmap'], vmin=pltConfig['sZco']['vmin'], vmax=pltConfig['sZco']['vmax'], shading='auto')
                            
                colorbar = fig.colorbar(cax, ax=ax, label=pltConfig['sZco']['label'], shrink=0.8, aspect=40, pad=0.05)
                ax.set_xlim([x_min, x_max])
                x_major_ticks = np.arange(-10, 6, 5)
                x_minor_ticks = np.arange(-10, 6, 1)
                ax.set_xticks(x_major_ticks, minor=False)
                ax.set_xticks(x_minor_ticks, minor=True)
                
                ax.set_ylim([y_min, y_max])
                y_major_ticks = np.arange(0, 11000, 1000)
                y_minor_ticks = np.arange(0, 10200, 200)
                ax.set_yticks(y_major_ticks, minor=False)
                ax.set_yticks(y_minor_ticks, minor=True)

                ax.grid(True, which='major', alpha=0.7, linestyle='-', linewidth=0.8)
                ax.grid(True, which='minor', alpha=0.6, linestyle='-', linewidth=0.5) 
                ax.axvline(x=0, linewidth=1.2, alpha=0.7, linestyle='--')
                ax.axvline(x=-10, linewidth=0.8, alpha=0.7, linestyle='-')
                ax.axvline(x=5, linewidth=0.8, alpha=0.7, linestyle='-')
                ax.axhline(y=0, linewidth=0.8, alpha=0.7, linestyle='-')
                ax.axhline(y=10000, linewidth=0.8, alpha=0.7, linestyle='-')
                
                ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
                ax.set_xlabel('Doppler velocity [m/s]')
                ax.set_ylabel('Height above radar')
                plt.suptitle(f"Spectral Ze for {pd.to_datetime(ds.time.values[closest_valid], unit='s')}", fontweight='bold')
                if savefig:
                    plt.savefig(outname_sZ, dpi=300, bbox_inches='tight', facecolor='w')
                    print(f"Saved {os.path.basename(outname_sZ)}")
                else:
                    plt.show()
                plt.close()
            del z
    ds.close()
    del ds, x_edges, y_edges, sZH_data


def plot_calibration(data1, data2, names, elv_range, date, times, output_path, limits, savefig=False, writedoc=False):
    """
    Function to plot radar calibration of the two selected datasets, and save calibration 
    constant + information to .txt file.
    Calibration methodology following Jorquera et al. 2023 (doi:10.1175/jtech-d-22-0087.1)

    input:
        data1 and data2 are nan-free flattened datasets (np arrays)
        names: tuple of radar names (origin of data1 and data2)
        date: date of radar calibration
        tmin, tmax: start and end time of interval for which to do calibration
        output_path: location where to store plots & calibration data
    """
    if 'BASTA' in names and len(names) >= 3:
        BASTA_mode = names[2]
    else:
        BASTA_mode = None
    tmin, tmax, t_int = times
    limit_sensitivity, lower_Ze_limit, limit_gradient, intercept = limits
    rmin = -50 # minimum reflectivity plotted
    rmax = 30 # maximum reflectivity plotted

    # set up x-range and fixed slope for plotting upper/lower reflectivity boundaries
    x_range = np.linspace(rmin, rmax, 100)
    a = -1

    cmap = cm.viridis

    year, month, day = date.year, date.month, date.day
    tmin_str = str(tmin.astype('datetime64[us]').tolist()).split('.')[0]
    tmax_str = str(tmax.astype('datetime64[us]').tolist()).split('.')[0]

    data1, data2 = remove_nans(data1, data2)
    # results, best_result, iterations = determine_range_limits(data1, data2)
    _, best_result, _, _, solution_found = determine_range_limits(data1, data2)

    # i, j, r_sq, rmse, data_perc, n_iter = best_result
    i, j, slope, r_sq, rmse, _, _ = best_result
    if j == np.inf: # set j to 15 dBZ in case no upperbest value is found
        j = 15
        
    data1_inside, _, data2_inside, _ = range_selection(data1, data2, i, j)
    data1_filtered, data2_filtered = density_filter(data1_inside, data2_inside)

    cc, cc_std, _ = fit_slope(data1_filtered, data2_filtered)
    perc_acc = len(data1_filtered) / len(data1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist2d(data1, data2, bins=100,
                        range=[[rmin, rmax], [rmin, rmax]],
                        cmap='viridis', norm=LogNorm(), alpha=0.4)
    h = ax.hist2d(data1_filtered, data2_filtered, bins=100,
                  range=[[rmin, rmax], [rmin, rmax]],
                  cmap='viridis', norm=LogNorm())
    ax.plot([rmin, rmax], [rmin, rmax], 'r--', label='1:1 line')
    ax.plot(x_range, x_range + cc, 'b--', label=f"Calibration coefficient ({cc:.3f} dBZ)")
    ax.plot(x_range, a*x_range + i, 'g--', label=f"Lower reflectivity boundary: {i:.2f} dBZ")
    ax.plot(x_range, a*x_range + j, 'k--', label=f"Upper reflectivity boundary: {j:.2f} dBZ")

    # add colorbar
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Count')

    # add labels and title
    hist_all_patch = mpatches.Patch(color=cmap(0.05), alpha=0.4, label='Unfiltered reflectivity pairs')
    hist_filt_patch = mpatches.Patch(color=cmap(0.99), label="Filtered & corrected reflectivity")

    ax.set_xlabel(f"{names[0]} reflectivity [dBZ]")
    ax.set_ylabel(f"{names[1]} reflectivity [dBZ]")
    if not BASTA_mode:
        ax.set_title(f"2D Histogram of {names[0]} vs {names[1]} reflectivity between {int(elv_range[0])} and {int(elv_range[1])} m a.g.l. on {year}-{month:02}-{day:02}{nl}{int(t_int)}s-averaged, {(len(data1_inside) / len(data1))*100:.0f}% used data, {tmin_str} to {tmax_str}")
    else:
        ax.set_title(f"2D Histogram of {names[0]} ({BASTA_mode}) vs {names[1]} reflectivity between {int(elv_range[0])} and {int(elv_range[1])} m a.g.l. on {year}-{month:02}-{day:02}{nl}{int(t_int)}s-averaged, {(len(data1_inside) / len(data1))*100:.0f}% used data, {tmin_str} to {tmax_str}")

    ax.legend(handles=ax.get_legend_handles_labels()[0] + [hist_all_patch, hist_filt_patch], loc='upper left')
    plt.tight_layout()
    if savefig:
        if limit_gradient and limit_sensitivity:
            if BASTA_mode:
                figname = os.path.join(output_path, f"figs/{year}{month:02}{day:02}_{datetime.strptime(tmin_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}-{datetime.strptime(tmax_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}_{int(elv_range[0])}-{int(elv_range[1])}m_{names[0]}{BASTA_mode}_{names[1]}_senslim{lower_Ze_limit/2:.0f}dB_intercept{intercept}.png")
            else:
                figname = os.path.join(output_path, f"figs/{year}{month:02}{day:02}_{datetime.strptime(tmin_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}-{datetime.strptime(tmax_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}_{int(elv_range[0])}-{int(elv_range[1])}m_{names[0]}_{names[1]}_senslim{lower_Ze_limit/2:.0f}dB_intercept{intercept}.png")
        elif limit_gradient and not limit_sensitivity:
            if BASTA_mode:
                figname = os.path.join(output_path, f"figs/{year}{month:02}{day:02}_{datetime.strptime(tmin_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}-{datetime.strptime(tmax_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}_{int(elv_range[0])}-{int(elv_range[1])}m_{names[0]}{BASTA_mode}_{names[1]}_intercept{intercept}.png")
            else:
                figname = os.path.join(output_path, f"figs/{year}{month:02}{day:02}_{datetime.strptime(tmin_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}-{datetime.strptime(tmax_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}_{int(elv_range[0])}-{int(elv_range[1])}m_{names[0]}_{names[1]}_intercept{intercept}.png")
        elif limit_sensitivity and not limit_gradient:
            if BASTA_mode:
                figname = os.path.join(output_path, f"figs/{year}{month:02}{day:02}_{datetime.strptime(tmin_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}-{datetime.strptime(tmax_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}_{int(elv_range[0])}-{int(elv_range[1])}m_{names[0]}{BASTA_mode}_{names[1]}_senslim{lower_Ze_limit/2:.0f}dB.png")
            else:
                figname = os.path.join(output_path, f"figs/{year}{month:02}{day:02}_{datetime.strptime(tmin_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}-{datetime.strptime(tmax_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}_{int(elv_range[0])}-{int(elv_range[1])}m_{names[0]}_{names[1]}_senslim{lower_Ze_limit/2:.0f}dB.png")
        else:
            if BASTA_mode:
                figname = os.path.join(output_path, f"figs/{year}{month:02}{day:02}_{datetime.strptime(tmin_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}-{datetime.strptime(tmax_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}_{int(elv_range[0])}-{int(elv_range[1])}m_{names[0]}{BASTA_mode}_{names[1]}.png")
            else:
                figname = os.path.join(output_path, f"figs/{year}{month:02}{day:02}_{datetime.strptime(tmin_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}-{datetime.strptime(tmax_str, '%Y-%m-%d %H:%M:%S').strftime('%H%M')}_{int(elv_range[0])}-{int(elv_range[1])}m_{names[0]}_{names[1]}.png")

        os.makedirs(f"{output_path}/figs", exist_ok=True)
        plt.savefig(figname)

    vr1 = lower_Ze_limit if limit_sensitivity else ""
    vr2 = intercept if limit_gradient else ""
    new_data = [tmin_str, tmax_str, cc, cc_std, perc_acc, elv_range, vr1, vr2, solution_found, (i, j), slope, r_sq, rmse]

    if writedoc:
        os.makedirs(output_path, exist_ok=True)
        if BASTA_mode:
            calibration_file = os.path.join(output_path, f"calibrationdata_{names[0]}{BASTA_mode}_{names[1]}.txt")
        else:
            calibration_file = os.path.join(output_path, f"calibrationdata_{names[0]}_{names[1]}.txt")
        # vr1 = lower_Ze_limit if limit_sensitivity else ""
        # vr2 = intercept if limit_gradient else ""
        # new_data = [tmin_str, tmax_str, cc, cc_std, perc_acc, elv_range, vr1, vr2, solution_found, (i, j), slope, r_sq, rmse]
        
        create_file_with_headers(headers, calibration_file)
        write_calibrationfile(calibration_file, new_data)
    
    return (i, j), new_data


def plot_calibration_timeline(df, names): 
    """plot calibration coefficients with preference for with gradient limit when both exist"""
    start_date = pd.to_datetime('2024-10-18')
    end_date = pd.to_datetime('2025-01-24')

    df_used, df_excl, stats = calib_stats(df)

    name1, name2 = names

    used_with_solution = df_used[df_used['solution_found'] == 1]
    used_no_solution = df_used[df_used['solution_found'] == -5]

    # Colors
    solution_color = 'green'
    no_solution_color = 'red'      # For solution_found = 1 (prioritized
    excluded_color = 'blue'   # For solution_found = -5
    
    included_mean = stats['included_mean']
    included_std = stats['included_std']
    solution_mean = stats['solution_mean']
    solution_std = stats['solution_std']
    no_solution_mean = stats['no_solution_mean']
    no_solution_std = stats['no_solution_std']
    all_mean = stats['all_mean']
    all_std = stats['all_std']
    
    fig, ax = plt.subplots(figsize=(15, 8))
    # plot used entries with solution_found = 1 (green circles)
    if len(used_with_solution) > 0:
        ax.scatter(used_with_solution['date'], used_with_solution['CC_dB'], 
                  color=solution_color, s=120, marker='o', alpha=1.0, 
                  label='Used calibration values (solution found)', 
                  edgecolor='darkgreen', linewidth=1)
        
        ax.errorbar(used_with_solution['date'], used_with_solution['CC_dB'], 
                   yerr=used_with_solution['std_dev'], fmt='none', 
                   color=solution_color, alpha=0.7, capsize=4)
    
    # Plot used entries with solution_found = -5 (red squares)
    if len(used_no_solution) > 0:
        ax.scatter(used_no_solution['date'], used_no_solution['CC_dB'], 
                  color=no_solution_color, s=120, marker='s', alpha=1.0, 
                  label='Used calibration values (no solution found)', 
                  edgecolor='darkred', linewidth=1)
        
        ax.errorbar(used_no_solution['date'], used_no_solution['CC_dB'], 
                   yerr=used_no_solution['std_dev'], fmt='none', 
                   color=no_solution_color, alpha=0.7, capsize=4)
    
    # average lines and standard deviation bands
    x_span = [start_date, end_date]

    # average of all used values
    if included_mean is not None:
        ax.axhline(y=included_mean, color='black', linestyle='-', linewidth=1.5, 
                  label=f'Average of used calibration values: {included_mean:.2f} ± {included_std:.2f} dB')
        ax.fill_between(x_span, 
                       [included_mean - included_std, included_mean - included_std], 
                       [included_mean + included_std, included_mean + included_std], 
                       color='black', alpha=0.1)
    
    # average of solution_found = 1 values (if any)
    if solution_mean is not None and len(used_with_solution) > 0:
        ax.axhline(y=solution_mean, color=solution_color, linestyle='--', linewidth=1.5, 
                  label=f'Average of solution found-values: {solution_mean:.2f} ± {solution_std:.2f} dB')
    
    # average of solution_found = -5 values (if any)
    if no_solution_mean is not None and len(used_no_solution) > 0:
        ax.axhline(y=no_solution_mean, color=no_solution_color, linestyle='--', linewidth=1.5, 
                  label=f"Average of no solution found-values: {no_solution_mean:.2f} ± {no_solution_std:.2f} dB")
    
    ax.set_xlim(start_date, end_date)
    ax.set_ylim([-1, 8])
    ax.set_xlabel('Date')
    ax.set_ylabel('Calibration Coefficient [dB]')
    ax.set_title(f'{name1}-{name2} calibration timeline', fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', framealpha=1, edgecolor='none')
    
    df['duration'] = pd.to_datetime(df['t_end']) - pd.to_datetime(df['t_start'])
    total_duration = df_used['duration'].sum()
    total_hours = total_duration.total_seconds() / 3600

    # summary text box with calibration statistics
    summary_text = f"Total used entries for calibration: {stats['priority_entries']}\n"
    summary_text += f"Entries with solution found: {len(used_with_solution)}\n"
    summary_text += f"Entries with no solution found: {stats['priority_entries'] - len(used_with_solution)}\n"
    summary_text += f"Total duration: {total_hours:.0f} hours"

    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([start_date, end_date])
    ax2.set_xticklabels([start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
    ax2.tick_params(axis='x', rotation=45, top=False, bottom=True, labeltop=False, labelbottom=True)

    plt.tight_layout()
    return fig, ax


def plot_calibration_quicklook(data1, data2, outputdir, names, elv_range, times, Z_limits, limits, savefig=False, maskname='clean_mask', pltConfig=pltConfig):
    """plotting quicklook of cleaned dataframes used for radar calibration"""
    date = pd.to_datetime(data1.time.values[0]) if isinstance(data1.time.values[0], np.datetime64) else data1.time.values[0]
    year, month, day = date.year, date.month, date.day
    start, end = times
    limit_sensitivity, lower_Ze_limit, limit_gradient, intercept = limits

    if 'BASTA' in names:
        name1, var1, mode, name2, var2 = names
    else:
        name1, var1, name2, var2 = names
    variables_to_plot = [var1, var2]

    # convert MIRA reflectivity to dB
    if 'MIRA' in names:
        data2[var2] = 10*np.log10(data2[var2])

    time_min, time_max = times
    i, j = Z_limits
    
    data_filtered = density_filter(data1[var1], data2[var2])
    data1_final, _, data2_final, _ = range_selection(data_filtered[0], data_filtered[1], i, j)
    
    data = [data1_final, data2_final]
    masks = [data1[maskname], data2[maskname]]
    try:
        fig = plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(2, 2, width_ratios=[30, 1], height_ratios=[1, 1])
       
        first_ax = None
        axs = []
        cbar_axs = []
        for i in range(2):
            ax = plt.subplot(gs[i, 0], sharex=first_ax if first_ax else None, 
                            sharey=first_ax if first_ax else None)
            cbar_ax = plt.subplot(gs[i, 1])
            axs.append(ax)
            cbar_axs.append(cbar_ax)
            if first_ax is None:
                first_ax = ax

        for i, variable in enumerate(variables_to_plot):
            clean_mask = masks[i]
            ax = axs[i]
            cbar_ax = cbar_axs[i]
                    
            norm = pltConfig[variable].get('norm', mcolors.Normalize(
                vmin=pltConfig[variable]['vmin'], vmax=pltConfig[variable]['vmax']
            ))
            
            plotdata = data[i]
            # plotdata = data_withinlimits[i]
            
            # Plot with cached parameters
            img = plotdata.plot.pcolormesh(
                x='time', y='range',
                vmin=pltConfig[variable]['vmin'],
                vmax=pltConfig[variable]['vmax'],
                cmap=pltConfig[variable]['cmap'],
                norm=norm, ax=ax, cbar_ax=cbar_ax,
                extend='both',
                add_colorbar=True,
                add_labels=False
            )
            
            img.colorbar.set_label(pltConfig[variable]['label'])

            # Set limits and labels once
            ax.set_xlim([time_min, time_max])
            ax.set_ylim(elv_range)
            ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
            
            ax.set_ylabel("Height (km)")
            if i == len(variables_to_plot) - 1:  # Only add xlabel to bottom plot
                ax.set_xlabel("Time (HH:MM)")
            else:
                ax.set_xlabel("")
        
        fig.suptitle(f"{name1} and {name2} reflectivity data used for radar calibration on {year}-{month:02}-{day:02}{nl}between {elv_range[0]/1000:.1f} and {elv_range[1]/1000:.1f} km a.g.l. from {pd.to_datetime(time_min).strftime('%H:%M')} to {pd.to_datetime(time_max).strftime('%H:%M')}", 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()

        if savefig:
            if limit_gradient and limit_sensitivity:
                if 'BASTA' in names:
                    outname = f"{outputdir}/{year}{month:02}{day:02}_{pd.to_datetime(start).strftime('%H%M')}-{pd.to_datetime(end).strftime('%H%M')}_{name1}{mode}_{name2}_senslim{lower_Ze_limit/2:.0f}dB_intercept{intercept}.png"
                else:
                    outname = f"{outputdir}/{year}{month:02}{day:02}_{pd.to_datetime(start).strftime('%H%M')}-{pd.to_datetime(end).strftime('%H%M')}_{name1}_{name2}_senslim{lower_Ze_limit/2:.0f}dB_intercept{intercept}.png"
            elif limit_gradient and not limit_sensitivity:
                if 'BASTA' in names:
                    outname = f"{outputdir}/{year}{month:02}{day:02}_{pd.to_datetime(start).strftime('%H%M')}-{pd.to_datetime(end).strftime('%H%M')}_{name1}{mode}_{name2}_intercept{intercept}.png"
                else:
                    outname = f"{outputdir}/{year}{month:02}{day:02}_{pd.to_datetime(start).strftime('%H%M')}-{pd.to_datetime(end).strftime('%H%M')}_{name1}_{name2}_intercept{intercept}.png"
            elif limit_sensitivity and not limit_gradient:
                if 'BASTA' in names:
                    outname = f"{outputdir}/{year}{month:02}{day:02}_{pd.to_datetime(start).strftime('%H%M')}-{pd.to_datetime(end).strftime('%H%M')}_{name1}{mode}_{name2}_senslim{lower_Ze_limit/2:.0f}dB.png"
                else:
                    outname = f"{outputdir}/{year}{month:02}{day:02}_{pd.to_datetime(start).strftime('%H%M')}-{pd.to_datetime(end).strftime('%H%M')}_{name1}_{name2}_senslim{lower_Ze_limit/2:.0f}dB.png"
            else:
                if 'BASTA' in names:
                    outname = f"{outputdir}/{year}{month:02}{day:02}_{pd.to_datetime(start).strftime('%H%M')}-{pd.to_datetime(end).strftime('%H%M')}_{name1}{mode}_{name2}.png"
                else:
                    outname = f"{outputdir}/{year}{month:02}{day:02}_{pd.to_datetime(start).strftime('%H%M')}-{pd.to_datetime(end).strftime('%H%M')}_{name1}_{name2}.png"
            os.makedirs(f"{outputdir}/figs", exist_ok=True)
            plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='w')
        
    except Exception as e:
        error_msg = f"Error {e} while creating quicklook for {year}-{month:02}-{day:02}"
        print(error_msg)
        return error_msg


def add_precipitation_to_plot(ax, df_precip):
    """add precipitation bars to existing axis"""
    if df_precip is None or len(df_precip) == 0:
        return ax
    
    if df_precip['start'].dtype == 'object':
        df_precip = df_precip.copy()  # Don't modify original
        df_precip['start'] = pd.to_datetime(df_precip['start'])
    
    liquid_labeled = False
    solid_labeled = False
    
    for _, period in df_precip.iterrows():
        duration = period['duration_min'] / 60 / 24  # Convert to days

        if period['temp_mean'] is not None and period['temp_mean'] < 1.5 and period['meltheight_mean'] <= 100 and period['duration_min'] > 10:
            # solid precipitation
            label = 'Potentially solid precipitation (T < 1.5°C)' if not solid_labeled else ""
            color = '#87CEEB'
            solid_labeled = True
        else:
            # liquid precipitation
            label = 'Liquid precipitation (T ≥ 1.5°C)' if not liquid_labeled else ""
            color = '#000080'
            liquid_labeled = True
        
        # precipitation indication only at bottom of plot        
        # ax.barh(250, duration, left=mdates.date2num(period['start']), 
        #         height=500, color=color, label=label)

        # precipitation indication in full plot height
        ax.barh(5000, duration, left=mdates.date2num(period['start']),
                height=10e3, color=color, label=label, zorder=0)

        # precipitation indication only at top of plot
        # ax.barh(9750, duration, left=mdates.date2num(period['start']), 
        #         height=500, color=color)
        
    return ax


def plot_DFR(DFRs, df_precip, clip=True, daily=True):
    """plot cloudtop DFRs with precipitation bars and calibration value"""
    start = pd.Timestamp('2024-10-10')
    end = pd.Timestamp('2025-01-25')

    days = [1, 10, 20]

    if daily == True:
        dates = DFRs['date']
        cloud_vals = DFRs['daily_cloudtop_avg']
        if clip == True:
            dfr_vals = np.clip(DFRs['daily_mean_DFR'] + calibrationvalues['BASTA'][0], 0, 20)
        else:
            dfr_vals = DFRs['daily_mean_DFR'] + calibrationvalues['BASTA'][0]
    else:
        dates = DFRs['time']
        cloud_vals = DFRs['cloudtop_avg']
        if clip == True:
            dfr_vals = np.clip(DFRs['mean_DFR'] + calibrationvalues['BASTA'][0], 0, 20)
        else:
            dfr_vals = DFRs['mean_DFR'] + calibrationvalues['BASTA'][0]

    major_ticks = pd.to_datetime([
        pd.Timestamp(year=dt.year, month=dt.month, day=d)
        for dt in pd.date_range(start, end, freq='MS')  # MS = Month Start
        for d in days
        if pd.Timestamp(year=dt.year, month=dt.month, day=1).replace(day=d) <= end
    ])
    add_dates = pd.to_datetime(['2025-01-25', '2024-10-14', '2024-10-20'])
    major_ticks = major_ticks.union(add_dates).sort_values()
    
    plt.clf()    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(dates, dfr_vals, 'o',label='mean DFR', color='red')
    ax1.set_xlabel('Time')
    if daily == True:
        ax1.set_ylabel('Daily mean (uncalibrated) DFR', color='red')
    else:
        ax1.set_ylabel('Mean (uncalibrated) DFR', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.axhline(y=calibrationvalues['BASTA'][0], linewidth=0.8, color='black', label='calibration value +/- 2 stdv')
    ax1.fill_between([start, end], y1=calibrationvalues['BASTA'][0] - 2*calibrationvalues['BASTA'][1], y2=calibrationvalues['BASTA'][0] + 2*calibrationvalues['BASTA'][1], alpha=0.1, color='black')
    ax1.set_xticks(major_ticks)
    ax1.tick_params(axis='x', which='major', length=10)
    ax1.tick_params(axis='x', which='minor', length=6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.grid(True, which='major', axis='x', linestyle=':', linewidth=1., alpha=0.3)
    ax1.grid(True, which='minor', axis='x', linestyle=':', alpha=0.3)
    ax1.set_xlim(datetime(2024, 10, 14), datetime(2025, 1, 25))
    plt.xticks(rotation=45)
    for label in ax1.get_xticklabels():
        label.set_horizontalalignment('right')

    ax2 = ax1.twinx()
    ax2 = add_precipitation_to_plot(ax2, df_precip)
    ax2.plot(dates, cloud_vals, '.', color='gray', label='Cloud top height')
    ax2.set_ylabel('Mean range of cloud top window [km]')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim([0, 10e3])
    ax2.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
    
    if clip == True and daily == True:
        ax1.set_ylim([0, 20])
        plt.suptitle('Daily mean BASTA-MIRA DFR at cloud top height (uncalibrated), clipped to [0, 20] dB', fontweight='bold')
    elif clip == True and daily == False:
        plt.suptitle('Mean BASTA-MIRA DFR at cloud top height (uncalibrated), clipped to [0, 20] dB', fontweight='bold')
    elif clip == False and daily == True:
        plt.suptitle('Daily BASTA-MIRA DFR at cloud top height (uncalibrated)', fontweight='bold')
    else:
        plt.suptitle('Mean BASTA-MIRA DFR at cloud top height (uncalibrated)', fontweight='bold')
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    ax1.legend(loc='upper left')
    # ax2.set_zorder(1)
    ax2.legend(loc='upper right')
    plt.tight_layout()

    return fig


def plot_daily_quicklook(datasets, ERA_data, DFRs, output_dir, savefig=True):
    """daily quicklook using combined zarr datasets"""
    names = list(datasets.keys())
    names = [name for name in datasets.keys() if not name.startswith('DFR_')]
    nplots = sum(len(variables_to_plot[name]) for name in names)
    
    time_ds = 6
    range_ds = 2
    tm = datasets[names[0]]['time'].values
    rg = datasets[names[0]]['range'].values / 1e3
    tm_mesh, rg_mesh = np.meshgrid(tm[::time_ds], rg[::range_ds], indexing='ij')
    tmin = pd.to_datetime(tm[0])
    tmax = pd.to_datetime(tm[-1])

    type_by_var['DFR'] = 'other'
    grouped = {'reflectivity': [], 'velocity': [], 'other': []}
    for name in sorted(names):
        for var in variables_to_plot[name]:
            var_type = type_by_var.get(var, 'other')
            grouped[var_type].append((name, var))

    plot_items = [item for t in ['reflectivity', 'velocity', 'other'] for item in grouped[t]]
    if 'MIRA' in names and 'BASTA' in names:
        DFR_KaW = datasets['MIRA']['Z'] - datasets['BASTA']['reflectivity_attn_DFR_corrected']
        datasets['BASTA']['DFR_KaW'] = DFR_KaW
        nplots = nplots + 1
        try:
            insert_idx = plot_items.index(('MIRA', 'LDRg')) + 1
        except ValueError:
            insert_idx = len(plot_items)
        plot_items.insert(insert_idx, ('BASTA', 'DFR_KaW'))
    if 'MXPol' in names and 'MIRA' in names:
        DFR_XKa = datasets['MXPol']['Zh'] - datasets['MIRA']['Z']
        datasets['MIRA']['DFR_XKa'] = DFR_XKa
        nplots = nplots + 1
        try:
            if ('BASTA', 'DFR_KaW') in plot_items:
                insert_idx = plot_items.index(('MIRA', 'LDRg')) + 2
            else:
                insert_idx = plot_items.index(('MIRA', 'LDRg')) + 1
        except ValueError:
            insert_idx = len(plot_items)
        plot_items.insert(insert_idx, ('MIRA', 'DFR_XKa'))

    # define altitudes in km and V_dop normalised color bar
    altitudes = (ERA_data['altitude'][1] - radar_altitude) / 1e3
    norm_vel = mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=5)

    # get radiosonde times if any
    matches = [pd.to_datetime(tm[0]).strftime('%Y%m%d') in i for i in radiosonde_dates]
    radiosondes = [d for d, m in zip (radiosonde_dates, matches) if m]

    # get cloudtop DFR corrections
    if 'BASTA' in names:
        DFRvals = DFRs[(DFRs['time'] >= tm[0]) & (DFRs['time'] <= tm[-1])]
        lowerDFR, upperDFR = (calibrationvalues['BASTA'][0] - 2*calibrationvalues['BASTA'][1], calibrationvalues['BASTA'][0] + 2*calibrationvalues['BASTA'][1])

    # get MXPol lost frequency intervals
    if 'MXPol' in names:
        MXPol_lost_frequencies = [
            (pd.to_datetime(start, format='%Y%m%d-%H%M%S'), pd.to_datetime(end, format='%Y%m%d-%H%M%S'))
            for start, end in MXPol_lost_frequency_times
        ]
    
    starttime = pd.Timestamp.now()
    print(f"starting plotting at {starttime}")
    fig, axs = plt.subplots(nrows=nplots, figsize=(20, (nplots+1)*3), sharex=True)
    for i, (radar, var) in enumerate(plot_items):
        if radar == 'MXPol':
            time_inds = np.arange(0, len(tm), time_ds)
            range_inds = np.arange(0, len(rg), range_ds)
            data = datasets[radar][var].isel(time=time_inds, range=range_inds).compute().values
        else:
            data = datasets[radar][var][::time_ds, ::range_ds].values
        data = np.ma.masked_invalid(data)
        ax = axs[i]
        if type_by_var[var] == 'velocity':
            im = ax.pcolormesh(tm_mesh, rg_mesh, data, 
                                   cmap=pltConfig[var]['cmap'],
                                   norm=norm_vel,
                                   rasterized=True)
        else:
            im = ax.pcolormesh(tm_mesh, rg_mesh, data,
                                   vmin=pltConfig[var]['vmin'],
                                   vmax=pltConfig[var]['vmax'],
                                   cmap=pltConfig[var]['cmap'],
                                   rasterized=True)

        ax.set_rasterization_zorder(0) 
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(pltConfig[var]['label'])
        ax.set_xlim(pd.Timestamp(tm[0]).round('h'), pd.Timestamp(tm[-1]).round('h'))
        ax.set_ylim(0, 10)
        ax.set_ylabel('Range [km]')
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(labelbottom=True)
        ax.contour(ERA_data['time'], altitudes, ERA_data['t'].T, levels=temps, colors='grey', linewidths=1, linestyles='--', alpha=0.8)
        add_isotherm_labels(ax, ERA_data, altitudes, temps=temps)

        nan_bars = find_nan_intervals(data, tm[::time_ds])
        for start, end in nan_bars:
            ax.axvspan(start, end, color='lightgrey', alpha=0.3, zorder=1, ymin=0, ymax=1, transform=ax.get_xaxis_transform())        

        # add radiosonde timings in top and bottom subplot
        if radiosondes and i in [0, int(0.5*nplots), int(nplots -1)]:
            for rs in radiosondes:
                ax.axvline(x=datetime.strptime(rs, '%Y%m%d%H'), linewidth=2, color='green', alpha=0.8)
        
        # add cloudtop DFR outside +-2 std range flag on BASTA plots:
        if radar == 'BASTA':
            DFRmask = (DFRvals['mean_DFR'] < lowerDFR) | (DFRvals['mean_DFR'] > upperDFR)
            DFRmask_diff = DFRmask.astype(int).diff().fillna(0).ne(0).cumsum()
            DFRints = DFRvals[DFRmask].groupby(DFRmask_diff[DFRmask])
            bars = [(group['time'].iloc[0], group['time'].iloc[-1]) for _, group in DFRints]
            for start, end in bars:
                ax.axvspan(start, end, ymin=0, ymax=0.02, color='purple', transform=ax.get_xaxis_transform())

        # add MXPol lost frequency intervals:
        if radar == 'MXPol':
            valid_bars = [(max(start, tmin), min(end, tmax)) for start, end in MXPol_lost_frequencies if end >= tmin and start <= tmax]
            for start, end in valid_bars:
                ax.axvspan(start, end, ymin=0.03, ymax = 0.05, color='orange', transform=ax.get_xaxis_transform())

    fig.suptitle(f"Quicklooks for {format_list(names)} on {pd.to_datetime(tm[0]).strftime('%Y-%m-%d')}", fontweight='bold', fontsize=18)
    savetime = pd.Timestamp.now()
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if savefig:
        print(f"start saving file at {savetime}{nl}time elapsed {(savetime - starttime).total_seconds():.2f} seconds")
        plt.savefig(f"{output_dir}/{pd.to_datetime(tm[0]).strftime('%Y%m%d')}.png", dpi=150, bbox_inches='tight', facecolor='w')
        plt.close()
    else:
        plt.show()
    endtime = pd.Timestamp.now()
    print(f"finished at {endtime}{nl}total took {(endtime - starttime).total_seconds():.2f} seconds")
    

def plot_peaktree_output(filenames, output_dir, savefig=True, variables=variables_peaktree):
    """
    plot a time-height plot of peaktree output files. Produces one plot per variable.
    - filenames: list of peaktree output.nc-file locations (strings)
    - variables: list of variable names (strings)
    - output_dir: directory to store plots in (string)
    - save_fig: whether to save plots (bool)
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        if isinstance(variables, str):
            variables = [variables]
        elif isinstance(variables, list):
            variables = variables
    except ValueError as ve:
        print(f"{ve}, 'variables' should be passed as a string or a list, quitting.")
        return None
    try:
        if isinstance(filenames, str):
            filenames = [filenames]
        elif isinstance(filenames, list):
            filenames = filenames
    except ValueError as ve:
        print(f"{ve}, 'filenames' should be passed as a string or a list, quitting.")
        return None

    for f in filenames:
        try:
            if isinstance(f, str):
                ds = xr.open_dataset(f)  
            elif isinstance(f, xr.Dataset):
                ds = f
        except ValueError as ve:
            print(f"{ve}: The data to plot should be passed as a filepath or an xarray dataset, {f} is neither of these")
            continue
        
        x = ds.time.values
        y = ds.range.values
        for variable in variables:
            if variable == 'no_nodes':
                z = ds[variable].T
                z = np.ceil(z/2.)
            else:
                try:
                    z = ds.sel(nodes=1)[variable].T
                except KeyError:
                    print(f"Variable {variable} not found, continuing to next variable")
                    continue

            fig, ax = plt.subplots(1, figsize=(12, 6))
            if type_by_var[variable] == 'velocity':
                im = ax.pcolormesh(x, y, z,
                                   cmap=pltConfig[variable]['cmap'],
                                   norm=mcolors.TwoSlopeNorm(vmin=pltConfig[variable]['vmin'], vcenter=0, vmax=pltConfig[variable]['vmax']))
            else:
                im = ax.pcolormesh(x, y, z,
                                   cmap=pltConfig[variable]['cmap'], 
                                   vmin=pltConfig[variable]['vmin'], 
                                   vmax=pltConfig[variable]['vmax'])
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label(pltConfig[variable]['label'])
            ax.set_xlim(pd.Timestamp(x[0]).round('h'), pd.Timestamp(x[-1]).round('h'))
            ax.set_ylim([0, 10e3])
            ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.suptitle(f"Peaktree output for {pltConfig[variable]['label_short']} on {pd.to_datetime(x[0]).strftime('%Y-%m-%d %H:%M')}", fontweight='bold', fontsize=18)
            if savefig:
                plt.savefig(f"{output_dir}/{Path(f).stem}_{variable}.png", dpi=150, bbox_inches='tight', facecolor='w')
                plt.close()
            else:
                plt.show()
        ds.close()
        

def plot_MIRA_scans(fn, outputdir, savefig=True):
    """quicklooks of MIRA scans using pyart"""
    radar = znc_to_pyart(fn)
    display = pyart.graph.RadarDisplay(radar)
    for variable in variables_to_plot_pyart:
        data = radar.fields[variable]['data']
        if np.ma.isMaskedArray(data):
            if np.all(data.mask):
                continue
        else:
            if np.isnan(data).mean() >= 0.9:
                continue

        plot = None
        if 'rhi' in fn:
            plot = 'rhi'
            fig, ax = plt.subplots(figsize=(12, 6))
            if type_by_var[variable] == 'velocity':
                display.plot_rhi(variable, cmap=pltConfig[variable]['cmap'], norm=mcolors.TwoSlopeNorm(vmin=pltConfig[variable]['vmin'], vcenter=0, vmax=pltConfig[variable]['vmax']), reverse_xaxis=True)
            else:
                display.plot_rhi(variable, vmin=pltConfig[variable]['vmin'], vmax=pltConfig[variable]['vmax'], cmap=pltConfig[variable]['cmap'], reverse_xaxis=True)
            ax.set_xlim([-12, 12])
            ax.set_ylim([0, 10])
            ax.set_title(f"MIRA RHI at {radar.time['units'].split('since')[1].strip()}, {int(np.nanmean(radar.azimuth['data']))}$^\circ$ azimuth")
            outname = f"{outputdir}/{Path(fn).stem.split('.')[0]}_{plot}_{variable}.png"
        if 'ppi' in fn:
            plot = 'ppi'
            fig, ax = plt.subplots(figsize=(8, 6))
            if type_by_var[variable] == 'velocity':
                display.plot_ppi(variable, cmap=pltConfig[variable]['cmap'], norm=mcolors.TwoSlopeNorm(vmin=pltConfig[variable]['vmin'], vcenter=0, vmax=pltConfig[variable]['vmax']))
            else:
                display.plot_ppi(variable, vmin=pltConfig[variable]['vmin'], vmax=pltConfig[variable]['vmax'], cmap=pltConfig[variable]['cmap'])
            ax.set_xlim([-4, 20])
            ax.set_ylim([-4, 20])
            ax.set_title(f"MIRA sector scan at {radar.time['units'].split('since')[1].strip()}, {int(np.nanmean(radar.elevation['data']))}$^\circ$ elevation")
            outname = f"{outputdir}/{Path(fn).stem.split('.')[0]}_{plot}_{variable}.png"
        if plot and savefig:
            plt.savefig(outname, dpi=150, bbox_inches='tight', facecolor='w')
            plt.close()


def plot_triple_frequency(DFR1, DFR2, altitudes, temp_range, output_dir, savefig=True, filtered=False):
    """
    plots triple frequency histogram for specified temperatures.
    DFR1 & DFR2: xr DataArrays (DFR1 being the higher frequency combination)
    altitudes: xr DataArray reindexed to DFR-data time range
    temp_range: tuple (max temp, min temp)
    output_dir: string
    """
    if temp_range:
        alt1 = altitudes[f"temp_{temp_range[0]}C_altitude"]
        alt2 = altitudes[f"temp_{temp_range[1]}C_altitude"]
        DFR1_crop = DFR1.where((DFR2.coords['range'] >= alt1) & (DFR2.coords['range'] <= alt2))
        DFR2_crop = DFR2.where((DFR2.coords['range'] >= alt1) & (DFR2.coords['range'] <= alt2))
        dfr1_crop_clean, dfr2_crop_clean = flatten_for_densityplot(DFR1_crop.values, DFR2_crop.values)
    else:
        dfr1_crop_clean, dfr2_crop_clean = flatten_for_densityplot(DFR1.values, DFR2.values)
    plt.figure(figsize=(10, 8))
    hist, xedges, yedges = np.histogram2d(dfr1_crop_clean, dfr2_crop_clean,
                                          bins=250, density=True)
    hist_percent = hist * 100
    hist_masked = np.ma.masked_where(hist_percent < 0.01, hist_percent)
    
    im = plt.imshow(hist_masked.T, origin='lower', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    cmap='turbo', aspect='auto')
    plt.colorbar(im, label='Relative Frequency [%]', format='%.1f')
    plt.text(0.02, 0.98, f"N = {len(dfr1_crop_clean):.2e}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    plt.xlabel('DFR_KaW [dB]')
    plt.ylabel('DFR_XKa [dB]')
    plt.xlim([-6, 15])
    plt.ylim([-6, 15])
    if temp_range and filtered:
        plt.title(f"triple-frequency histogram (temp between {int(temp_range[0])} and {int(temp_range[1])}degC)")
        outname = f"{output_dir}/triplefreqhist_{int(temp_range[0])}to{int(temp_range[1])}_filtered.png"
    elif temp_range and not filtered:
        plt.title(f"triple-frequency histogram (temp between {int(temp_range[0])} and {int(temp_range[1])}degC){nl}X-band filtered to SNR > 15 dB only")
        outname = f"{output_dir}/triplefreqhist_{int(temp_range[0])}to{int(temp_range[1])}.png"
    elif filtered:
        plt.title(f"triple-frequency histogram (full temperature range){nl}X-band filtered to SNR > 15 dB only")
        outname = f"{output_dir}/triplefreqhist_(fulltemprange)_filtered.png"
    else:
        plt.title(f"triple-frequency histogram (full temperature range)")
        outname = f"{output_dir}/triplefreqhist_(fulltemprange).png"
    if savefig:
        plt.savefig(outname, dpi=150, bbox_inches='tight', facecolor='w')
        plt.close()


def plot_variable_summary(radar, ds, variable, dimname, save_fig=True):
    """
    summary plot of radar variable showing statistics across chosen dimension (range/no_nodes)
    radar : str, radar name (e.g., 'BASTA', 'MIRA', etc.)    
    ds : xarray Dataset, radar statistics dataset
    var_name : str, variable name (e.g., 'reflectivity', 'velocity', etc.)
    dimname : str, dimension name ('no_peaks' or 'range')
    save_fig : bool, whether to save the figure to file
    """
    sns.set_style("whitegrid")
    assert dimname in ['range', 'no_peaks'], "dimname must be either 'range' or 'no_peaks'"

    if dimname == 'range':
        var_name = variable
        varname_plot = f"{variable}_range"
    if dimname == 'no_peaks':
        var_name = f"{variable}_npeaks"
        varname_plot = f"{variable}_npeaks"

    count_var = f'{var_name}_count'
    label_var = default_variable_maps[radar][variable]
    
    fig, ax1 = plt.subplots(figsize=(6, 10))
    y = ds[dimname].values
   
    # Plot statistics on primary x-axis (range on y-axis)
    ax1.plot(ds[f'{var_name}_mean'].values, y, 'o-', label='mean', 
             linewidth=2, markersize=4)
    ax1.plot(ds[f'{var_name}_median'].values, y, 's-', label='median', 
             linewidth=2, markersize=4)
    
    # Plot min/max as shaded region
    ax1.fill_betweenx(y, 
                      ds[f'{var_name}_min'].values, 
                      ds[f'{var_name}_max'].values, 
                      alpha=0.2, label='min-max range')
    
    # Plot standard deviation as error bars or shaded region around mean
    ax1.fill_betweenx(y, 
                      ds[f'{var_name}_mean'].values - ds[f'{var_name}_std'].values,
                      ds[f'{var_name}_mean'].values + ds[f'{var_name}_std'].values,
                      alpha=0.3, label='mean ± std dev')
    
    # Set labels and title for primary axis
    ax1.set_xlim(pltConfig[label_var]['vmin'], pltConfig[label_var]['vmax'])
    ax1.set_xlabel(pltConfig[label_var]['label'], fontsize=12)
    ax1.set_title(f"{radar} {pltConfig[label_var]['label_short']} statistics vs. range", 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary x-axis for count at top (reverse axis)
    ax2 = ax1.twiny()
    ax2.plot(ds[count_var].values, y, alpha=0.5, color='k', label='sample count')
    ax2.set_xlabel('sample count', fontsize=12)
    ax2.xaxis.set_inverted(True)
    ax2.legend(loc='upper left', fontsize=10)
    
    if dimname == 'range':
        ax1.set_ylim([0, 12e3])
        # ax1.set_yticks(np.arange(0, 12e3, 2e3), minor=False)
        ax1.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
        
    if dimname == 'no_peaks':
        ax1.set_ylim([1, 5])
        ax1.set_yticks([1, 2, 3, 4, 5])
        ax1.set_ylabel('Number of peaks', fontsize=12) 
        ax2.set_xscale('log')
        
        ax1_ticks = ax1.get_xticks()
        ax1_min, ax1_max = ax1.get_xlim()

        count_values = ds[count_var].values
        count_values = count_values[np.isfinite(count_values) & (count_values > 0)]
        
        count_min = np.nanmin(count_values)
        count_max = np.nanmax(count_values)

        if np.isfinite(count_min) and np.isfinite(count_max):
            ax2_min = 10 ** np.floor(np.log10(count_min))
            ax2_max = 10 ** np.ceil(np.log10(count_max))

            ax1_positions = (ax1_ticks - ax1_min) / (ax1_max - ax1_min)
            ax2_ticks = ax2_max * (ax2_min / ax2_max) ** ax1_positions
            
            ax2.set_xlim(ax2_min, ax2_max)
            ax2.set_xticks(ax2_ticks)
            
            def scientific_format(x, pos):
                if x == 0:
                    return '0'
                exponent = int(np.floor(np.log10(abs(x))))
                mantissa = x / 10**exponent
                return f'${mantissa:.1f} \\times 10^{{{exponent}}}$'
            
            ax2.xaxis.set_major_formatter(FuncFormatter(scientific_format))
            
            ax2.tick_params(axis='x', which='major', labelsize=8)
            ax2.xaxis.set_tick_params(labeltop=True)
        else:
            # Fallback
            ax2.set_xlim(1e6, 1)

    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f"{radar}_{varname_plot}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_variables(ds, variables=None, range_dim='range', save_figs=True):
    """
    summary plots for all radar variables.    
    ds : xarray Dataset, dataset containing the radar statistics
    variables : list, optional, list of variable names to plot (if None, will infer from data variables)
    range_dim : str, name of range dimension
    save_figs : bool
    """
    if variables is None:
        # Infer variable names from data variables
        variables = []
        for var in ds.data_vars:
            if '_mean' in var:
                base_var = var.replace('_mean', '')
                if base_var not in variables:
                    variables.append(base_var)
    
    print(f"Creating plots for variables: {variables}")
    
    for var in variables:
        print(f"\nPlotting {var}...")
        plot_variable_summary(ds, var, range_dim=range_dim, save_fig=save_figs)


def plot_all_in_one(ds, variables=None, range_dim='range', save_fig=True):
    """
    create a single figure with subplots for all variables
    
    ds : xarray Dataset, dataset containing the radar statistics
    variables : list, optional, list of variable names to plot (if None, will infer from data variables)
    range_dim : str, name of range dimension
    save_figs : bool
    """
    if variables is None:
        variables = ['reflectivity', 'velocity', 'spectrum_width', 'num_peaks', 'cloudtop']
    # filter to only variables that exist in the dataset
    variables = [v for v in variables if f'{v}_mean' in ds.data_vars]
    
    if not variables:
        print("No valid variables found in dataset")
        return
    
    # determine subplot layout
    n_vars = len(variables)
    n_cols = 2
    n_rows = (n_vars + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    if range_dim in ds.dims:
        x = ds[range_dim].values
        xlabel = f'{range_dim.title()} (km)' if range_dim == 'range' else range_dim.title()
    else:
        x = np.arange(len(ds[f'{variables[0]}_mean']))
        xlabel = 'Index'
    
    for idx, var in enumerate(variables):
        ax1 = axes[idx]
        
        ax1.plot(ds[f'{var}_mean'].values, x, 'o-', label='Mean', 
                linewidth=2, markersize=3)
        ax1.fill_betweenx(x, 
                          ds[f'{var}_min'].values, 
                          ds[f'{var}_max'].values, 
                          alpha=0.2, label='Min-Max')
        
        ax1.set_ylabel(xlabel, fontsize=10)
        ax1.set_xlabel(f'{var.replace("_", " ").title()}', fontsize=10)
        ax1.set_title(var.replace("_", " ").title(), fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower left', fontsize=8)
        
        ax2 = ax1.twiny()
        height = (x[1] - x[0]) * 0.8 if len(x) > 1 else 1
        ax2.barh(x, ds[f'{var}_count'].values, alpha=0.2, color='gray', height=height)
        ax2.set_xlabel('Count', fontsize=9, color='gray')
        ax2.tick_params(axis='x', labelsize=8, labelcolor='gray')
    
    # remove extra subplots if odd number
    for idx in range(len(variables), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Radar Campaign Statistics Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('all_variables_overview.png', dpi=300, bbox_inches='tight')
        print("Saved all_variables_overview.png")
    
    plt.show()
    
    return fig


def plot_density_histogram(x, y, xlabel, ylabel='Range [km]', title=None, 
                           cmap='plasma', bins=None, log_scale=True):
    """plotting density histograms w.r.t range"""

    fig, ax = plt.subplots(figsize=(6,8))
    if bins:
        xbins = ybins = bins
        h, xedges, yedges = np.histogram2d(x, y, bins=bins)
    else:
        xbins = determine_nbins(x)
        ybins = determine_nbins(y)
        h, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
    h = np.ma.masked_where(h == 0, h)

    if log_scale:
        im = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmin=1), shading='auto')
    else:
        im = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, shading='auto')

    if title is None:
        title = f"Density: {xlabel} vs. {ylabel}"
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim([0, 12e3])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
    ax.grid(True, alpha=0.5)

    plt.colorbar(im, ax=ax, label='Count')
    ax.figure.text(0.04, 0.97, f'N = {len(x):,}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.figure.text(0.67, 0.98, f'# of x_bins = {int(xbins)}\n# of y_bins = {int(ybins)}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    return fig, ax


def plot_density(x, y, xlabel, ylabel='range [km]', bins=None, title=None, cmap='plasma', log_scale=True):
    """plotting density histograms w.r.t. range (hexagonal binning)"""
    if not bins:
        bins = determine_nhexbins(x, y)

    fig, ax = plt.subplots(figsize=(6, 8))
    if log_scale:
        hb = ax.hexbin(x, y, gridsize=bins, cmap=cmap, norm=LogNorm(vmin=1))
    else:
        hb = ax.hexbin(x, y, gridsize=bins, cmap=cmap, mincnt=1, shading='auto')

    if title is None:
        title = f"Density: {xlabel} vs. {ylabel}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim([0, 12e3])
    ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
    ax.grid(True, alpha=0.5)

    plt.colorbar(hb, ax=ax, label='Count')
    ax.figure.text(0.07, 0.98, f'N = {len(x):,}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.figure.text(0.72, 0.98, f'# of bins = {int(bins)}', 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()

    return fig, ax


def plot_density_subplots(data_list, xlabel_list, ylabel='range [km]', bins=None, 
                          titles=None, cmap='plasma', log_scale=True, figsize=(18, 6)):
    """
    figure with 3 subplots for density plots
    data_list : list of tuples, (x, y) data pairs for each subplot
    xlabel_list : list of str, x-axis labels for each subplot
    titles : list of str, optional, titles for each subplot
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for idx, (ax, (x, y), xlabel) in enumerate(zip(axes, data_list, xlabel_list)):
        # Determine bins if not provided
        plot_bins = bins if bins else determine_nhexbins(x, y)
        
        # Create hexbin plot
        if log_scale:
            hb = ax.hexbin(x, y, gridsize=plot_bins, cmap=cmap, norm=LogNorm(vmin=1))
        else:
            hb = ax.hexbin(x, y, gridsize=plot_bins, cmap=cmap, mincnt=1)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=13, fontweight='bold')
        else:
            ax.set_title(f"Density: {xlabel} vs. {ylabel}", fontsize=13, fontweight='bold')
        
        ax.set_ylim([0, 12e3])
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
        ax.grid(True, alpha=0.5)
        
        # Add colorbar
        plt.colorbar(hb, ax=ax, label='Count')
        
        # Add text annotations
        ax.text(0.07, 0.98, f'N = {len(x):,}', 
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.72, 0.98, f'# of bins = {int(plot_bins)}', 
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, axes


def weekly_plot(full_dataset, ERA_data, start, end, figpath, savefig=True):
    """
    triple frequency weekly data visualisation, using combined zarr dataset and ERA5 data (outdated - use paper_plots)
    full_dataset (xr Dataset): combined radar zarr dataset (multi-frequency, multi-mode)
    ERA_data (xr Dataset): ERA5 reanalysis dataset (temperature, altitude) for isotherm overlay
    start, end (datetime): datetime bounds of the week to plot
    figpath (str): output path for the saved figure
    savefig (bool): if True, save and close; if False, show interactively
    """
    # define altitudes in km and V_dop normalised color bar
    altitudes = (ERA_data['altitude'][1] - radar_altitude)
    norm_vel = mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=5)
    DFRlim = 2*calibrationvalues['BASTA'][1]
    MXPol_lost_frequencies = [
        (pd.to_datetime(startm, format='%Y%m%d-%H%M%S'), pd.to_datetime(endm, format='%Y%m%d-%H%M%S'))
        for startm, endm in MXPol_lost_frequency_times
        ]
    
    # get radiosonde times if any
    radiosondes = [
        d.strftime('%Y%m%d%H')
        for d in radiosonde_datetimes
        if start <= d <= end
    ]

    rdr = 'Ka' if end > datetime(2024, 11, 12) else 'W'
    plotvars = [f"Z_{rdr}", f"velocity_{rdr}", 'width_Ka', 'LDR_Ka']
    keys = ['Z', 'VELg', 'RMSg', 'LDR'] if rdr == 'Ka' else ['reflectivity', 'velocity', 'RMSg', 'LDR']
    
    subset = full_dataset.sel(time=slice(start, end))
    ERA_subset = ERA_data.sel(time=slice(start, end))

    BASTA_intervals = mask_to_intervals(subset['Z_W'].notnull().any(dim='range'))
    DFR_intervals = mask_to_intervals((subset['DFRcorrection_W'] >= DFRlim) & subset['Z_W'].notnull().all(dim='range'))
    MXPol_intervals = mask_to_intervals(subset['Z_X'].notnull().any(dim='range'))

    lostfreq_mask = pd.Series(False, index=subset['time'].to_pandas())
    for startx, endx in MXPol_lost_frequencies:
        lostfreq_mask.loc[(lostfreq_mask.index >= startx) & (lostfreq_mask.index <= endx)] = True
    
    lostfreq_mask = xr.DataArray(lostfreq_mask.values, coords={'time': subset['time'].values}, dims=['time'])
    lostfreq_intervals = mask_to_intervals(lostfreq_mask)
    
    oro_intervals = mask_to_intervals(subset['orographic'])
    front_intervals = mask_to_intervals(subset['frontal'])
    # precip_intervals = mask_to_intervals(subset['is_precip'])
    
    fig, axs = plt.subplots(nrows=4, figsize=(20, 16))

    for i, ax in enumerate(axs):
        dataset = subset[plotvars[i]]
        if type_by_var[keys[i]] == 'velocity':
            im = ax.pcolormesh(dataset.time.values, dataset.range.values, 
                            dataset.values.T, 
                            cmap=pltConfig[keys[i]]['cmap'],
                            norm=norm_vel, rasterized=True)
        else:
            im = ax.pcolormesh(dataset.time.values, dataset.range.values, dataset.values.T, 
                            cmap=pltConfig[keys[i]]['cmap'], 
                            vmin=pltConfig[keys[i]]['vmin'], vmax=pltConfig[keys[i]]['vmax'],
                            rasterized=True)
        ax.plot(subset.time.values, subset.cloudtop.values, 'k', linewidth=0.8)
        ax.set_rasterization_zorder(0) 
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(pltConfig[keys[i]]['label'])
        
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
        ax.set_xlim([start, end])
        ax.set_ylim([0, 12e3])
        ax.grid(True, alpha=0.5)
        ax.xaxis.grid(True, which='major', color='gray', linewidth=0.8)

        ax.contour(ERA_subset['time'], altitudes, ERA_subset['t'].T, levels=temps, colors='gray', linewidths=1, linestyles='--', alpha=0.8)
        add_isotherm_labels(ax, ERA_subset, altitudes, alt_range=(0, 12e3), temps=temps, xlims=(start, end))

        if i in [0, 2] and radiosondes:
            for rs in radiosondes:
                ax.axvline(x=datetime.strptime(rs, '%Y%m%d%H'), linewidth=1.5, color='green', alpha=0.8)

        # orographic vs. frontal cloud flags (above 2nd and bottom plot)
        if i in [1, 3]:
            for starto, endo in oro_intervals:
                mask = (subset.time.values >= starto) & (subset.time.values <= endo)
                ax.fill_between(subset.time.values, subset.cloudtop.values, 12e3, 
                            where=mask, color='deepskyblue', alpha=0.15, zorder=0)
            for startf, endf in front_intervals:
                mask = (subset.time.values >= startf) & (subset.time.values <= endf)
                ax.fill_between(subset.time.values, subset.cloudtop.values, 12e3,
                            where=mask, color='fuchsia', alpha=0.15, zorder=0)
    
        # BASTA & MXPol data available second and bottom plot
        if i in [1, 3]:
            ax_bars1_bottom = ax.inset_axes([0, -0.04, 1, 0.015])
            ax_bars1_bottom.set_xlim(ax.get_xlim())
            ax_bars1_bottom.set_ylim([0, 1])
            ax_bars1_bottom.axis('off')

            ax_bars2_bottom = ax.inset_axes([0, -0.075, 1, 0.015])
            ax_bars2_bottom.set_xlim(ax.get_xlim())
            ax_bars2_bottom.set_ylim([0, 1])
            ax_bars2_bottom.axis('off')

            if rdr == 'Ka':
                for startb, endb in BASTA_intervals:
                    ax_bars1_bottom.axvspan(startb, endb, ymin=0, ymax=1, color='darkgreen')
            for startb, endb in DFR_intervals:
                ax_bars1_bottom.axvspan(startb, endb, ymin=0, ymax=1, color='limegreen')
            
            for startb, endb in MXPol_intervals:
                ax_bars2_bottom.axvspan(startb, endb, ymin=0, ymax=1, color='saddlebrown')
            for startb, endb in lostfreq_intervals:
                ax_bars2_bottom.axvspan(startb, endb, ymin=0, ymax=1, color='tab:orange')
            
            ax_bars1_bottom.set_clip_on(False)
            ax_bars2_bottom.set_clip_on(False)
            
        ax.tick_params(axis='x', pad=30)

    plt.subplots_adjust(top=0.88)
    plt.suptitle(f"Radar data overview for {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}", fontweight='bold', fontsize=18)
    plt.tight_layout()
    
    if savefig:
        plt.savefig(figpath, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()
        print(f"Saved {figpath}")
    else:
        plt.show()


def paper_plots(ds_subset, ds_mode, ERA_data, HALO_data, figpath, y_range=(0, 12e3), savefig=True):
    """    
    two publication-style figures (4 and 6-panels) combining Ka/W-band reflectivity, velocity, LDR, and DFR panels 
    with ERA5 isotherms, HALO PBL height, radiosonde markers, and BASTA/MXPol/DFR/lost-frequency availability bars. 
    ds_subset (xr Dataset): full radar dataset for the time period to plot (from zarr)
    ds_mode (xr Dataset): mode-specific BASTA dataset for the time period to plot (from zarr)
    ERA_data (xr Dataset): ERA5 reanalysis dataset (temperature, altitude) for isotherm overlay of days to plot
    HALO_data (xr Dataset): HALO dataset for PBL height information for time period to plot
    figpath (str): output path for the saved figure
    y_range (tuple): vertical axis limits
    savefig (bool): if True, save and close; if False, show interactively"""
    assert np.all(ds_subset['time'].values == ds_mode['time'].values), "datasets don't cover the same time"
    T = Timer()
    T.log("start")

    altitudes = (ERA_data['altitude'] - radar_altitude)
    norm_vel = mcolors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=2)
    time_vals = ds_subset['time'].values
    T.log("setup constants")

    HALO_mask = ((HALO_data['bl_classification_3min'] > 0) & (HALO_data['height'] < 8000)).load()
    pbltimes = HALO_data['time_3min'].values
    pblheight = HALO_data['height'].where(HALO_mask).max(dim='height').values
    jumps = np.abs(np.diff(pblheight, prepend=np.nan, append=np.nan))
    bad = (jumps[:-1] > 2000) | (jumps[1:] > 2000)
    pblheight[bad] = np.nan    
    T.log("HALO load mask & alignment")

    cloudtopDFRs = ds_subset['DFRcorrection_W'].values
    try:
        ZW_valid = (
            ds_mode['reflectivity'].where(ds_mode['mask_artefacts_loose'] == 1).notnull().any(dim='range')
            & ds_subset['DFRcorrection_W'].notnull()
        )
        
        BASTA_intervals = mask_to_intervals(ZW_valid)
        BASTA_intervals_merged = merge_close_intervals(BASTA_intervals)
        mask_basta = intervals_to_mask(BASTA_intervals_merged, time_vals)[None, :]
        BASTA_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping BASTA bars — {e}")
        BASTA_bars = False
    try:
        lowerDFR = calibrationvalues['BASTA'][0] - 2*calibrationvalues['BASTA'][1]
        upperDFR = calibrationvalues['BASTA'][0] + 2*calibrationvalues['BASTA'][1]
        DFR_invalid = (((ds_subset['DFRcorrection_W'] < lowerDFR) | (ds_subset['DFRcorrection_W'] > upperDFR)) 
                       & ds_mode['reflectivity'].where(ds_mode['mask_artefacts_loose'] == 0).notnull().any(dim='range'))
        DFR_intervals = mask_to_intervals(DFR_invalid)
        DFR_intervals_merged = merge_close_intervals(DFR_intervals)
        mask_dfr = intervals_to_mask(DFR_intervals_merged, time_vals)[None, :]
        DFR_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping DFR bars — {e}")
        DFR_bars = False
    T.log("BASTA intervals")
    
    try:
        ZX_valid = ds_subset['DFR_XKa'].notnull().any(dim='range')
        MXPol_intervals = mask_to_intervals(ZX_valid)
        MXPol_intervals_merged = merge_close_intervals(MXPol_intervals)
        mask_mxpol = intervals_to_mask(MXPol_intervals_merged, time_vals)[None, :]
        MXPol_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping MXPol bars — {e}")
        MXPol_bars = False
    try:
        MXPol_lost_frequencies = [
            (pd.to_datetime(startm, format='%Y%m%d-%H%M%S'),
            pd.to_datetime(endm, format='%Y%m%d-%H%M%S'))
            for startm, endm in MXPol_lost_frequency_times
        ]
        lostfreq_mask = pd.Series(False, index=pd.to_datetime(time_vals))
        for startx, endx in MXPol_lost_frequencies:
            lostfreq_mask.loc[(lostfreq_mask.index >= startx) & (lostfreq_mask.index <= endx)] = True
        lostfreq_da = xr.DataArray(lostfreq_mask.values,
                                    coords={'time': lostfreq_mask.index.values}, dims='time')
        lostfreq_intervals = mask_to_intervals(lostfreq_da)   # no second positional arg
        lostfreq_intervals_merged = merge_close_intervals(lostfreq_intervals)
        mask_lostfreq = intervals_to_mask(lostfreq_intervals_merged, time_vals)[None, :]
        LF_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping lost frequency bars — {e}")
        LF_bars = False
    T.log("MXPol intervals")

    radiosondes = [d.strftime('%Y%m%d%H') for d in radiosonde_datetimes if time_vals[0] <= d <= time_vals[-1]]
    T.log("radiosondes")
    
    keys_ext = ['Z', 'reflectivity', 'VELg', 'LDRg', 'DFR_KaW', 'DFR_XKa']
    keys = ['Z', 'reflectivity', 'VELg', 'DFR_KaW']
    plots_ext = {0: ds_subset['Z_Ka'], 
             1: ds_mode['reflectivity'].where(ds_mode['mask_artefacts_loose'] == 0), 
             2: ds_subset['velocity_Ka'],
             3: ds_subset['LDR_Ka'],
             4: ds_subset['DFR_KaW'],
             5: ds_subset['DFR_XKa']}
    plots = {0: ds_subset['Z_Ka'], 
             1: ds_mode['reflectivity'].where(ds_mode['mask_artefacts_loose'] == 0), 
             2: ds_subset['velocity_Ka'],
             3: ds_subset['DFR_KaW']}
    

    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(nrows=6, ncols=1, figure=fig, hspace=0.3)
    T.log("figure created")
    
    axs = []
    for i in range(6):
        axs.append(fig.add_subplot(gs[i, 0]))
    
    for i, ax in enumerate(axs):
        T.log(f"subplot {i} start")
        dset = plots_ext[i]
        timevals = mdates.date2num(dset.time.values)
        rangevals = dset.range.values

        if type_by_var[keys_ext[i]] == 'velocity':
            im = ax.pcolormesh(timevals, rangevals, dset.values.T, cmap='seismic', norm=norm_vel, rasterized=True)
        else:
            im = ax.pcolormesh(timevals, rangevals, dset.values.T, cmap=pltConfig[keys_ext[i]]['cmap'], vmin=pltConfig[keys_ext[i]]['vmin'], vmax=pltConfig[keys_ext[i]]['vmax'], rasterized=True)
        T.log(f"subplot {i}: pcolormesh")

        if i == 1:
            ax2 = ax.twinx()
            ax2.plot(time_vals, cloudtopDFRs, color='red', linewidth=1, alpha=0.8)
            ax2.set_ylim(0, 36)
            ax2.set_yticks(np.linspace(0, 36, 7))
            ax2.set_ylabel('BASTA DFR-based correction [dB]', color='red', labelpad=6)
            ax2.tick_params(axis='y', labelcolor='red')
            
            cbar_ax = ax.inset_axes([1.055, 0.1, 0.02, 0.8])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label(pltConfig[keys_ext[i]]['label'])
            T.log(f"subplot {i}: BASTA correction based on DFR")
        else:
            cbar_ax = ax.inset_axes([1.02, 0.1, 0.02, 0.8])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label(pltConfig[keys_ext[i]]['label'])        
        T.log(f"subplot {i}: colorbar")

        if i == 1 and (any([BASTA_bars, DFR_bars, MXPol_bars, LF_bars])): # add intervals 
            ax_bars1 = ax.inset_axes([0, -0.08, 1, 0.025])
            ax_bars2 = ax.inset_axes([0, -0.14, 1, 0.025])

            for a in (ax_bars1, ax_bars2):
                a.set_xlim(ax.get_xlim())
                a.set_ylim([0, 1])
                a.axis('off')

            if BASTA_bars:
                ax_bars1.imshow(mask_basta, aspect='auto', vmin=0, vmax=1,
                                cmap=mcolors.ListedColormap(['none', 'darkgreen']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if DFR_bars:
                ax_bars1.imshow(mask_dfr, aspect='auto', vmin=0, vmax=1,
                                cmap=mcolors.ListedColormap(['none', 'limegreen']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if MXPol_bars:
                ax_bars2.imshow(mask_mxpol, aspect='auto', vmin=0, vmax=1,
                                cmap=mcolors.ListedColormap(['none', 'saddlebrown']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if LF_bars:
                ax_bars2.imshow(mask_lostfreq, aspect='auto', vmin=0, vmax=1,
                                cmap=mcolors.ListedColormap(['none', 'tab:orange']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            ax_bars1.set_clip_on(False)
            ax_bars2.set_clip_on(False)
            ax.tick_params(axis='x', pad=15)
            T.log(f"subplot {i}: interval bars")

        if i == 2 and radiosondes:
            for rs in radiosondes:
                ax.axvline(x=datetime.strptime(rs, '%Y%m%d%H'), linewidth=1.5, color='green', alpha=0.8, zorder=10)
            T.log(f"subplot {i}: radiosonde lines")

        ax.set_xlim([time_vals.min(), time_vals.max()])
        ax.set_ylim(y_range)
        ax.grid(True, which='major', color='gray', linewidth=0.8, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
        ax.plot(pbltimes, pblheight, color='grey', linewidth=2, linestyle='-', label='PBL height')

        if i in [0, 2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        else:
            ax.tick_params(labelbottom=False)
                    
        ax.contour(ERA_data['time'], altitudes, ERA_data['t'].T, levels=temps, colors='gray', linewidths=1, linestyles='--', alpha=0.8)
        T.log(f"subplot {i}: ERA contour")

        add_isotherm_labels(ax, ERA_data, altitudes, alt_range=y_range, temps=temps, xlims=(time_vals.min(), time_vals.max()))
        T.log(f"subplot {i}: isotherm labels")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.8)
            spine.set_alpha(0.8)

    legend_handles = [
        Line2D([0], [0], color='green', lw=1.5, label='radiosonde launch') if radiosondes else None,
        Line2D([0], [0], color='grey', lw=2, linestyle='-', label='PBL height'),
        mpatches.Patch(facecolor='darkgreen', label='valid BASTA data') if BASTA_bars else None,
        mpatches.Patch(facecolor='limegreen', label=r'BASTA DFR correction ≥ 2$\sigma$ calibration') if DFR_bars else None,
        mpatches.Patch(facecolor='saddlebrown', label='valid MXPol data') if MXPol_bars else None,
        mpatches.Patch(facecolor='tab:orange', label='MXPol data quality issues') if LF_bars else None
    ]
    legend_handles = [h for h in legend_handles if h is not None]

    axs[2].legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.035, 1), fontsize=10, framealpha=0.3)
    T.log("legend")

    if pd.to_datetime(time_vals[0]).hour != 0:
        start_label = pd.to_datetime(time_vals[0]).strftime('%Y-%m-%d %H:%M')
    else:
        start_label = pd.to_datetime(time_vals[0]).strftime('%Y-%m-%d')

    if pd.to_datetime(time_vals[-1]).hour != 0:
        end_label = pd.to_datetime(time_vals[-1]).strftime('%Y-%m-%d %H:%M')
    else:
        end_label = pd.to_datetime(time_vals[-1]).strftime('%Y-%m-%d')

    plt.suptitle(f"Radar data overview for {start_label} to {end_label}", fontweight='bold', fontsize=18)
    T.log("title")

    plt.subplots_adjust(top=0.96)
    T.log("tight_layout")

    if savefig:
        plt.savefig(f"{str(figpath)}_ext.png", dpi=300, bbox_inches='tight', facecolor='w')
        T.log("savefig")
        plt.close()

    
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(nrows=4, ncols=1, figure=fig, hspace=0.3)
    T.log("figure created")

    axs = []
    for i in range(4):
        axs.append(fig.add_subplot(gs[i, 0]))

    for i, ax in enumerate(axs):
        T.log(f"subplot {i} start")
        dset = plots[i]
        timevals = mdates.date2num(dset.time.values)
        rangevals = dset.range.values

        if type_by_var[keys[i]] == 'velocity':
            im = ax.pcolormesh(timevals, rangevals, dset.values.T, cmap='seismic', norm=norm_vel, rasterized=True)
        else:
            im = ax.pcolormesh(timevals, rangevals, dset.values.T, cmap=pltConfig[keys[i]]['cmap'], vmin=pltConfig[keys[i]]['vmin'], vmax=pltConfig[keys[i]]['vmax'], rasterized=True)
        T.log(f"subplot {i}: pcolormesh")

        if i == 1: # add second axis + colorbar
            ax2 = ax.twinx()
            ax2.plot(time_vals, cloudtopDFRs, color='red', linewidth=1, alpha=0.8)
            ax2.set_ylim(0, 36)
            ax2.set_yticks(np.linspace(0, 36, 7))
            ax2.set_ylabel('BASTA DFR-based correction [dB]', color='red', labelpad=6)
            ax2.tick_params(axis='y', labelcolor='red')
            # ax2.spines['right'].set_position(('axes', 1.08))

            cbar_ax = ax.inset_axes([1.055, 0.1, 0.02, 0.8])
            cbar = plt.colorbar(im, cax=cbar_ax)
            # cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
            cbar.set_label(pltConfig[keys[i]]['label'])

            T.log(f"subplot {i}: BASTA correction based on DFR")
        else:
            # cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
            cbar_ax = ax.inset_axes([1.02, 0.1, 0.02, 0.8])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label(pltConfig[keys[i]]['label'])
        T.log(f"subplot {i}: colorbar")

        if i == 1 and (any([BASTA_bars, DFR_bars, MXPol_bars, LF_bars])): # add intervals 
            ax_bars1 = ax.inset_axes([0, -0.08, 1, 0.025])
            ax_bars2 = ax.inset_axes([0, -0.14, 1, 0.025])

            for a in (ax_bars1, ax_bars2):
                a.set_xlim(ax.get_xlim())
                a.set_ylim([0, 1])
                a.axis('off')

            if BASTA_bars:
                ax_bars1.imshow(mask_basta, aspect='auto', vmin=0, vmax=1,
                                cmap=mcolors.ListedColormap(['none', 'darkgreen']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if DFR_bars:
                ax_bars1.imshow(mask_dfr, aspect='auto', vmin=0, vmax=1,
                                cmap=mcolors.ListedColormap(['none', 'limegreen']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if MXPol_bars:
                ax_bars2.imshow(mask_mxpol, aspect='auto', vmin=0, vmax=1,
                                cmap=mcolors.ListedColormap(['none', 'saddlebrown']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if LF_bars:
                ax_bars2.imshow(mask_lostfreq, aspect='auto', vmin=0, vmax=1,
                                cmap=mcolors.ListedColormap(['none', 'tab:orange']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            ax_bars1.set_clip_on(False)
            ax_bars2.set_clip_on(False)
            ax.tick_params(axis='x', pad=15)
            T.log(f"subplot {i}: interval bars")

        if i == 2 and radiosondes:
            for rs in radiosondes:
                ax.axvline(x=datetime.strptime(rs, '%Y%m%d%H'), linewidth=1.5, color='green', alpha=0.8, zorder=10)
            T.log(f"subplot {i}: radiosonde lines")

        ax.set_xlim([time_vals.min(), time_vals.max()])
        ax.set_ylim(y_range)
        ax.grid(True, which='major', color='gray', linewidth=0.8, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
        ax.plot(pbltimes, pblheight, color='grey', linewidth=2, linestyle='-', label='PBL height')

        if i in [0, 2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        else:
            ax.tick_params(labelbottom=False)
                    
        ax.contour(ERA_data['time'], altitudes, ERA_data['t'].T, levels=temps, colors='gray', linewidths=1, linestyles='--', alpha=0.8)
        T.log(f"subplot {i}: ERA contour")

        add_isotherm_labels(ax, ERA_data, altitudes, alt_range=y_range, temps=temps, xlims=(time_vals.min(), time_vals.max()))
        T.log(f"subplot {i}: isotherm labels")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.8)
            spine.set_alpha(0.8)

    legend_handles = [
        Line2D([0], [0], color='green', lw=1.5, label='radiosonde launch') if radiosondes else None,
        Line2D([0], [0], color='grey', lw=2, linestyle='-', label='PBL height'),
        mpatches.Patch(facecolor='darkgreen', label='valid BASTA data') if BASTA_bars else None,
        mpatches.Patch(facecolor='limegreen', label=r'BASTA DFR correction ≥ 2$\sigma$ calibration') if DFR_bars else None,
        mpatches.Patch(facecolor='saddlebrown', label='valid MXPol data') if MXPol_bars else None,
        mpatches.Patch(facecolor='tab:orange', label='MXPol data quality issues') if LF_bars else None
    ]
    legend_handles = [h for h in legend_handles if h is not None]

    axs[2].legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.035, 1), fontsize=10, framealpha=0.3)
    T.log("legend")

    if pd.to_datetime(time_vals[0]).hour != 0:
        start_label = pd.to_datetime(time_vals[0]).strftime('%Y-%m-%d %H:%M')
    else:
        start_label = pd.to_datetime(time_vals[0]).strftime('%Y-%m-%d')

    if pd.to_datetime(time_vals[-1]).hour != 0:
        end_label = pd.to_datetime(time_vals[-1]).strftime('%Y-%m-%d %H:%M')
    else:
        end_label = pd.to_datetime(time_vals[-1]).strftime('%Y-%m-%d')

    plt.suptitle(f"Radar data overview for {start_label} to {end_label}", fontweight='bold', fontsize=18)
    T.log("title")

    plt.subplots_adjust(top=0.96)
    T.log("tight_layout")

    if savefig:
        plt.savefig(f"{str(figpath)}.png", dpi=300, bbox_inches='tight', facecolor='w')
        T.log("savefig")
        plt.close()

    del ds_subset, ds_mode, HALO_data, dset, fig, axs
    gc.collect()


def weekly_plot_paper(dataset, ERA_data, HALO_data, figpath, y_range=(0, 12e3), savefig=True):
    """like paper_plots' (but outdated): 4-panel figure (Z_Ka, vel_Ka, spectral width, LDR_Ka), 
    bulk-loading required variables from `dataset' upfront instead of taking pre-split ds_subset/ds_mode, 
    using numpy-based mask_to_intervals_numpy for interval detection instead of the xarray version.

    dataset (xr Dataset): combined radar dataset from zarr
    ERA_data (xr Dataset): ERA5 reanalysis dataset (temperature, altitude) for isotherm overlay
    HALO_data (xr Dataset): HALO boundary-layer classification dataset, used for PBL height line
    figpath (str): output path for the saved figure (used as-is, no suffix)
    y_range (tuple): altitude range (m) for the y-axis of all panels
    savefig (bool): if True, save and close; if False, leave figure open"""

    T = Timer()
    T.log("start")

    plotvars = ["Z_Ka", "Z_W", 'velocity_Ka', 'DFR_KaW']
    keys = ['Z', 'reflectivity', 'VELg', 'DFR_KaW']

    altitudes = (ERA_data['altitude'] - radar_altitude)
    norm_vel = mcolors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=2)
    time_vals = dataset['time'].values
    T.log("setup constants")

    vars_to_load = plotvars + ['Z_X', 'DFRcorrection_W']
    loaded = {var: dataset[var].load() for var in vars_to_load}
    T.log("bulk load dataset variables")

    # HALO_mask = HALO_data['aerosol_layer_mask_3min'].astype(bool).load()
    HALO_mask = ((HALO_data['bl_classification_3min'] > 0) & (HALO_data['height'] < 8000)).load()
    T.log("HALO mask load")

    pbltimes = HALO_data['time_3min'].values
    pblheight = HALO_data['height'].where(HALO_mask).max(dim='height').values
    jumps = np.abs(np.diff(pblheight, prepend=np.nan, append=np.nan))
    bad = (jumps[:-1] > 2000) | (jumps[1:] > 2000)
    pblheight[bad] = np.nan
    T.log("HALO alignment")

    dfr_vals = loaded['DFRcorrection_W'].values
    
    try:
        ZW_valid = loaded['Z_W'].notnull().any(dim='range')
        BASTA_intervals = mask_to_intervals_numpy(ZW_valid.values, time_vals)
        BASTA_intervals_merged = merge_close_intervals(BASTA_intervals)
        mask_basta = intervals_to_mask(BASTA_intervals_merged, time_vals)[None, :]
        BASTA_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping BASTA bars — {e}")
        BASTA_bars = False
    T.log("BASTA intervals")
    try:
        lowerDFR = calibrationvalues['BASTA'][0] - 2*calibrationvalues['BASTA'][1]
        upperDFR = calibrationvalues['BASTA'][0] + 2*calibrationvalues['BASTA'][1]
        DFR_valid = (((loaded['DFRcorrection_W'] < lowerDFR) | (loaded['DFRcorrection_W'] > upperDFR)) & ZW_valid)
        DFR_intervals = mask_to_intervals_numpy(DFR_valid.values, time_vals)
        DFR_intervals_merged = merge_close_intervals(DFR_intervals)
        mask_dfr = intervals_to_mask(DFR_intervals_merged, time_vals)[None, :]
        DFR_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping DFR bars — {e}")
        DFR_bars = False
    T.log("DFR outside range intervals")

    try:
        ZX_valid = loaded['DFR_XKa'].notnull().any(dim='range')
        MXPol_intervals = mask_to_intervals_numpy(ZX_valid.values, time_vals)
        MXPol_intervals_merged = merge_close_intervals(MXPol_intervals)
        mask_mxpol = intervals_to_mask(MXPol_intervals_merged, time_vals)[None, :]
        MXPol_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping MXPol bars — {e}")
        MXPol_bars = False
    T.log("MXPol intervals")
    try:
        MXPol_lost_frequencies = [
            (pd.to_datetime(startm, format='%Y%m%d-%H%M%S'),
            pd.to_datetime(endm, format='%Y%m%d-%H%M%S'))
            for startm, endm in MXPol_lost_frequency_times
        ]
        lostfreq_mask = pd.Series(False, index=pd.to_datetime(time_vals))
        for startx, endx in MXPol_lost_frequencies:
            lostfreq_mask.loc[(lostfreq_mask.index >= startx) & (lostfreq_mask.index <= endx)] = True
        lostfreq_intervals = mask_to_intervals_numpy(lostfreq_mask.values, time_vals)
        lostfreq_intervals_merged = merge_close_intervals(lostfreq_intervals)
        mask_lostfreq = intervals_to_mask(lostfreq_intervals_merged, time_vals)[None, :]
        LF_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping lost frequency bars — {e}")
        LF_bars = False
    T.log("lost frequency intervals")

    radiosondes = [d.strftime('%Y%m%d%H') for d in radiosonde_datetimes if time_vals[0] <= d <= time_vals[-1]]
    T.log("radiosondes")

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(nrows=4, ncols=1, figure=fig, hspace=0.3)
    T.log("figure created")

    axs = []
    for i in range(4):
        axs.append(fig.add_subplot(gs[i, 0]))

    for i, ax in enumerate(axs):
        T.log(f"subplot {i} start")
        dset = loaded[plotvars[i]]
        timevals = mdates.date2num(dset.time.values)
        rangevals = dset.range.values

        if type_by_var[keys[i]] == 'velocity':
            im = ax.pcolormesh(timevals, rangevals, dset.values.T, cmap='seismic', norm=norm_vel, rasterized=True)
        else:
            im = ax.pcolormesh(timevals, rangevals, dset.values.T, cmap=pltConfig[keys[i]]['cmap'], vmin=pltConfig[keys[i]]['vmin'], vmax=pltConfig[keys[i]]['vmax'], rasterized=True)
        T.log(f"subplot {i}: pcolormesh")

        if i == 1: # add second axis + colorbar
            ax2 = ax.twinx()
            ax2.plot(time_vals, dfr_vals, color='red', linewidth=1, alpha=0.8)
            ax2.set_ylim(0, 36)
            ax2.set_yticks(np.linspace(0, 36, 7))
            ax2.set_ylabel('BASTA DFR-based correction [dB]', color='red', labelpad=6)
            ax2.tick_params(axis='y', labelcolor='red')
            # ax2.spines['right'].set_position(('axes', 1.08))

            cbar_ax = ax.inset_axes([1.055, 0.1, 0.02, 0.8])
            cbar = plt.colorbar(im, cax=cbar_ax)
            # cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
            cbar.set_label(pltConfig[keys[i]]['label'])

            T.log(f"subplot {i}: BASTA correction based on DFR")
        else:
            # cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
            cbar_ax = ax.inset_axes([1.02, 0.1, 0.02, 0.8])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label(pltConfig[keys[i]]['label'])
        T.log(f"subplot {i}: colorbar")

        if i == 1 and (any([BASTA_bars, DFR_bars, MXPol_bars, LF_bars])): # add intervals 
            ax_bars1 = ax.inset_axes([0, -0.08, 1, 0.025])
            ax_bars2 = ax.inset_axes([0, -0.14, 1, 0.025])

            for a in (ax_bars1, ax_bars2):
                a.set_xlim(ax.get_xlim())
                a.set_ylim([0, 1])
                a.axis('off')

            if BASTA_bars:
                ax_bars1.imshow(mask_basta, aspect='auto',
                                cmap=mcolors.ListedColormap(['none', 'darkgreen']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if DFR_bars:
                ax_bars1.imshow(mask_dfr, aspect='auto',
                                cmap=mcolors.ListedColormap(['none', 'limegreen']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if MXPol_bars:
                ax_bars2.imshow(mask_mxpol, aspect='auto',
                                cmap=mcolors.ListedColormap(['none', 'saddlebrown']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if LF_bars:
                ax_bars2.imshow(mask_lostfreq, aspect='auto',
                                cmap=mcolors.ListedColormap(['none', 'tab:orange']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            ax_bars1.set_clip_on(False)
            ax_bars2.set_clip_on(False)
            ax.tick_params(axis='x', pad=15)
            T.log(f"subplot {i}: interval bars")

        if i == 2 and radiosondes:
            for rs in radiosondes:
                ax.axvline(x=datetime.strptime(rs, '%Y%m%d%H'), linewidth=1.5, color='green', alpha=0.8, zorder=10)
            T.log(f"subplot {i}: radiosonde lines")

        ax.set_xlim([time_vals.min(), time_vals.max()])
        ax.set_ylim(y_range)
        ax.grid(True, which='major', color='gray', linewidth=0.8, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
        ax.plot(pbltimes, pblheight, color='grey', linewidth=2, linestyle='-', label='PBL height')

        if i in [0, 2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        else:
            ax.tick_params(labelbottom=False)
                    
        ax.contour(ERA_data['time'], altitudes, ERA_data['t'].T, levels=temps, colors='gray', linewidths=1, linestyles='--', alpha=0.8)
        T.log(f"subplot {i}: ERA contour")

        add_isotherm_labels(ax, ERA_data, altitudes, alt_range=y_range, temps=temps, xlims=(time_vals.min(), time_vals.max()))
        T.log(f"subplot {i}: isotherm labels")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.8)
            spine.set_alpha(0.8)

    legend_handles = [
        Line2D([0], [0], color='green', lw=1.5, label='radiosonde launch') if radiosondes else None,
        Line2D([0], [0], color='grey', lw=2, linestyle='-', label='PBL height'),
        mpatches.Patch(facecolor='darkgreen', label='valid BASTA data') if BASTA_bars else None,
        mpatches.Patch(facecolor='limegreen', label=r'BASTA DFR correction ≥ 2$\sigma$ calibration') if DFR_bars else None,
        mpatches.Patch(facecolor='saddlebrown', label='valid MXPol data') if MXPol_bars else None,
        mpatches.Patch(facecolor='tab:orange', label='MXPol data quality issues') if LF_bars else None
    ]
    legend_handles = [h for h in legend_handles if h is not None]

    axs[2].legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.035, 1), fontsize=10, framealpha=0.3)
    T.log("legend")

    if pd.to_datetime(time_vals[0]).hour != 0:
        start_label = pd.to_datetime(time_vals[0]).strftime('%Y-%m-%d %H:%M')
    else:
        start_label = pd.to_datetime(time_vals[0]).strftime('%Y-%m-%d')

    if pd.to_datetime(time_vals[-1]).hour != 0:
        end_label = pd.to_datetime(time_vals[-1]).strftime('%Y-%m-%d %H:%M')
    else:
        end_label = pd.to_datetime(time_vals[-1]).strftime('%Y-%m-%d')

    plt.suptitle(f"Radar data overview for {start_label} to {end_label}", fontweight='bold', fontsize=18)
    T.log("title")

    plt.subplots_adjust(top=0.96)
    T.log("tight_layout")

    if savefig:
        plt.savefig(str(figpath), dpi=300, bbox_inches='tight', facecolor='w')
        T.log("savefig")
        plt.close()

    del dataset, HALO_data, loaded, fig, axs
    gc.collect()


def weekly_plot_paper_ext(dataset, ERA_data, HALO_data, figpath, y_range=(0, 12e3), savefig=True):
    """same as weekly_plot_paper but with additional panels for LDR_Ka and DFR_XKa (outdated)"""
    T = Timer()
    T.log("start")

    plotvars = ["Z_Ka", "Z_W", 'velocity_Ka', 'LDR_Ka', 'DFR_KaW', 'DFR_XKa']
    keys = ['Z', 'reflectivity', 'VELg', 'LDRg', 'DFR_KaW', 'DFR_XKa']

    altitudes = (ERA_data['altitude'] - radar_altitude)
    norm_vel = mcolors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=2)
    time_vals = dataset['time'].values
    T.log("setup constants")

    vars_to_load = plotvars + ['Z_X', 'DFRcorrection_W']
    loaded = {var: dataset[var].load() for var in vars_to_load}
    T.log("bulk load dataset variables")

    # HALO_mask = HALO_data['aerosol_layer_mask_3min'].astype(bool).load()
    HALO_mask = ((HALO_data['bl_classification_3min'] > 0) & (HALO_data['height'] < 8000)).load()
    T.log("HALO mask load")

    pbltimes = HALO_data['time_3min'].values
    pblheight = HALO_data['height'].where(HALO_mask).max(dim='height').values
    jumps = np.abs(np.diff(pblheight, prepend=np.nan, append=np.nan))
    bad = (jumps[:-1] > 2000) | (jumps[1:] > 2000)
    pblheight[bad] = np.nan
    T.log("HALO alignment")

    dfr_vals = loaded['DFRcorrection_W'].values
    try:
        ZW_valid = loaded['Z_W'].notnull().any(dim='range')
        BASTA_intervals = mask_to_intervals_numpy(ZW_valid.values, time_vals)
        BASTA_intervals_merged = merge_close_intervals(BASTA_intervals)
        mask_basta = intervals_to_mask(BASTA_intervals_merged, time_vals)[None, :]
        BASTA_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping BASTA bars — {e}")
        BASTA_bars = False
    T.log("BASTA intervals")
    try:
        lowerDFR = calibrationvalues['BASTA'][0] - 2*calibrationvalues['BASTA'][1]
        upperDFR = calibrationvalues['BASTA'][0] + 2*calibrationvalues['BASTA'][1]
        DFR_valid = (((loaded['DFRcorrection_W'] < lowerDFR) | (loaded['DFRcorrection_W'] > upperDFR)) & ZW_valid)
        DFR_intervals = mask_to_intervals_numpy(DFR_valid.values, time_vals)
        DFR_intervals_merged = merge_close_intervals(DFR_intervals)
        mask_dfr = intervals_to_mask(DFR_intervals_merged, time_vals)[None, :]
        DFR_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping DFR bars — {e}")
        DFR_bars = False
    T.log("DFR outside range intervals")

    try:
        ZX_valid = loaded['DFR_XKa'].notnull().any(dim='range')
        MXPol_intervals = mask_to_intervals_numpy(ZX_valid.values, time_vals)
        MXPol_intervals_merged = merge_close_intervals(MXPol_intervals)
        mask_mxpol = intervals_to_mask(MXPol_intervals_merged, time_vals)[None, :]
        MXPol_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping MXPol bars — {e}")
        MXPol_bars = False
    T.log("MXPol intervals")
    try:
        MXPol_lost_frequencies = [
            (pd.to_datetime(startm, format='%Y%m%d-%H%M%S'),
            pd.to_datetime(endm, format='%Y%m%d-%H%M%S'))
            for startm, endm in MXPol_lost_frequency_times
        ]
        lostfreq_mask = pd.Series(False, index=pd.to_datetime(time_vals))
        for startx, endx in MXPol_lost_frequencies:
            lostfreq_mask.loc[(lostfreq_mask.index >= startx) & (lostfreq_mask.index <= endx)] = True
        lostfreq_intervals = mask_to_intervals_numpy(lostfreq_mask.values, time_vals)
        lostfreq_intervals_merged = merge_close_intervals(lostfreq_intervals)
        mask_lostfreq = intervals_to_mask(lostfreq_intervals_merged, time_vals)[None, :]
        LF_bars = True
    except (KeyError, ValueError) as e:
        print(f"Warning: skipping lost frequency bars — {e}")
        LF_bars = False
    T.log("lost frequency intervals")

    radiosondes = [d.strftime('%Y%m%d%H') for d in radiosonde_datetimes if time_vals[0] <= d <= time_vals[-1]]
    T.log("radiosondes")
    
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(nrows=6, ncols=1, figure=fig, hspace=0.3)
    T.log("figure created")

    axs = []
    for i in range(6):
        axs.append(fig.add_subplot(gs[i, 0]))

    for i, ax in enumerate(axs):
        T.log(f"subplot {i} start")
        dset = loaded[plotvars[i]]
        timevals = mdates.date2num(dset.time.values)
        rangevals = dset.range.values

        if type_by_var[keys[i]] == 'velocity':
            im = ax.pcolormesh(timevals, rangevals, dset.values.T, cmap='seismic', norm=norm_vel, rasterized=True)
        else:
            im = ax.pcolormesh(timevals, rangevals, dset.values.T, cmap=pltConfig[keys[i]]['cmap'], vmin=pltConfig[keys[i]]['vmin'], vmax=pltConfig[keys[i]]['vmax'], rasterized=True)
        T.log(f"subplot {i}: pcolormesh")

        if i == 1:
            ax2 = ax.twinx()
            ax2.plot(time_vals, dfr_vals, color='red', linewidth=1, alpha=0.8)
            ax2.set_ylim(0, 36)
            ax2.set_yticks(np.linspace(0, 36, 7))
            ax2.set_ylabel('BASTA DFR-based correction [dB]', color='red', labelpad=6)
            ax2.tick_params(axis='y', labelcolor='red')
            
            cbar_ax = ax.inset_axes([1.055, 0.1, 0.02, 0.8])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label(pltConfig[keys[i]]['label'])
            T.log(f"subplot {i}: BASTA correction based on DFR")
        else:
            cbar_ax = ax.inset_axes([1.02, 0.1, 0.02, 0.8])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label(pltConfig[keys[i]]['label'])        
        T.log(f"subplot {i}: colorbar")

        if i == 1 and (any([BASTA_bars, DFR_bars, MXPol_bars, LF_bars])): # add intervals 
            ax_bars1 = ax.inset_axes([0, -0.08, 1, 0.025])
            ax_bars2 = ax.inset_axes([0, -0.14, 1, 0.025])

            for a in (ax_bars1, ax_bars2):
                a.set_xlim(ax.get_xlim())
                a.set_ylim([0, 1])
                a.axis('off')

            if BASTA_bars:
                ax_bars1.imshow(mask_basta, aspect='auto',
                                cmap=mcolors.ListedColormap(['none', 'darkgreen']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if DFR_bars:
                ax_bars1.imshow(mask_dfr, aspect='auto',
                                cmap=mcolors.ListedColormap(['none', 'limegreen']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if MXPol_bars:
                ax_bars2.imshow(mask_mxpol, aspect='auto',
                                cmap=mcolors.ListedColormap(['none', 'saddlebrown']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            if LF_bars:
                ax_bars2.imshow(mask_lostfreq, aspect='auto',
                                cmap=mcolors.ListedColormap(['none', 'tab:orange']),
                                extent=[time_vals[0], time_vals[-1], 0, 1])

            ax_bars1.set_clip_on(False)
            ax_bars2.set_clip_on(False)
            ax.tick_params(axis='x', pad=15)
            T.log(f"subplot {i}: interval bars")

        if i == 2 and radiosondes:
            for rs in radiosondes:
                ax.axvline(x=datetime.strptime(rs, '%Y%m%d%H'), linewidth=1.5, color='green', alpha=0.8, zorder=10)
            T.log(f"subplot {i}: radiosonde lines")

        ax.set_xlim([time_vals.min(), time_vals.max()])
        ax.set_ylim(y_range)
        ax.grid(True, which='major', color='gray', linewidth=0.8, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
        ax.plot(pbltimes, pblheight, color='grey', linewidth=2, linestyle='-', label='PBL height')

        if i in [0, 2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        else:
            ax.tick_params(labelbottom=False)
                    
        ax.contour(ERA_data['time'], altitudes, ERA_data['t'].T, levels=temps, colors='gray', linewidths=1, linestyles='--', alpha=0.8)
        T.log(f"subplot {i}: ERA contour")

        add_isotherm_labels(ax, ERA_data, altitudes, alt_range=y_range, temps=temps, xlims=(time_vals.min(), time_vals.max()))
        T.log(f"subplot {i}: isotherm labels")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.8)
            spine.set_alpha(0.8)

    legend_handles = [
        Line2D([0], [0], color='green', lw=1.5, label='radiosonde launch') if radiosondes else None,
        Line2D([0], [0], color='grey', lw=2, linestyle='-', label='PBL height'),
        mpatches.Patch(facecolor='darkgreen', label='valid BASTA data') if BASTA_bars else None,
        mpatches.Patch(facecolor='limegreen', label=r'BASTA DFR correction ≥ 2$\sigma$ calibration') if DFR_bars else None,
        mpatches.Patch(facecolor='saddlebrown', label='valid MXPol data') if MXPol_bars else None,
        mpatches.Patch(facecolor='tab:orange', label='MXPol data quality issues') if LF_bars else None
    ]
    legend_handles = [h for h in legend_handles if h is not None]

    axs[2].legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.035, 1), fontsize=10, framealpha=0.3)
    T.log("legend")

    if pd.to_datetime(time_vals[0]).hour != 0:
        start_label = pd.to_datetime(time_vals[0]).strftime('%Y-%m-%d %H:%M')
    else:
        start_label = pd.to_datetime(time_vals[0]).strftime('%Y-%m-%d')

    if pd.to_datetime(time_vals[-1]).hour != 0:
        end_label = pd.to_datetime(time_vals[-1]).strftime('%Y-%m-%d %H:%M')
    else:
        end_label = pd.to_datetime(time_vals[-1]).strftime('%Y-%m-%d')

    plt.suptitle(f"Radar data overview for {start_label} to {end_label}", fontweight='bold', fontsize=18)
    T.log("title")

    plt.subplots_adjust(top=0.96)
    T.log("tight_layout")

    if savefig:
        plt.savefig(str(figpath), dpi=300, bbox_inches='tight', facecolor='w')
        T.log("savefig")
        plt.close()

    del dataset, HALO_data, loaded, fig, axs
    gc.collect()


def plot_single_spectrum(ds, time, range, freq, dopplervar='sZh'):
    """plot single spectrum for given time and range"""
    # include nearest non-nan finding algo
    d = ds[dopplervar].sel(range=range, method='nearest').sel(time=time, method='nearest')
    plt.plot(ds.doppler.values, d.values)


def plot_weekly_HALO(full_dataset, HALO_data, ERA_data, start, end, figpath, savefig=True):
    """"6-panel weekly overview of radar (Z, velocity, spectral width) and HALO products (classification, 
    turbulence coupling, epsilon) with ERA5 isotherm overlay and radiosonde launch times
    full_dataset (xr Dataset): combined radar dataset from zarr containing all plotvars plus Z_X and DFRcorrection_W"""
    altitudes = (ERA_data['altitude'][1] - radar_altitude) / 1e3
    norm_vel = mcolors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=5)

    radiosondes_dt = [
        d for d in radiosonde_datetimes
        if start <= d <= end
    ]

    rdr = 'Ka' if end > datetime(2024, 11, 12) else 'W'
    plotvars = [f"Z_{rdr}", f"velocity_{rdr}", 'width_Ka', 'bl_classification_3min', 'turbulence_coupling_3min', 'epsilon_3min']
    keys = ['Z', 'VELg', 'RMSg', 'ABL', 'turbulence', 'EDR'] if rdr == 'Ka' else ['reflectivity', 'velocity', 'RMSg', 'ABL', 'turbulence', 'EDR']
    
    subset = full_dataset.sel(time=slice(start, end))
    ERA_subset = ERA_data.sel(time=slice(start, end))
    HALO_subset = HALO_data.sel(time_3min=slice(start, end))

    ERA_contour_T = ERA_subset['t'].T.values

    oro_intervals = mask_to_intervals(subset['orographic'])
    front_intervals = mask_to_intervals(subset['frontal'])

    fig, axs = plt.subplots(nrows=6, figsize=(20, 16))

    for i, ax in enumerate(axs):
        if i < 3:
            dataset = subset[plotvars[i]]
            timevar, rangevar = 'time', 'range'
        else:
            dataset = HALO_subset[plotvars[i]]
            timevar, rangevar = 'time_3min', 'height'

        config = pltConfig[keys[i]]

        if type_by_var[keys[i]] == 'velocity':
            im = ax.pcolormesh(dataset[timevar].values, dataset[rangevar].values, 
                              dataset.values.T, cmap=config['cmap'],
                              norm=norm_vel, rasterized=True)
        else:
            im = ax.pcolormesh(dataset[timevar].values, dataset[rangevar].values, 
                              dataset.values.T, cmap=config['cmap'], 
                              vmin=config['vmin'], vmax=config['vmax'],
                              rasterized=True)
        ax.plot(subset.time.values, subset.cloudtop.values, 'k', linewidth=0.8)
        ax.set_rasterization_zorder(0) 
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(pltConfig[keys[i]]['label'])
        
        ax.yaxis.set_major_formatter(FuncFormatter(meters_to_km))
        ax.set_xlim([start, end])
        ax.set_ylim([0, 12e3])
        ax.grid(True, alpha=0.5)
        ax.xaxis.grid(True, which='major', color='gray', linewidth=0.8)

        ax.contour(ERA_subset['time'].values, altitudes, ERA_contour_T, levels=temps, colors='gray', linewidths=1, linestyles='--', alpha=0.8)
        add_isotherm_labels(ax, ERA_subset, altitudes, alt_range=(0, 12e3), temps=temps, xlims=(start, end))

        if i in [0, 2, 4] and radiosondes_dt:
            for rs_dt in radiosondes_dt:
                ax.axvline(x=rs_dt, linewidth=1.5, color='green', alpha=0.8)

        time_vals = subset.time.values
        cloudtop_vals = subset.cloudtop.values

        # orographic vs. frontal cloud flags (above 2nd and bottom plot)
        if i in [1, 3, 5]:
            for starto, endo in oro_intervals:
                mask = (time_vals >= starto) & (time_vals <= endo)
                ax.fill_between(time_vals, cloudtop_vals, 12e3, 
                            where=mask, color='deepskyblue', alpha=0.15, zorder=0)
            for startf, endf in front_intervals:
                mask = (time_vals >= startf) & (time_vals <= endf)
                ax.fill_between(time_vals, cloudtop_vals, 12e3,
                            where=mask, color='fuchsia', alpha=0.15, zorder=0)
    
        ax.tick_params(axis='x', pad=30)

    plt.subplots_adjust(top=0.88)
    plt.suptitle(f"Radar data overview for {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}", fontweight='bold', fontsize=18)
    plt.tight_layout()
    
    if savefig:
        plt.savefig(figpath, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()
        print(f"Saved {figpath}")
    else:
        plt.show()
            