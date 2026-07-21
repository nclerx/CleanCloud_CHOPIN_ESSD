#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:03:42 2025

@author: clerx
"""
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from numpy.lib.stride_tricks import sliding_window_view
from functools import partial
from scipy.stats import linregress
from scipy.ndimage import minimum_filter, label
from src import re, os, glob, pyart, np, pd, xr, da, datetime
from src.constants_input import radar_altitude, MXPol_Zdr_corrections, cloudtop_params, cloudtop_criteria, scantimes_MXPol
from src.constants_input import metek_fieldmapping, metek_fieldmetadata, metek_attrmapping, common_range, common_range_spacing
from src.constants_input import variables_MIRA_refl, RHOHV_thres, SNRH_thres, variables, calibrationvalues, frequencies
from src.utils import is_within_limits, safe_reindex, read_ERA_data, check_resampling, find_MXPol_files, find_MIRA_files

total_height, selected_height = cloudtop_params

def preprocess(ds, chunk_size={'time': 500, 'range':100, 'doppler': 64}):
    """
    preprocess dataset before combining
    """
    if 'time' in ds.coords:
        _, unique_index = np.unique(ds.indexes['time'], return_index=True)
        ds = ds.isel(time=unique_index)
        if not ds.indexes['time'].is_monotonic_increasing:
            ds = ds.sortby('time')
    
    if 'doppler' in ds.coords and not ds.indexes['doppler'].is_monotonic_increasing:
        ds = ds.sortby('doppler')
    
    if chunk_size:
        # filter chunk_size to only include dimensions present in dataset
        ds = ds.chunk({dim: size for dim, size in chunk_size.items() if dim in ds.dims})
    
    return ds


def preprocess_duplicates(ds):
    """filter out duplicates"""
    if 'time' in ds.coords:
        _, unique_indices = np.unique(ds.time, return_index=True)
        ds = ds.isel(time=unique_indices)

        if not ds.indexes['time'].is_monotonic_increasing:
            ds = ds.sortby('time')

    return ds


def get_clean_mask_dask(data: xr.DataArray, min_points_range=3, min_time=30) -> xr.DataArray:
    """
    dask-friendly (lazy) version of get_clean_mask
    """    
    isval = data.notnull()

    # derive sampling step (in seconds)
    if not np.issubdtype(data.time.dtype, np.integer):
        dt = (data.time[1] - data.time[0]).values / np.timedelta64(1, 's')
        min_points_time = int(np.round(min_time / dt))
    else:
        min_points_time = min_time

    # moving window checks for validity
    valid_range = isval.rolling(range=min_points_range, center=True).sum() >= min_points_range
    valid_time = isval.rolling(time=min_points_time, center=True).sum() >= min_points_time

    clean_mask = isval & valid_range & valid_time

    return ~clean_mask


def get_clean_mask(data: xr.DataArray, min_points_range=3, min_time=30) -> np.ndarray:
    """
    input: 
    data : xr.DataArray of input reflectivity/power, shape (time, range)
    min_points_range : int, min. distance between consecutive valid range gates [m]
    min_time : int, min. time between consecutive valid time steps [s]

    returns a binary noise mask (0 = noise/invalid, 1 = valid data) with the same shape as 'data'
    """
    isval = data.notnull().values
    
    # time step in secondsds.t
    if not (type(data.time.values[0]) == int or type(data.time.values[0]) == np.int32):
        dt = int((data.time.values[1] - data.time.values[0])/1e9)
        min_points_time = int(np.round(min_time / dt))
    else:
        min_points_time = min_time
    
    # valid ranges
    consecutive_range_check = np.zeros_like(isval, dtype=bool)
    
    # (manually shifting the data, check if the window contains enough consecutive valid points) - binary dilation
    for offset in range(-min_points_range + 1, min_points_range):
        shifted = np.zeros_like(isval)
        
        if offset < 0:
            shifted[:, :offset] = isval[:, -offset:]
        elif offset > 0:
            shifted[:, offset:] = isval[:, :-offset]
        else:  # offset == 0
            shifted = isval.copy()
            
        if offset == -min_points_range + 1:
            window = shifted
        else:
            window = window & shifted
            
        if (offset - (-min_points_range + 1) + 1) >= min_points_range:
            valid = window
            consecutive_range_check = consecutive_range_check | valid
            if offset < min_points_range - 1:
                window = shifted
    
    # valid times
    consecutive_time_check = np.zeros_like(isval, dtype=bool)
    
    # binary dilation for time dimension
    for offset in range(-min_points_time + 1, min_points_time):
        # Create shifted arrays
        shifted = np.zeros_like(isval)
        
        if offset < 0:
            shifted[:offset, :] = isval[-offset:, :]
        elif offset > 0:
            shifted[offset:, :] = isval[:-offset, :]
        else:  # offset == 0
            shifted = isval.copy()
            
        if offset == -min_points_time + 1:
            window = shifted
        else:
            window = window & shifted
            
        if (offset - (-min_points_time + 1) + 1) >= min_points_time:
            valid = window
            consecutive_time_check = consecutive_time_check | valid
            if offset < min_points_time - 1:
                window = shifted
    
    # combine time and range masks
    clean_mask = isval & (~consecutive_range_check | ~consecutive_time_check)
    
    return ~clean_mask


def safe_openmfdataset(file_paths, **kwargs):
    """
    Open multiple .nc files but skip corrupted ones and generate lists of in-/
    excluded files. Used to open multiple MXPol files of which some are corrupted/empty.
    
    Args:
    - file_paths: List of file paths to the NetCDF files
    - **kwargs: Additional keyword arguments to pass to open_mfdataset
    
    Returns:
    - valid_files: list of valid/non-corrupted files
    - excluded_files: list of excluded files
    """
    valid_files = []
    excluded_files = []

    mfdataset_only_kwargs = {'combine', 'concat_dim', 'compat', 'join', 'preprocess'}
    open_dataset_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in mfdataset_only_kwargs}
    
    for file in file_paths:
        try:
            ds = xr.open_dataset(file, **open_dataset_kwargs)
            ds.close()
            # _ = ds.variables[list(ds.variables)[0]][0].values  # test if loading a small bit of data works
            valid_files.append(file)
            ds.close()
        except Exception as e:
            excluded_files.append(f"error while processing {file}: {e}")

    if valid_files:
        return valid_files, excluded_files
    else:
        return valid_files, excluded_files


def correct_gas_attenuation(att_fn, radar, Zvariable, frequency, time_start, time_end):
    """
    Correct the gas attenuation for reflectivity values using ERA5 attenuation data.

    Parameters:
        att_dir : Path to the directory containing ERA5 attenuation data (str)
        radar_altitude : Altitude of the radar (meters above sea level)
        radar : Radar dataset (xr)
        Zvariable: variable name of reflectivity to be used (str)
        time_start : pd.Datetime start time
        time_end : pd.Datetime end time
        frequency : radar frequency to be extracted from ERA file (can be 94.95/BASTA, 35.2/MIRA or 9.41) but is accepted if value is within +/- 1 GHz from input frequency

    Returns:
        Z_corrected : array of reflectivity corrected for gas attenuation (xr)
    """
    # load ERA5 attenuation data
    era_all = xr.open_dataset(att_fn)
    start = np.datetime64(time_start)
    end = np.datetime64(time_end)
    era = era_all.sel(time=slice(start, end))

    # only consider values that are at or above radar altitude
    height_above_radar_full = era['elevation'].values - radar_altitude
    mask_above = height_above_radar_full >= 0
    era = era.isel(elevation=mask_above)
    height_above_radar = era['elevation'] - radar_altitude

    # PIA at specified frequency
    if frequency not in era['frequency']:
        era_frequencies = era['frequency'].values
        differences = np.abs(era_frequencies - frequency)
        if np.min(differences) <= 1.0:
            closest_freq = era_frequencies[np.argmin(differences)]
            print(f"warning: using closest frequency {closest_freq} GHz instead of input {frequency} GHz")
            frequency = closest_freq
        else:
            raise ValueError(f'provided input frequency of {frequency} GHz not found in ERA data (no match within ±1 GHz)')

    PIA = era['PIA'].sel(frequency=frequency)
    # assign and rename coordinate
    PIA = PIA.assign_coords(height_above_radar=height_above_radar)
    PIA = PIA.swap_dims({'elevation': 'height_above_radar'})

    radar = radar.sel(time=slice(start, end))
    # interp PIA to radar time & height levels
    PIA_interp = PIA.interp(
        time=radar.time, height_above_radar=radar['range'],
        method='linear',
        kwargs={'bounds_error': False, 'fill_value': 'extrapolate'},
    )

    Z_corrected = radar[Zvariable] + PIA_interp
    return Z_corrected


def estimate_noise(power, n_fft=1, calc_stdv=True):
    """
    Get noise from radar data.

    Estimate noise parameters of a Doppler spectrum.
    Use the method of estimating the noise level in Doppler spectra outlined
    by "P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13, 808-811."

    input: 
        - power (array): Doppler spectrum in linear units
        - n_fft (int, optional): The number of spectral bins over which a moving 
            average has been taken. Corresponds to the **p** variable from 
            equation 9 of the article. The default value of 1 is appropriate 
            when no moving average has been applied to the spectrum.
        - calc_stdv (bool, optional): if true will the stdev of the noise estimates will be 
            returned as well (bool)
    returns:
        - lnoise: estimated noise level (np.array)
        - std deviation - unless specified as "False" (float)
    """
    power = np.asarray(power)
    if power.ndim == 2:
        power = power[None, :, :] # (1, range, doppler)
    
    n_time, n_range, n_doppler = power.shape
    lnoise = np.zeros((n_time, n_range))
    stdv = np.zeros((n_time, n_range))

    for it in range(n_time):
        for ir in range(n_range):
            P = np.sort(power[it, ir, :])
            N = len(P)

            sum_p = np.cumsum(P)
            sum_p2 = np.cumsum(P**2)

            k = np.arange(1, N+1)
            mean_p = sum_p / k
            var_p = (sum_p2 / k) - mean_p**2

            ratio = var_p / (mean_p**2 + 1e-12)

            idx = np.where (ratio < 1.0 / n_fft)[0]
            if len(idx) > 0:
                k0 = idx[-1]
                lnoise[it, ir] = mean_p[k0]
                stdv[it, ir] = np.sqrt(var_p[k0])
            else:
                lnoise[it, ir] = mean_p[-1]
                stdv[it, ir] = np.sqrt(var_p[-1])

    if lnoise.shape[0] == 1:
        return lnoise[0], stdv[0]
    else:
        return lnoise, stdv


def prepare_MIRAdata(filename, remove_noise=False, noise_method=None, stdev_threshold=None):
    """
    Parameters:
    -----------
    filename : str
        File path
    calc_LDR : bool, default False
        Whether to calculate LDR and return 
    remove_noise : bool, default False
        Whether to remove noise
    noise_method : str, optional
        Method for noise removal ('quantile' or 'hildebrandsekhon')
        Required if remove_noise=True
    stdev_threshold : float, optional
        Standard deviation threshold for hildebrandsekhon method
        Required if noise_method='hildebrandsekhon'

    returns MIRA-dataset which includes noise-filtered (if specified) spectral reflectivity & sLDR
    (all reflectivity values are converted to dB)
    """
    if remove_noise and noise_method is None:
        raise ValueError("noise_method must be specified when remove_noise=True")
    
    if remove_noise and noise_method == 'hildebrandsekhon' and stdev_threshold is None:
        raise ValueError("stdev_threshold must be specified for hildebrandsekhon method")
    
    if len(filename) == 1:
        ds = xr.open_dataset(filename)
    else:
        ds = xr.open_mfdataset(filename, combine='nested', concat_dim='time', chunks='auto')
    var_co = 'sZco'
    var_cx = 'sZcx'
    var_doppler = 'doppler'
    vars = [var_co, var_cx]
    
    ds[var_co], ds[var_cx], ds['sLDR'] = calculate_spectralZ(ds)

    if remove_noise:
        if noise_method == 'quantile':
            for var in vars:
                # calculate noise background (10th percentile at each range gate) and set to NaN if considered as noise
                noise_background = ds[var].quantile(0.1, dim=var_doppler)
                ds[var] = ds[var].where(ds[var_co] > noise_background)
        elif noise_method == 'hildebrandsekhon':
            for nt in range(len(ds['time'])):
                if nt >= ds['time'].shape[0]:
                    continue
                for var in vars:
                    spectra = ds[var].isel(time=nt).values
                    noise_background, stdev = estimate_noise(spectra)
                    
                    threshold = noise_background[:, None] + stdev_threshold * stdev[:, None]
                    new_spectrum = np.where(spectra > threshold, spectra - threshold, 0)
                    
                    ds[var][nt, :, :] = new_spectrum

    ds = ds.sortby(var_doppler)

    mask = get_clean_mask(ds['Z'])
    mask_xr = ds['Z'].copy(data=mask)
    ds = ds.assign(get_clean_mask=mask_xr)

    for var in ds.data_vars:
        if ds[var].dims == mask_xr.dims:
            ds[var] = ds[var].where(mask)
    
    # convert from linear to dB values
    for var in vars:
        ds[var] = 10 * np.log10(ds[var].where(ds[var] > 0))
    for var in ['Z', 'Zg', 'Zcx', 'LDRg']:
        ds[var] = 10 * np.log10(ds[var].where(ds[var] > 0))
    ds['sLDR'] = 10 * np.log10(ds['sLDR'].where(ds['sLDR'] > 0))

    return ds


def prepare_MXPoldata(filename, remove_noise=False, noise_method=None, stdev_threshold=None, Zdr_corr=True, variable='Zh'):
    """
    Parameters:
    -----------
    filename : str
        File path
    remove_noise : bool, default False
        Whether to remove noise
    noise_method : str, optional
        Method for noise removal ('quantile' or 'hildebrandsekhon')
        Required if remove_noise=True
    stdev_threshold : float, optional
        Standard deviation threshold for hildebrandsekhon method
        Required if noise_method='hildebrandsekhon'

    returns MXPol-dataset which includes noise-filtered (if specified) spectral reflectivity
    (all reflectivity values are converted to dB)
    """
    var_co = 'sPowH'
    var_cx = 'sPowV'
    var_doppler = 'nfft'
    vars = {var_co: 'sZH', var_cx: 'sZV'}
    
    if remove_noise and noise_method is None:
        raise ValueError("noise_method must be specified when remove_noise=True")
    
    if remove_noise and noise_method == 'hildebrandsekhon' and stdev_threshold is None:
        raise ValueError("stdev_threshold must be specified for hildebrandsekhon method")
    
    if len(filename) == 1:
        ds = xr.open_dataset(filename)
    else:
        ds = xr.open_mfdataset(filename, combine='nested', concat_dim='time', chunks='auto')
        if 'sweep' in ds.dims:
            numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
            ds = ds[numeric_vars].mean(dim="sweep")
        
        ds = ds.groupby('time').mean()

    if remove_noise:
        for key, var in vars.items():    
            if noise_method == 'quantile':
                # calculate noise background (10th percentile at each range gate) and set to NaN if considered as noise
                noise_background = ds[key].quantile(0.1, dim=var_doppler)
                filtered_data = ds[key].where(ds[key] > noise_background)
                
                ds[key] = filtered_data
                
            elif noise_method == 'hildebrandsekhon':
                data = ds[key].values.copy()
                for nt in range(len(ds['time'])):
                    spectra = ds[key].isel(time=nt).values
                    noise_background, stdev = estimate_noise(spectra, n_fft=ds.attrs.get('CAve_value', 1) * ds.attrs.get('PostAve_value', 1))
                    
                    threshold = noise_background[:, None] + stdev_threshold * stdev[:, None]
                    data[nt, :, :] = np.where(spectra > threshold, spectra, np.nan)

                ds[key] = xr.DataArray(
                    data,
                    dims=ds[key].dims,
                    coords=ds[key].coords,
                    attrs=ds[key].attrs
                )

    ds = ds.sortby(var_doppler)
    
    # convert from linear to dB values - now using the filtered data
    for key, variable in vars.items():
        ds[variable] = 10 * np.log10(ds[key] * ds['range']**2)

    if Zdr_corr:
        ds = Zdr_corr_xr(ds)
        
    return ds


def filter_radar(radar, SNR_thresh=-5):
    """apply SNR filter to (pyart) data"""
    gf = pyart.correct.GateFilter(radar)
    gf.exclude_below('SNRh_full', SNR_thresh)
    gf.exclude_below('SNRv_full', SNR_thresh)

    for v in radar.fields.keys() - {'SCC', 'sPowH', 'sPowV'}:
        if not v.endswith("_full"):
            v_full = v + '_full'
            if v_full in radar.fields.keys():
                V = radar.fields[v_full]['data'].copy()
                V =  np.ma.masked_where(gf.gate_excluded, V)
                radar.add_field_like(v, v, V, replace_existing=True)
    

def get_Zdr_corr(input_fn, corrections=MXPol_Zdr_corrections):
    """gets Zdr corrections from input file (imported from "constants input file" parsing date/time)"""
    corrections['datetime_str'] = [i[5:] for i in corrections['filename']]
    
    pattern = re.compile(r"XPOL-(\d{8})-(\d{6})")
    input_match = pattern.match(input_fn)
    
    if not input_match:
        raise ValueError(f"Could not match input filename {input_fn} to files with value for Zdr correction.")
    
    datetime_str = f"{input_match.group(1)}-{input_match.group(2)}"    
    input_time = pd.to_datetime(datetime_str, format='%Y%m%d-%H%M%S')

    time_diffs = (corrections['datetime'] - input_time).abs()
    closest_idx = time_diffs.idxmin()
        
    Zdr_corr = float(corrections.loc[closest_idx, 'correction_dB'])
    
    return Zdr_corr, corrections.loc[closest_idx, 'filename']


def Zdr_corr_xr(MXPol, corrections=MXPol_Zdr_corrections):
    """add MXPol Zdr-correction for xarray dataformat"""
    Zdr_corr_df = corrections.sort_values('datetime').copy()
    time_df = pd.DataFrame({'time': pd.to_datetime(MXPol['time'].values)})

    MXPol['Zdr_corrected'] = MXPol['Zdr'].copy()
    merged = pd.merge_asof(
        time_df,
        Zdr_corr_df[['datetime', 'correction_dB']].rename(columns={'datetime': 'time'}),
        on='time',
        direction='nearest'
    )
    MXPol['Zdr_corrected'] = MXPol['Zdr'] + merged['correction_dB'].values[:, np.newaxis]

    return MXPol


def Zdr_corr_pyart(radar, corrections=MXPol_Zdr_corrections):
    """create additional Zdr-corrected field in pyart radar object"""
    Zdr_corr_df = corrections.sort_values('datetime').copy()

    radar_time = pd.to_datetime(radar.time['data'], unit='s')
    radar.add_field('Zdr_corrected', radar.fields['Zdr'].copy())

    radar_df = pd.DataFrame({'time': radar_time})    
    merged = pd.merge_asof(
        radar_df.sort_values('time'),
        Zdr_corr_df[['datetime', 'correction_dB']].rename(columns={'datetime': 'time'}),
        on='time',
        direction='nearest'
    )
    
    for i, correction in enumerate(merged['correction_dB']):
        if not pd.isna(correction):
            radar.fields['Zdr_corrected']['data'][i] = radar.fields['Zdr']['data'][i] + correction

    return radar


def density_filter(data1, data2, lower_percentile=2.5, bins=100):
    """
    Filters out the lower 2.5% of data pairs based on a 2D density estimate.

    Parameters:
    - data1 (array-like): First dataset (e.g., Zh_all).
    - data2 (array-like): Second dataset (e.g., Zg_all).
    - lower_percentile (float): Lower cutoff for density (default: 2.5%).
    - bins (int or tuple): number of bins for the 2D histogram (default: 100).

    Returns:
    - filtered_data1 (ndarray): Filtered version of data1.
    - filtered_data2 (ndarray): Filtered version of data2.
    """

    is_xarray = isinstance(data1, xr.DataArray) and isinstance(data2, xr.DataArray)
    if is_xarray:
        data1_flat = data1.values.compute().flatten() if hasattr(data1.values, 'compute') else data1.values.flatten()
        data2_flat = data2.values.compute().flatten() if hasattr(data2.values, 'compute') else data2.values.flatten()
    else:
        data1_flat = np.asarray(data1).flatten()
        data2_flat = np.asarray(data2).flatten()

    if len(data1_flat) != len(data2_flat):
        raise ValueError("Both arrays must have the same length.")

    valid_mask = ~(np.isnan(data1_flat) | np.isnan(data2_flat))
    data1_valid = data1_flat[valid_mask]
    data2_valid = data2_flat[valid_mask]

    if len(data1_valid) == 0:
        raise ValueError("No valid data points after filtering for NaNs")

    hist, x_edges, y_edges = np.histogram2d(data1_valid, data2_valid, bins=bins)
    x_idx = np.clip(np.digitize(data1_valid, x_edges) - 1, 0, hist.shape[0] - 1)
    y_idx = np.clip(np.digitize(data2_valid, y_edges) - 1, 0, hist.shape[1] - 1)
    density = hist[x_idx, y_idx]

    lower_bound = np.percentile(density, lower_percentile)
    keep_mask = density > lower_bound

    if is_xarray:
        filtered_data1_flat = np.full_like(data1_flat, np.nan, dtype=np.float64)
        filtered_data2_flat = np.full_like(data2_flat, np.nan, dtype=np.float64)

        valid_indices = np.flatnonzero(valid_mask)
        kept_indices = valid_indices[keep_mask]
        filtered_data1_flat[kept_indices] = data1_flat[kept_indices]
        filtered_data2_flat[kept_indices] = data2_flat[kept_indices]

        filtered_data1 = xr.DataArray(filtered_data1_flat.reshape(data1.shape), coords=data1.coords, dims=data1.dims)
        filtered_data2 = xr.DataArray(filtered_data2_flat.reshape(data2.shape), coords=data2.coords, dims=data2.dims)
    else:
        filtered_data1 = data1_valid[keep_mask]
        filtered_data2 = data2_valid[keep_mask]

    return filtered_data1, filtered_data2


def range_selection(data1, data2, i, j):
    """
    Function to split data into 'inside' and 'outside' reflectivity range based 
    on lower/upper intercepts as determined iteratively based on .
    Input: 
    - data1 (array-like): first dataset - to be calibrated (or not), not containing NaNs
    - data2 (array-like): second dataset, not containing NaNs
    - i (float): lower boundary intercept
    - j (float): upper boundary intercept

    Returns:
    - data1_inside 
    - data1_outside
    - data2_inside
    - data2_outside
    """
    is_xarray = isinstance(data1, xr.DataArray) and isinstance(data2, xr.DataArray)

    if data1.shape != data2.shape:
        raise ValueError("Input arrays must have the same shape.")

    # create mask for data below lower and above higher boundary values
    mask_lower = (data2 < (-1 * data1 + i))
    mask_upper = (data2 > (-1 * data1 + j))

    # combine masks and apply combined mask to both datasets
    mask_outside = mask_lower | mask_upper
    mask_inside = ~mask_outside

    if is_xarray:
        data1_inside = data1.where(mask_inside)
        data1_outside = data1.where(mask_outside)
        data2_inside = data2.where(mask_inside)
        data2_outside = data2.where(mask_outside)
    else:
        data1_inside = data1[mask_inside]
        data1_outside = data1[mask_outside]
        data2_inside = data2[mask_inside]
        data2_outside = data2[mask_outside]

    return data1_inside, data1_outside, data2_inside, data2_outside


def determine_range_limits(data1, data2, n_iterations=100):
    """
    Function to determine optimal range limits for colocated radar reflectivity data.

    Input:
    - data1 (array-type, not containing any NaNs) - reflectivity in dBz
    - data2 (array-type, not containing any NaNs) - reflectivity in dBz
    - n_iterations (int): maximum number of iterations
    - rmse_threshold (float): the threshold below which RMSE is considered to have no significant change

    Returns:
    - results (dict): Dictionary storing results for all iterations.
    - best_result (tuple): (i, j, r**2, RMSE, data_perc) with the lowest RMSE.
    - actual_iterations (int): number of iterations before too little data left or limits too close together
    """
    # limits of accepted values for slope, r**2 and data percentage used
    slope_limits = [0.85, 1.15]
    r_sq_limits = [0.8, 1.0]
    min_data_perc = 0.6

    # get initial values for i and j based on min/max data pairs (excluding outliers)
    i = np.percentile(data1 + data2, 2.5)
    j = np.percentile(data1 + data2, 97.5)

    results = {}
    best_rmse = float('inf')
    true_best_rmse = float('inf')
    best_result = None
    true_best_result = None
    j_best = j
    phase = 'lowering_j'

    actual_iterations = 0
    for n_iter in range(n_iterations):
        actual_iterations += 1
        data1_inside, data1_outside, data2_inside, data2_outside = range_selection(data1, data2, i, j)
        data_perc = len(data1_inside) / (len(data1_inside) + len(data1_outside))

        if data_perc > min_data_perc:
            slope, intercept, r_val, _, _ = linregress(data1_inside, data2_inside)
            rsq = r_val**2

            # calculate RMSE
            data2_pred = slope * data1_inside + intercept
            rmse = np.sqrt(np.mean((data2_inside - data2_pred) ** 2))

            # store results of this iteration
            results[n_iter] = (i, j, slope, rsq, rmse, data_perc)
            
            # track *any* best RMSE (regardless of limits)
            if rmse < true_best_rmse:
                true_best_rmse = rmse
                true_best_result = (i, j, slope, rsq, rmse, data_perc, n_iter)
                j_best = j

            # update best result based on lowest RMSE
            if is_within_limits(slope, slope_limits) and is_within_limits(r_val**2, r_sq_limits):
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_result = (i, j, slope, rsq, rmse, data_perc, n_iter)
                    j_best = j

        # adjust i or j based on phase
        if phase == 'lowering_j':
            if (j - i) > 2:
                j -= 2
            else:
                phase = 'raising_i'
                j = j_best  # Keep track of the best 'j' before switching to 'raising_i'
        elif phase == 'raising_i':
            i += 2
            if (j - i) <= 2:
                break  # stop when (j - i) <= 2

    if best_result is None:
        print("Warning: No solution found meeting all criteria.")
        solution_found = -5
        valid_results = {k: v for k, v in results.items() if v[4] > min_data_perc}
        if not valid_results:
            print("No valid fits found that meet the minimum data percentage criterion.")
            return {}, None, None, actual_iterations, solution_found
        else:
            n_iter, (i, j, slope, rsq, rmse, data_perc) = min(valid_results.items(), key=lambda x: (-x[1][2], x[1][3])) # select best iteration based on maximum r_sq and minimum rmse
            best_result = (i, j, slope, rsq, rmse, data_perc, n_iter)

            return {}, best_result, None, actual_iterations, solution_found
    else:
        solution_found = 1

    return results, best_result, true_best_result, actual_iterations, solution_found


def calib_stats(df):
    """
    Calculates calibration statistics, returns dataframes of in- and excluded calibration 
    values.
    Input: dataframe containing calibration data (raw "pd.read_csv" from calibration output .txt-file)
    Output:
        priority_values: dataframe containing used calibration values
        excluded_values: dataframe containing excluded calibration values
        stats: dictionary containing main calibration statistics
    """
    df['t_start'] = pd.to_datetime(df['t_start'])
    df['t_end'] = pd.to_datetime(df['t_end'])
    df['date'] = df['t_start'].dt.date
    df['duration'] = df['t_end'] - df['t_start']

    no_solution = df[df['solution_found'] == -5].copy()  # Secondary priority
    with_solution = df[df['solution_found'] == 1].copy()  # Primary priority

    solution_values = with_solution['CC_dB']
    solution_mean = np.nanmean(solution_values)
    solution_std = solution_values.std()

    no_solution_values = no_solution['CC_dB']
    no_solution_mean = np.mean(no_solution_values)
    no_solution_std = no_solution_values.std()

    priority_values = with_solution.copy()
    excluded_values = pd.DataFrame()

    no_solution_dates = set(no_solution['date'])
    with_solution_dates = set(with_solution['date'])
    missing_dates = no_solution_dates - with_solution_dates

    no_solution_to_add = no_solution[no_solution['date'].isin(missing_dates)]
    priority_values = pd.concat([priority_values, no_solution_to_add], ignore_index=True)

    overlap_dates = no_solution_dates & with_solution_dates
    excluded_overlap = no_solution[no_solution['date'].isin(overlap_dates)].copy()
    excluded_values = pd.concat([excluded_values, excluded_overlap], ignore_index=True)

    duplicate_dates = priority_values['date'].value_counts()
    duplicate_dates = duplicate_dates[duplicate_dates > 1].index
    
    for date in duplicate_dates:
        date_entries = priority_values[priority_values['date'] == date]
        to_exclude = date_entries.iloc[1:]  # All other rows
        excluded_values = pd.concat([excluded_values, to_exclude], ignore_index=True)

    priority_values = priority_values.sort_values('rmse')  # sort by RMSE (lowest first)
    priority_values = priority_values.drop_duplicates(subset=['date'], keep='first')  # Keep first (lowest RMSE)
    priority_values = priority_values.sort_values('date').reset_index(drop=True)
    excluded_values = excluded_values.sort_values('date').reset_index(drop=True)

    priority_mean = np.mean(priority_values['CC_dB'])
    priority_std = priority_values['CC_dB'].std()

    no_solution_date_counts = no_solution['date'].value_counts()
    single_entries_no_solution = len(no_solution_date_counts[no_solution_date_counts == 1])
    multi_entries_no_solution = np.sum(no_solution_date_counts[no_solution_date_counts > 1])

    stats = {
        'included_mean': priority_mean,
        'included_std': priority_std,
        'solution_mean': solution_mean,
        'solution_std': solution_std,
        'no_solution_mean': no_solution_mean,
        'no_solution_std': no_solution_std,
        'all_mean': np.mean(df['CC_dB']),
        'all_std': df['CC_dB'].std(),
        'total_entries': len(df),
        'single_entries': len(df) - len(duplicate_dates),
        'single_entries_no_solution': single_entries_no_solution,
        'multi_entries_no_solution': multi_entries_no_solution,
        'priority_entries': len(priority_values),
        'excluded_entries': len(excluded_values)
    }
    
    return priority_values, excluded_values, stats


def calculate_spectralZ(ds):
    sZco = (ds['SPCco'] - ds['HSDco']) * ds['RadarConst'] * ((ds.range/5000)**2) * ds['SNRCorFaCo'] / ds['npw1']
    sZcx = (ds['SPCcx'] - ds['HSDcx']) * ds['RadarConst'] * ((ds.range/5000)**2) * ds['SNRCorFaCx'] / ds['npw2']
    sLDR = sZcx / sZco

    return sZco, sZcx, sLDR


def calculate_spectralZ_MXPol(ds):
    """calculate spectra reflectivity in dB"""
    var_co = 'sPowH'
    var_cx = 'sPowV'
    vars = {var_co: 'sZH', var_cx: 'sZV'}
    for key, variable in vars.items():
        ds[variable] = 10 * np.log10(ds[key] * ds['range']**2)

    return ds


def find_cloudtop(reflectivity, rres, total_height=total_height, selected_height=selected_height, offset_from_topcloud=0):
    """
    Use sliding window trick to find cloud top window based on consecutive non-NaN gates (vectorised).
    From this valid window, return only the top 'selected_gates'.
    
    Parameters:
        reflectivity : A 2D array (time x height) of reflectivity data.
        total_height (m): range interval of consecutive non-NaN values required to qualify
        selected_height (m): range interval to return from the top of that valid window
        offset_from_top (m): distance from the top of the clouds found
        rres (m): range resolution

    Returns:
        np.ndarray: A 2D array of shape (time, 2) with start and end index of selected_gates.
                    If no valid range found, returns (-1, -1) for that time step.
    """
    total_consecutive_gates = int(total_height / rres)
    selected_gates = int(selected_height / rres)
    offset_gates = int(offset_from_topcloud / rres)

    time_len, height_len = reflectivity.shape
    result = np.full((time_len, 2), -1, dtype=int)
    
    valid_mask = ~np.isnan(reflectivity) # Convert to binary mask (1 if valid, 0 if NaN)
    T, H = valid_mask.shape

    if H < total_consecutive_gates:
        return None
        
    # sliding windows along the height axis
    windows = sliding_window_view(valid_mask, window_shape=(total_consecutive_gates), axis=1)  # (T, H - W + 1, W)
    window_sum = windows.sum(axis=2)

    # find the highest valid window (from bottom of radar up = highest alt)
    valid_windows = window_sum == total_consecutive_gates
    result = np.full((T, 2), -1, dtype=int)

    last_valid_idx = np.argmax(valid_windows[:, ::-1], axis=1)
    has_valid = valid_windows.any(axis=1)
    actual_start_indices = (valid_windows.shape[1] - 1) - last_valid_idx

    # compute cloud top ranges
    cloud_start = actual_start_indices + total_consecutive_gates - offset_gates - selected_gates
    cloud_end = actual_start_indices + total_consecutive_gates - offset_gates - 1

    # apply only to valid rows
    result[has_valid, 0] = cloud_start[has_valid]
    result[has_valid, 1] = cloud_end[has_valid]

    return result
    

def find_cloudtop_xr(dataarray, total_height=total_height, rres=25, offset_from_topcloud=0, range_dim='range', time_dim='time'):
    """
    Vectorized cloud top finder for xarray.DataArray (time x height).

    Parameters:
        reflectivity : xr.DataArray (time x height)
        total_height : range interval of consecutive non-NaN values required to qualify (m)
        rres : range resolution (m)
        offset_from_topcloud : distance from the top of clouds (m)

    Returns:
        xr.DataArray of shape (time, 2) with start and end indices of cloud top.
        Returns (-1, -1) for time steps with no valid cloud top.
    """
    total_consecutive_gates = int(total_height / rres)
    offset_gates = int(offset_from_topcloud / rres)

    valid_mask = ~dataarray.isnull().values  # shape: (time, height)
    T, H = valid_mask.shape

    if H < total_consecutive_gates:
        return xr.DataArray(np.full((T, 2), -1, dtype=int),
                            coords={str(time_dim): dataarray[time_dim]}, dims=[time_dim, 'range_idx'])

    # sliding windows along the height axis
    windows = sliding_window_view(valid_mask, window_shape=total_consecutive_gates, axis=1)  # (T, H-W+1, W)
    window_sum = windows.sum(axis=2)
    valid_windows = window_sum == total_consecutive_gates

    # find the highest valid window (from top of atmosphere / highest alt)
    reversed_idx = np.argmax(valid_windows[:, ::-1], axis=1)
    has_valid = valid_windows.any(axis=1)
    highest_window_start = (valid_windows.shape[1] - 1) - reversed_idx  # start index of window

    cloudtop_idx = highest_window_start + total_consecutive_gates - 1 - offset_gates
    cloudtop = np.full(T, np.nan)
    cloudtop[has_valid] = dataarray[range_dim].values[cloudtop_idx[has_valid]]

    return xr.DataArray(cloudtop, coords={str(time_dim): dataarray[time_dim]}, dims=time_dim)


def cloudtop_height(mask, n_consec=6, dim='range'):
    """
    Find the highest 'range' coordinate below which there are at least
    n_consecutive True values along `dim`.
    Works for 1D or 2D DataArrays.
    """
    mask_np = mask.values  # convert to numpy array
    coords_range = mask[dim].values

    if mask_np.ndim == 1:
        labeled, num_features = label(mask_np)
        max_idx = -1
        for i in range(1, num_features + 1):
            inds = np.where(labeled == i)[0]
            if len(inds) >= n_consec:
                max_idx = max(max_idx, inds[-1])
        return coords_range[max_idx] if max_idx >= 0 else np.nan

    elif mask_np.ndim == 2:
        results = []
        for row in mask_np:
            labeled, num_features = label(row)
            max_idx = -1
            for i in range(1, num_features + 1):
                inds = np.where(labeled == i)[0]
                if len(inds) >= n_consec:
                    max_idx = max(max_idx, inds[-1])
            results.append(coords_range[max_idx] if max_idx >= 0 else np.nan)
        return xr.DataArray(results, dims=[mask.dims[0]], coords={mask.dims[0]: mask.coords[mask.dims[0]]})

    else:
        raise NotImplementedError("Only 1D or 2D arrays supported")

        
def find_continuous_cloudtop(dataarray, cloudtop, gap_height=200, rres=25):
    """
    Vectorized cloud top finder for xarray.DataArray (time x height).
    If a gap of gap_height is detected, returns the highest cloud top below the gap.

    Parameters:
        dataarray : xr.DataArray (time x height) - data array to check for gaps
        cloudtop : xr.DataArray (time,) - cloud top heights for each timestep
        gap_height : max. allowable gap size (m). If a gap >= this size exists, return cloud top below it
        rres : range resolution (m)
        offset_from_topcloud : distance from the top of clouds (m)

    Returns:
        xr.DataArray of shape (time,) with cloud top heights.
        Returns NaN for time steps with no valid cloud top.
    """
    gap_gates = int(gap_height / rres)

    valid_mask = ~dataarray.isnull().values  # shape: (time, height)
    T, H = valid_mask.shape

    continuous_cloudtop = cloudtop.copy()

    for t in range(T):
        if np.isnan(cloudtop.values[t]):
            continue
        
        # Find the height index corresponding to this cloud top
        cloudtop_height = cloudtop.values[t]
        cloudtop_idx = np.argmin(np.abs(dataarray.range.values - cloudtop_height))
        
        # Start checking from this cloud top downward
        search_from_idx = cloudtop_idx
        found_valid = False
        
        while search_from_idx > 0:
            # Check region below the current position
            region_below = valid_mask[t, :search_from_idx]
            
            # Look for a gap of gap_height below
            consecutive_invalid = 0
            gap_end_idx = None
            
            for i in range(len(region_below) - 1, -1, -1):  # Search downward
                if not region_below[i]:
                    consecutive_invalid += 1
                    if consecutive_invalid >= gap_gates:
                        # Found a significant gap
                        gap_end_idx = i
                        break
                else:
                    consecutive_invalid = 0
            
            if gap_end_idx is None:
                # No significant gap below, current cloud top is valid
                continuous_cloudtop.values[t] = dataarray.range.values[search_from_idx]
                found_valid = True
                break
            else:
                # Gap found, need to find next valid cloud top below the gap
                # Search for valid data below the gap
                if gap_end_idx == 0:
                    # Gap extends to the bottom
                    break
                
                # Find the next valid point below the gap
                next_valid_idx = None
                for i in range(gap_end_idx - 1, -1, -1):
                    if valid_mask[t, i]:
                        next_valid_idx = i
                        break
                
                if next_valid_idx is None:
                    # No valid data below the gap
                    break
                else:
                    # Continue searching from below the gap
                    search_from_idx = next_valid_idx
        
        if not found_valid:
            continuous_cloudtop.values[t] = np.nan

    return continuous_cloudtop


def cloudbase_height(mask, n_consec=6, dim='range'):
    """
    Find the lowest 'range' coordinate above which there are at least
    n_consecutive True values along `dim`.
    Works for 1D or 2D DataArrays.
    """
    mask_np = mask.values  # convert to numpy array
    coords_range = mask[dim].values

    if mask_np.ndim == 1:
        labeled, num_features = label(mask_np)
        min_idx = -1
        for i in range(1, num_features + 1):
            inds = np.where(labeled == i)[0]
            if len(inds) >= n_consec:
                min_idx = inds[0] if min_idx == -1 else min(min_idx, inds[0])
        return coords_range[min_idx] if min_idx >= 0 else np.nan

    elif mask_np.ndim == 2:
        results = []
        for row in mask_np:
            labeled, num_features = label(row)
            min_idx = -1
            for i in range(1, num_features + 1):
                inds = np.where(labeled == i)[0]
                if len(inds) >= n_consec:
                    min_idx = inds[0] if min_idx == -1 else min(min_idx, inds[0])
            results.append(coords_range[min_idx] if min_idx >= 0 else np.nan)
        return xr.DataArray(results, dims=[mask.dims[0]], coords={mask.dims[0]: mask.coords[mask.dims[0]]})

    else:
        raise NotImplementedError("Only 1D or 2D arrays supported")

        
def find_continuous_cloudbase(dataarray, cloudbase, gap_height=200, rres=25, n_consec=4):
    """
    Vectorized cloud base finder for xarray.DataArray (time x height).
    Accepts the cloud base immediately if it starts a run of at least n_consec valid gates.
    If that run is too short, searches upward past any gap for the next qualifying run.

    Parameters:
        dataarray : xr.DataArray (time x height) - data array to check for gaps
        cloudbase : xr.DataArray (time,) - cloud base heights for each timestep
        gap_height : max. allowable gap size (m) to skip over when searching upward
        rres : range resolution (m)
        n_consec : minimum number of consecutive valid gates required to accept a base

    Returns:
        xr.DataArray of shape (time,) with cloud base heights.
        Returns NaN for time steps with no valid cloud base.
    """
    gap_gates = int(gap_height / rres)

    valid_mask = ~dataarray.isnull().values  # shape: (time, height)
    T, H = valid_mask.shape

    continuous_cloudbase = cloudbase.copy()

    for t in range(T):
        if np.isnan(cloudbase.values[t]):
            continue

        cloudbase_height_val = cloudbase.values[t]
        cloudbase_idx = np.argmin(np.abs(dataarray.range.values - cloudbase_height_val))

        search_from_idx = cloudbase_idx
        found_valid = False

        while search_from_idx < H:
            # check if the run starting here is already long enough — accept immediately if so
            run_len = 0
            for i in range(search_from_idx, H):
                if valid_mask[t, i]:
                    run_len += 1
                    if run_len >= n_consec:
                        continuous_cloudbase.values[t] = dataarray.range.values[search_from_idx]
                        found_valid = True
                        break
                else:
                    break
            if found_valid:
                break

            # run too short — search upward past this gap for the next valid point
            next_valid_idx = None
            for i in range(search_from_idx + 1, H):
                if valid_mask[t, i]:
                    next_valid_idx = i
                    break
            if next_valid_idx is None:
                break
            search_from_idx = next_valid_idx

        if not found_valid:
            continuous_cloudbase.values[t] = np.nan

    return continuous_cloudbase


def cloudtop_dask(dataarray, total_height=None, offset_from_topcloud=0, gap_height=None, rres=25):
    """
    Fully Dask-compatible cloudtop finder.
    Supports optional total_height, offset_from_topcloud, and gap_height.
    Returns a lazy xr.DataArray (time,) that can be rolled and smoothed before compute.

    Parameters:
        dataarray : xr.DataArray (time x range)
        total_height : minimum cloud depth to qualify (m)
        offset_from_topcloud : optional offset from top cloud (m)
        gap_height : optional gap height to enforce continuous cloudtop (m)
        rres : range resolution (m)
    """
    valid_mask = ~dataarray.isnull()

    total_gates = int(total_height / rres) if total_height is not None else 1
    offset_gates = int(offset_from_topcloud / rres)
    gap_gates = int(gap_height / rres) if gap_height is not None else None

    ranges = dataarray.range.values

    def _cloudtop_lazy(valid_mask_block, total_gates, offset_gates, gap_gates, ranges):
        """
        Lazy block function for Dask map_blocks.
        valid_mask_block: bool array (time, range)
        Returns: cloudtop heights (time,)
        """
        valid_mask_block = np.asarray(valid_mask_block)
        T, H = valid_mask_block.shape
        cloudtop = np.full(T, np.nan, dtype=float)

        for t in range(T):
            row = valid_mask_block[t]
            if not row.any():                
                continue

            # Optional: apply total_height constraint from top
            if total_gates > 1:
                candidates = []
                for idx in np.where(row)[0]:
                    start = max(0, idx - total_gates + 1)
                    if row[start:idx+1].all():
                        candidates.append(idx)
                if not candidates:
                    continue
                top_idx = candidates[-1]  # highest cloud satisfying total_height
            else:
                top_idx = np.where(row)[0][-1]

            # Apply offset from top cloud
            top_idx = max(0, top_idx - offset_gates)

            # Apply gap_height constraint
            if gap_gates is not None:
                search_idx = top_idx
                while search_idx > 0:
                    below = row[:search_idx]
                    gaps = np.where(~below)[0]
                    if len(gaps) == 0:
                        break  # no gaps

                    gap_sizes = np.diff(np.concatenate([[-1], gaps, [search_idx]])) - 1
                    max_gap_idx = np.argmax(gap_sizes)

                    if gap_sizes[max_gap_idx] >= gap_gates:
                        if max_gap_idx >= len(gaps):
                            next_gap_idx = gaps[-1]
                        else:
                            next_gap_idx = gaps[max_gap_idx]

                        prev_idx = search_idx
                        search_idx = max(0, next_gap_idx - 1)
                        if search_idx >= prev_idx:
                            break
                    else:
                        break

                top_idx = search_idx
            cloudtop[t] = ranges[top_idx]

        return cloudtop

    cloudtop_da = da.map_blocks(
        _cloudtop_lazy,
        valid_mask.data,
        total_gates,
        offset_gates,
        gap_gates,
        ranges,
        dtype=float,
        chunks=(dataarray.chunks[0],),
        drop_axis=1
    )

    return xr.DataArray(cloudtop_da, coords={'time': dataarray.time}, dims='time')


def mask_range(window, index_ranges):
    """
    Create mask corresponding to the top of the cloud defined with the function 'find_topcloud'

    Parameters:
        window : A 2D array (time x height) of reflectivity data.
        index_ranges (min_idx,max_idx): Index of the min and max of the cloud top

    Returns:
        Mask, min and max index of valid range.
    """
    time_len, range_len = window.shape
    index_ranges = np.array(index_ranges)  # shape (time, 2)

    valid = (index_ranges[:, 0] >= 0) & (index_ranges[:, 1] >= 0)
    index_ranges[~valid] = [0, -1] # Set invalid ranges to (0, -1) 

    # Create min and max index arrays
    min_idx = index_ranges[:, 0][:, np.newaxis]  # shape (time, 1)
    max_idx = index_ranges[:, 1][:, np.newaxis]  

    range_grid = np.arange(range_len)[np.newaxis, :]
    range_grid = np.tile(range_grid, (time_len, 1))

    # Create mask: True where range index is within min and max
    mask = (range_grid >= min_idx) & (range_grid <= max_idx)  # shape (time, range)

    return mask, min_idx, max_idx


def preprocess_BASTA(ds, variable='reflectivity', drop_variables=False):
    """filter/preprocess BASTA-data to remove noise & background mask"""
    if drop_variables:
        keep_vars = variables['BASTA']
        drop_vars = [v for v in ds.data_vars if v not in keep_vars]
        ds = ds.drop_vars(drop_vars)

    ds[variable] = ds[variable].where((ds[variable] >= -60) & (ds[variable] <= 100), np.nan)
    mask = get_clean_mask(ds[variable])
    mask_xr = ds[variable].copy(data=mask)    
    ds = ds.assign(clean_mask=mask_xr)
    ds[variable] = ds[variable].where(ds['clean_mask']).where(ds['background_mask'])
    ds['velocity'] = ds['velocity'].where(ds['clean_mask']).where(ds['background_mask'])
    ds = ds.sortby('time')

    return ds


def preprocess_MIRA(ds, drop_spectral=True, variable='Z'):
    """filter/preprocess MIRA-data to remove noise & background mask"""
    if drop_spectral:
        ds = ds.max(dim='doppler', keep_attrs=True) # remove unnecessary dimensions but preserve metadata
        
    mask = get_clean_mask(ds[variable])
    mask_xr = ds[variable].copy(data=mask)
    ds = ds.assign(clean_mask=mask_xr)
    
    for var in ds.data_vars: # apply mask to all variables
        if ds[var].dims == mask_xr.dims:
            ds[var] = ds[var].where(mask)
    
    for var in variables_MIRA_refl:
        if var in ds.data_vars:
            ds[var] = 10 * np.log10(ds[var].where(ds[var] > 0))

    return ds
    

def preprocess_MXPol(ds, Zdr_corr=True, drop_spectral=True, SNR_thresh=-9999.):
    """
    filter/preprocess MXPol-data to remove noise & create background mask
    """
    # remove unnecessary dimensions
    if 'sweep' in ds.dims:
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        ds = ds[numeric_vars].mean(dim="sweep")
        # ds = ds.mean(dim='sweep')
    if drop_spectral == True and 'nfft' in ds.dims:
        ds = ds.mean(dim='nfft')

    _, index = np.unique(ds['time'], return_index=True)
    ds = ds.isel(time=index) # remove duplicates in time axis

    # mask = get_clean_mask(ds[variable])
    mask_SNR = (ds['SNRh'] > SNR_thresh) & (ds['SNRv'] > SNR_thresh)
        
    # apply SNR & structural mask to all data variables with time/range dims
    for var in ds.data_vars:
        if set(['time', 'range']).issubset(ds[var].dims):
            # ds[var] = ds[var].where(combined_mask)
            ds[var] = ds[var].where(mask_SNR)

    ds['clean_mask'] = mask_SNR
    if Zdr_corr:
        ds = Zdr_corr_xr(ds)
        
    return ds


def DFR_offset(BASTA, MIRA, checks, window_size='15min'):
    """
    calculate DFR correction from cloud top reflectivity values
    input
        BASTA: data array of BASTA reflectivity (common range, common time)
        - calibrated & corrected for gas attenuation
        MIRA: data array of MIRA reflectivity (common range, common time)
        window_size: time window (str) to create pd.date_range of time windows for which to calculate the DFR offset
    """
    check1, check2, check3 = checks
    total_height, selected_height = cloudtop_params
    rres = float(BASTA.range[1] - BASTA.range[0])
    t_int = int(pd.to_timedelta((BASTA.time[1] - BASTA.time[0]).values).total_seconds())

    t_min = pd.to_datetime(BASTA.time.min().values)
    t_max = pd.to_datetime(BASTA.time.max().values)
    time_windows = pd.date_range(t_min, t_max, freq=window_size)

    records = []

    for t_start in time_windows:
        t_end = t_start + pd.Timedelta(window_size)
        print(f"Calculating window {t_start} to {t_end}")
        
        BASTA_window = BASTA.sel(time=slice(t_start, t_end))
        MIRA_window = MIRA.sel(time=slice(t_start, t_end))

        BASTA_nan = BASTA_window.isnull()
        MIRA_nan = MIRA_window.isnull()

        valid_mask = ~(BASTA_nan['reflectivity_corrected'] | MIRA_nan['Z'])
        valid_mask = valid_mask.compute()

        BASTA_window = BASTA_window.where(valid_mask, drop=True)
        MIRA_window = MIRA_window.where(valid_mask, drop=True)
        
        cloudtop = find_cloudtop(BASTA_window['reflectivity_corrected'], total_height, selected_height, rres)
        if cloudtop is None:
            print(f"Window {t_start} error: height dimension is smaller than required window")
            continue
        mask_cloudtop, min_idx, max_idx = mask_range(BASTA_window['reflectivity_corrected'], cloudtop)
        mask_cloudtop = xr.DataArray(mask_cloudtop, 
                                     dims=('time', 'range'), 
                                     coords={'time': BASTA_window.coords['time'], 
                                             'range': BASTA_window.coords['range'],})

        BASTA_top = BASTA_window.where(mask_cloudtop)
        MIRA_top = MIRA_window.where(mask_cloudtop)

        # check 1: number of valid points
        num_valid = int(MIRA_top['Z'].count(dim=['time', 'range']).compute().item())
        if check1:
            if num_valid < cloudtop_criteria['min_valid_points']:
                print(f"Window {t_start} failed check 1: not enough valid points ({num_valid})")
                continue
        
        # check 2: correlation across reflectivity measurements
        BASTA_top_flat = BASTA_top['reflectivity_corrected'].values.flatten()
        MIRA_top_flat = MIRA_top['Z'].values.flatten()
        finite_mask = np.isfinite(MIRA_top_flat) & np.isfinite(BASTA_top_flat)
        if finite_mask.sum() < 10:
            print(f"Window {t_start} contains too little data")
            continue

        corr = np.corrcoef(MIRA_top_flat[finite_mask], BASTA_top_flat[finite_mask])[0, 1]
        if check2:
            if np.isnan(corr) or corr < cloudtop_criteria['min_correlation']:
                print(f"Window {t_start} failed check 2: correlation too low ({corr:.2f})")
                continue

        # check 3: spatio-temporal DFR variability
        min_values = cloudtop_criteria['min_values']
        DFR = MIRA_top['Z'] - BASTA_top['reflectivity_corrected']
        rolling_window_time = int(cloudtop_criteria['time_window'] / t_int)
        rolling_window_range = int(cloudtop_criteria['range_window'] / rres)

        DFR_var = DFR.rolling(time = rolling_window_time, 
                              range = rolling_window_range, 
                              center = True, 
                              min_periods = min_values).var(skipna=True)

        count_above_thres = int((DFR_var > cloudtop_criteria['DFR_variance_threshold']).sum())
        total_count = int(DFR_var.count())
        if total_count == 0:
            ratio = 0
        else:
            ratio = count_above_thres / total_count
        if check3:
            if ratio < 0.5: 
                print(f"Window {t_start} failed check 3: too high DFR variance")
                continue

        record = {
            'time': t_start,
            'num_valid_points': num_valid,
            'correlation': corr,
            'mean_DFR': np.nanmean(DFR.values), 
            'ratio_DFR_var': ratio
        }
        max_idx = max_idx.squeeze()
        if max_idx.size > 0:
            record['cloudtop_ranges'] = BASTA_top.range[max_idx].values
            record['cloudtop_times'] = BASTA_top.time.values
            record['DFR'] = DFR.values

        records.append(record)

    # construct DataFrame from results
    output_df = pd.DataFrame(records)

    return output_df


def DFRcorrection(DFRdata, BASTA):
    """correct BASTA reflectivity using cloudtop DFR"""
    times = pd.to_datetime(BASTA.time.values)

    if not np.issubdtype(DFRdata['time'].dtype, np.datetime64):
        DFRdata['time'] = pd.to_datetime(DFRdata['time'])
    DFRdata = DFRdata.sort_values('time').reset_index(drop=True)

    indices = np.searchsorted(DFRdata['time'].values, times, side='right') - 1
    valid = (indices >= 0) & (indices < len(DFRdata))

    corrections = np.full(times.shape, np.nan, dtype=float)
    corrections[valid] = DFRdata['mean_DFR'].iloc[indices[valid]].values

    BASTA['DFR_correction'] = (('time', corrections))
    if 'reflectivity_attn_corrected' in BASTA.data_vars:
        varname = 'reflectivity_attn_corrected'
    else:
        varname = 'reflectivity_corrected'
    BASTA['reflectivity_attn_DFR_corrected'] = BASTA[varname] + BASTA['DFR_correction']

    return BASTA


def DFRcorrection_array(DFRdata, timevalues):
    """as DFRcorrection but for arrays"""
    times = pd.to_datetime(timevalues)

    if not np.issubdtype(DFRdata['time'].dtype, np.datetime64):
        DFRdata['time'] = pd.to_datetime(DFRdata['time'])
    DFRdata = DFRdata.sort_values('time').reset_index(drop=True)

    indices = np.searchsorted(DFRdata['time'].values, times, side='right') - 1
    valid = (indices >= 0) & (indices < len(DFRdata))

    corrections = np.full(times.shape, np.nan, dtype=float)
    corrections[valid] = DFRdata['mean_DFR'].iloc[indices[valid]].values

    return (('time', corrections))


def add_DFRcorrection_xr(DFRdata, timevalues):
    times = pd.to_datetime(timevalues)

    if not np.issubdtype(DFRdata['time'].dtype, np.datetime64):
        DFRdata['time'] = pd.to_datetime(DFRdata['time'])
    DFRdata = DFRdata.sort_values('time').reset_index(drop=True)

    indices = np.searchsorted(DFRdata['time'].values, times, side='right') - 1
    valid = (indices >= 0) & (indices < len(DFRdata))

    corrections = np.full(times.shape, np.nan, dtype=float)
    corrections[valid] = DFRdata['mean_DFR'].iloc[indices[valid]].values

    return xr.DataArray(corrections, dims=('time',), coords={'time': times}, name='DFR_correction')


def find_nan_intervals(data, tm, length='10min'):
    """find continuous time intervals longer than "length" where all data is NaN
    data: masked np.ndarray (2D array time x something e.g. (n_times, n_heights))
    tm: array-like (time array of length n_times)
    returns list of (start_time, end_time) tuples
    """
    nan_mask = np.all(data.mask, axis=1)  

    # identify start and end of NaN gaps
    diffs = np.diff(nan_mask.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]

    # handle case where a NaN gap starts at the first sample or ends at the last sample
    if nan_mask[0]:
        starts = np.r_[0, starts]
    if nan_mask[-1]:
        ends = np.r_[ends, len(nan_mask) - 1]

    intervals = [(pd.to_datetime(tm[start]), pd.to_datetime(tm[end])) for start, end in zip(starts, ends)]
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = interval
            if curr_start <= prev_end + pd.Timedelta(np.diff(pd.to_datetime(tm[:2]))[0]):
                merged[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged.append(interval)

    min_duration = pd.Timedelta(length)
    filtered = [(start, end) for start, end in merged if (end - start) >= min_duration]

    return filtered


def handle_nfft_mismatch(ds):
    """make sure spectral dimension matches required output size"""
    if 'nfft' in ds.dims:
        nfft_size = ds.dims['nfft']
        if nfft_size == 128:
            spectral_vars = ['sPowH', 'sPowV']

            for var in spectral_vars:
                if var in ds.data_vars:
                    old_nfft = np.arange(128)
                    new_nfft = np.linspace(0, 127, 256)

                    # interpolate along fft dimension
                    ds[var] = ds[var].interp(nfft=new_nfft, method='linear')

            if 'sVel' in ds.data_vars:
                vel_old = ds['sVel'].values
                vel_min, vel_max = vel_old[0], vel_old[-1]
                vel_spacing = (vel_max - vel_min) / 127

                vel_new = np.linspace(vel_min, vel_max, 256)
                ds['sVel'] = xr.DataArray(vel_new, dims=['nfft'], coords={'nfft': np.arange(256)})
            
            ds = ds.assign_coords(nfft=np.arange(256))

    return ds


def drop_nfft(ds):
    """drop spectral variables that have nfft dimension"""
    if 'nfft' in ds.dims:
        spectral_vars = ['sPowH', 'sPowV', 'sVel']
        vars_to_drop = [var for var in spectral_vars if var in ds.data_vars]
        
        if vars_to_drop:
            ds = ds.drop_vars(vars_to_drop)
    
    return ds


def map_metek_to_pyart(ds):
    """map Metek MIRA attributes to PyART standard attributes"""
    mapped_fields = {}
    ngates = len(ds.range)
    nrays = len(ds.time)

    for metek_name in ds.data_vars:
        var = ds[metek_name]

        if len(var.dims) != 2:
            continue

        pyart_name = metek_fieldmapping.get(metek_name, metek_name)

        data = var.values
        if data.shape != (nrays, ngates):
            data = data.T
        
        field_dict = {
            'data': data,
            '_FillValue': var.attrs.get('_FillValue', -999.0),
        }

        if pyart_name in metek_fieldmetadata:
            field_dict.update(metek_fieldmetadata[pyart_name])
        else:
            field_dict.update({
                'units': var.attrs.get('units', ''),
                'long_name': var.attrs.get('long_name', metek_name),
            })
        mapped_fields[pyart_name] = field_dict

    return mapped_fields


def znc_to_pyart(fn):
    """convert Metek MIRA .znc file to PyART radar object"""
    ds = preprocess_MIRA(xr.open_dataset(fn), variable='Zg')
    nrays = len(ds.time)
    ngates = len(ds.range)
    
    if 'ppi' in fn:
        radar = pyart.testing.make_empty_ppi_radar(nrays, ngates, nsweeps=1)

    elif 'rhi' in fn:
        radar = pyart.testing.make_empty_rhi_radar(nrays, ngates, nsweeps=1)

    radar.metadata['instrument_name'] = ds.attrs.get('instrument_name', 'MIRA MBR5')
    radar.sweep_start_ray_index['data'][0] = 0
    radar.sweep_end_ray_index['data'][0] = nrays - 1

    if 'range' in ds.coords:
        radar.range['data'] = ds.range.values
        for attr_name, attr_value in ds.range.attrs.items():
            radar.range[attr_name] = attr_value
    elif 'height' in ds.coords:
        radar.range['data'] = ds.height.values
        for attr_name, attr_value in ds.height.attrs.items():
            radar.range[attr_name] = attr_value

    # correct for north angle offset manually
    radar.azimuth['data'] = (ds.azi + ds.northangle).where(~np.isnan(ds.azi)).values
    radar.elevation['data'] = ds.elv.values
    
    if 'time' in ds.coords: # convert to seconds since start of file
        time_vals = pd.to_datetime(ds.time.values, unit='s')
        time_vals = time_vals.append(pd.Index([time_vals[-1] + (time_vals[-1] - time_vals[-2])]))
        start_time = time_vals[0]
        radar.time['data'] = (time_vals - start_time) / pd.Timedelta(seconds=1)
        radar.time['units'] = f'seconds since {start_time}'

    # get/set PyART variable names
    mapped_fields = map_metek_to_pyart(ds)
    radar.fields.update(mapped_fields)

    # set location 
    if 'lat' in ds.attrs or 'Latitude' in ds.attrs:
        lat = ds.attrs.get('lat', float(ds.attrs.get('Latitude', 0.0)[:-1]))
        radar.latitude['data'] = np.array([lat])
    if 'lon' in ds.attrs or 'Longitude' in ds.attrs:
        lon = ds.attrs.get('lon', float(ds.attrs.get('Longitude', 0.0)[:-1]))
        radar.longitude['data'] = np.array([lon])
    if 'alt' in ds.attrs or 'Altitude' in ds.attrs:
        alt = ds.attrs.get('alt', float(ds.attrs.get('Altitude', 0.0)[:-1]))
        radar.altitude['data'] = np.array([alt])
    
    for metek_key, pyart_key in metek_attrmapping.items():
        if metek_key in ds.attrs:
            val = ds.attrs[metek_key]
            
            if pyart_key in ['latitude', 'longitude', 'altitude']: # already handled above
                continue

            elif pyart_key in ['frequency', 'wavelength', 'pulse_width', 'prt',
                               'prt_mode', 'nyquist_velocity', 'unambiguous_range',
                               'radar_beam_width_h', 'radar_beam_width_v',
                               'radar_antenna_gain', 'signal_processor_version',
                               'radar_constant', 'receiver_gain', 'noise_floor',
                               'range_resolution']:
                if pyart_key not in radar.instrument_parameters:
                    radar.instrument_parameters[pyart_key] = {}
                radar.instrument_parameters[pyart_key]['data'] = np.array([val])
            
            else:
                radar.metadata[pyart_key] = str(val)

    return radar


def noise_threshold(z, min_bins=3, axis=1, threshold=-35):
    """
    remove detections that don't have at least 'min_bins' contiguous bins
    z: input reflectivity values 2D ndarray (range x doppler)
    axis: along which to check contiguity (1 = Doppler bins)
    threshold: minimum dBZ to consider a bin

    filtered_input: 2D ndarray of filtered data
    """
    mask = z > threshold
    filtered_mask = np.zeros_like(mask, dtype=bool)

    for idx in range(mask.shape[1-axis]):
        if axis == 1:
            row = mask[idx, :]
        else:
            row = mask[:, idx]
        labelled, num_features = label(row)
        keep = np.zeros_like(row, dtype=bool)
        for f in range(1, num_features+1):
            region = (labelled == f)
            if region.sum() >= min_bins:
                keep[region] = True
        if axis == 1:
            filtered_mask[idx, :] = keep
        else:
            filtered_mask[:, idx] = keep

    filtered_z = np.where(filtered_mask, z, np.nan)
    return filtered_z


def add_Kdp(radar, thres_RHO=RHOHV_thres, thres_SNR=SNRH_thres, fieldnames=['Rhohv', 'SNRh']):
    """"add Kdp to radar object (MXPol)"""
    rhohv = fieldnames[0]
    snrh = fieldnames[1]
    radar.fields['Psidp']['data'].mask = False
    radar.fields['Psidp']['data'].mask[(radar.fields[rhohv]['data'].data < thres_RHO) & 
                                    (radar.fields[snrh]['data'].data < thres_SNR)] = True
    
    radar.fields['Kdp'] = pyart.retrieve.kdp_schneebeli(radar, psidp_field='Psidp', band='X', 
                                                        prefilter_psidp=True, 
                                                        filter_opt={'rhohv_field': rhohv})[0]
    

def load_zenithdata(date, dirs, DFRs, att_fn, BASTAmode, rolling_window='10min', resample_time=True, target_time=None, resample_range=True, interp_method = 'nearest',
                    useMXPol=True, useMIRA=True, useBASTA=True, peakTree=True, ERA=True, drop_spectral=True, start_time=None, end_time=None):
    """ loads all radar data + does processing
    MXPol L1 data (Zdr-corrected, including hydrometeor classification)
    """
    year, month, day = date.year, date.month, date.day

    tmin = np.datetime64(datetime(year, month, day))
    tmax = tmin + np.timedelta64(1, 'D')

    window_start = pd.Timestamp(start_time) if start_time is not None else pd.Timestamp(tmin)
    window_end = pd.Timestamp(end_time) if end_time is not None else pd.Timestamp(tmax)

    pad = pd.Timedelta(rolling_window)
    pad_start = max(pd.Timestamp(tmin), window_start - pad)
    pad_end = min(pd.Timestamp(tmax), window_end + pad)

    if resample_time:
        target_freq = '5s' if target_time is None else target_time
    else:
        target_freq = None
    # target_freq = '5s' if resample_time and target_time is None else target_time
    full_time = pd.date_range(window_start, window_end, freq=target_freq) # start to end of available data resampled to every 5 seconds
    datasets = {}

    def _trim(ds, name, start=pad_start, end=pad_end):
        if ds is not None or 'time' not in ds.dims:
            return ds
        try:
            return ds.sel(time=slice(start, end))
        except Exception as e:
            print(f"Could not trim {name} to window {start} - {end}: {e}")
            return ds

    if useMXPol:
        # print(f'loading MXPol data for {year}-{month:02}-{day:02}')
        MXPol_filedir = os.path.join(dirs['MXPol'], f"{year}/{month:02}/{day:02}")
        MXPol_files = find_MXPol_files(sorted(glob.glob(f"{MXPol_filedir}/*_Zdr.nc")), pad_start, pad_end)
        if MXPol_files:
            try:
                MXPol_files, _ = safe_openmfdataset(MXPol_files)
                prprocess = drop_nfft if drop_spectral is True else None
                MXPol = preprocess_MXPol(xr.open_mfdataset(MXPol_files, 
                                                                combine='nested',
                                                                concat_dim='time',
                                                                chunks={'time': 3600},
                                                                preprocess=prprocess),
                                                                Zdr_corr=True, 
                                                                drop_spectral=drop_spectral,
                                                                SNR_thresh=-9999.)
                
                if not drop_spectral and not all(var in MXPol.data_vars for var in ['sZH', 'sZV']):
                    MXPol = calculate_spectralZ_MXPol(MXPol)
                if MXPol is not None:
                    if resample_time and check_resampling(MXPol, 'time', target_coords=full_time, tolerance='0.5s'):
                        MXPol = safe_reindex(MXPol, 'time', full_time, tolerance='0.5s', name='MXPol', method=interp_method)
                    if resample_range and check_resampling(MXPol, 'range', target_coords=common_range, tolerance=common_range_spacing):
                        MXPol = safe_reindex(MXPol, 'range', common_range, tolerance=common_range_spacing, name='MXPol', method=interp_method)
                    datasets['MXPol'] = MXPol
            except Exception as e:
                print(f"Error loading MXPol data for {date}: {e}")

    if useMIRA:
        # MIRA_files = glob.glob(f"{dirs['MIRA']}/{year}{month:02}{day:02}_small*") # for quick plotting
        MIRA_files = find_MIRA_files(sorted(glob.glob(f"{dirs['MIRA']}/{year}/{month:02}/{day:02}/{year}{month:02}{day:02}*_merged.nc")), pad_start, pad_end)
        if MIRA_files:
            # print(f'loading MIRA data for {year}-{month:02}-{day:02}')
            try:
                # MIRA = xr.open_dataset(MIRA_files[0])
                MIRA = preprocess_MIRA(xr.open_mfdataset(MIRA_files), drop_spectral=drop_spectral)
                if resample_time and check_resampling(MIRA, 'time', target_coords=full_time, tolerance='2.5s'):
                    MIRA = safe_reindex(MIRA, 'time', full_time, tolerance='2.5s', name='MIRA', method=interp_method)
                if resample_range and check_resampling(MIRA, 'range', target_coords=common_range, tolerance=common_range_spacing):
                    MIRA = safe_reindex(MIRA, 'range', common_range, tolerance=common_range_spacing, method=interp_method, name='MIRA')
                MIRA_dt = pd.to_timedelta((MIRA.time[1] - MIRA.time[0]).values).total_seconds()
                rolling_MIRA = int(pd.Timedelta(rolling_window).total_seconds() / MIRA_dt)
                if rolling_MIRA < 1:
                    rolling_MIRA = 1
                minperiods_MIRA = max(1, int(round(0.2 * rolling_MIRA)))
                if not drop_spectral and not all(var in MIRA.data_vars for var in ['sZco', 'sZcx', 'sLDR']): # check if spectral variables are already there or not
                    MIRA['sZco'], MIRA['sZcx'], MIRA['sLDR'] = calculate_spectralZ(MIRA)
                if not 'cloud_top_height' in MIRA.data_vars:
                    MIRA['cloud_top_height'] = cloudtop_height(MIRA['Z'] > -35, n_consec=4)
                MIRA['cloud_top_height_mean'] = MIRA['cloud_top_height'].rolling(time=rolling_MIRA, center=True, min_periods=minperiods_MIRA).mean()
                MIRA['cloud_top_height_continuous'] = find_continuous_cloudtop(MIRA['Z'], MIRA['cloud_top_height'], gap_height=cloudtop_params[1], rres=(MIRA.range.values[1] - MIRA.range.values[0]))
                MIRA['cloud_top_height_continuous_mean'] = MIRA['cloud_top_height_continuous'].rolling(time=rolling_MIRA, center=True, min_periods=int(60/MIRA_dt)).mean()
                datasets['MIRA'] = MIRA
            except Exception as e:
                print(f"Error loading MIRA data for {date}: {e}")

    if useBASTA:
        BASTA_files = sorted(glob.glob(os.path.join(os.path.dirname(dirs['BASTA']), BASTAmode, f"BASTA_L1_{BASTAmode}_{year}{month:02}{day:02}*.nc")))
        if BASTA_files:
            # print(f'loading BASTA data for {year}-{month:02}-{day:02}')
            try:
                BASTA = preprocess_BASTA(xr.open_mfdataset(BASTA_files, combine='by_coords'), drop_variables=False)
                if BASTA is not None:
                    if resample_time and check_resampling(BASTA, 'time', target_coords=full_time, tolerance='1.5s'):
                        BASTA = safe_reindex(BASTA, 'time', full_time, tolerance='1.5s', method=interp_method, name='BASTA')
                    if resample_range and check_resampling(BASTA, 'range', target_coords=common_range, tolerance=common_range_spacing):
                        BASTA = safe_reindex(BASTA, 'range', common_range, tolerance=common_range_spacing, method=interp_method, name='BASTA')
                    # remap to nearest second
                    BASTA['time_original'] = BASTA.time
                    BASTA = BASTA.assign_coords(time=BASTA.time.dt.round('s'))
                    BASTA_dt = min(np.diff(BASTA.time.values)).astype('timedelta64[s]').astype(float)
                    rolling_BASTA = int(pd.Timedelta(rolling_window).total_seconds() / BASTA_dt)
                    if rolling_BASTA < 1:
                        rolling_BASTA = 1
                    minperiods_BASTA = max(1, int(round(0.2 * rolling_BASTA)))
                    BASTA['reflectivity_attn_corrected'] = xr.DataArray(correct_gas_attenuation(att_fn, BASTA, 'reflectivity', frequencies['BASTA'], tmin, tmax).values, dims=BASTA['reflectivity'].dims, 
                                                                        coords={dim: BASTA[dim] for dim in BASTA['reflectivity'].dims})
                    BASTA = DFRcorrection(DFRs, BASTA)
                    BASTA['reflectivity_attn_DFR_corrected'] = BASTA['reflectivity_attn_DFR_corrected'] + calibrationvalues['BASTA'][0]
                    BASTA['cloud_top_height'] = cloudtop_height(BASTA['reflectivity_attn_DFR_corrected'] > -35, n_consec=4)
                    BASTA['cloud_top_height_mean'] = BASTA['cloud_top_height'].rolling(time=rolling_BASTA, center=True, min_periods=minperiods_BASTA).mean()
                    BASTA['cloud_top_height_continuous'] = find_continuous_cloudtop(BASTA['reflectivity'], BASTA['cloud_top_height'], gap_height=cloudtop_params[1], rres=(BASTA.range.values[1] - BASTA.range.values[0]))
                    BASTA['cloud_top_height_continuous_mean'] = BASTA['cloud_top_height_continuous'].rolling(time=rolling_BASTA, center=True, min_periods=minperiods_BASTA).mean()
                    datasets['BASTA'] = BASTA
            except Exception as e:
                print(f"Error loading BASTA data for {date}: {e}")

    if peakTree:
        peaktree_fn = glob.glob(f"{dirs['peaktree']}/{year}{month:02}{day:02}_daily*.nc4")
        if len(peaktree_fn) >= 1:
            # print(f'loading peaktree data for {year}-{month:02}-{day:02}')
            try:
                peaktree = xr.open_dataset((peaktree_fn)[0])
                if 'MIRA' in datasets:
                    MIRA_ds = datasets['MIRA']
                    peaktree = peaktree.reindex(time=MIRA_ds.time, method='nearest')
                    peaktree = peaktree.reindex(range=MIRA_ds.range, method='nearest')
                    peaktree['cloud_top_height'] = cloudtop_height(peaktree['Z'].sel(nodes=0) > -35, n_consec=4)
                    peaktree['cloud_top_height_mean'] = peaktree['cloud_top_height'].rolling(time=rolling_MIRA, center=True, min_periods=minperiods_MIRA).mean()
                    peaktree['cloud_top_height_continuous'] = find_continuous_cloudtop(peaktree['Z'].sel(nodes=0), peaktree['cloud_top_height'], gap_height=cloudtop_params[1], rres=(peaktree.range.values[1] - peaktree.range.values[0]))
                    peaktree['cloud_top_height_continuous_mean'] = peaktree['cloud_top_height_continuous'].rolling(time=rolling_MIRA, center=True, min_periods=minperiods_MIRA).mean()
                    if peaktree is not None:
                        datasets['peaktree'] = peaktree
            except Exception as e:
                print(f"Error loading peaktree data for {date}: {e}")

    if ERA:
        # print(f'loading ERA data for {year}-{month:02}-{day:02}')
        try:
            datasets['ERA'] = xr.concat([read_ERA_data(date, dirs['ERA']), read_ERA_data(date + pd.Timedelta(days=1), dirs['ERA'])], dim='time').sortby('time')
        except Exception as e:
            ds1 = read_ERA_data(date, dirs['ERA']).reset_coords('altitude')
            ds2 = read_ERA_data(date + pd.Timedelta(days=1), dirs['ERA']).reset_coords('altitude')
            datasets['ERA'] = xr.concat([ds1, ds2], dim='time').sortby('time')
            print(f"workaround for ERA concatenation, error: {e}")

    return datasets


def correct_MXPol_file(fname, scantype):
    """check whether MXPol file contains enough data and add Zdr correction"""
    if type(fname) == str:
        radar = pyart.io.read_cfradial(fname)
    else:
        radar = fname

    t_scan = len(radar.time['data'])
    if t_scan > scantimes_MXPol[scantype]*2: # if more than 2x longer than expected scan time, likely a corrupted file
        print(f"too many data points (time is {int(t_scan)} s instead of {scantimes_MXPol[scantype]}) - corrupted file")
        return
    # if t_scan < scantimes_MXPol[scantype]*0.75: 
    #     print(f"too little data points (time is {int(t_scan)} s instead of {scantimes_MXPol[scantype]}) - corrupted file")

    filter_radar(radar)
    # add Kdp if not already present
    if not 'Kdp' in radar.fields.keys():
        try:
            add_Kdp(radar, fieldnames=['Rhohv_full', 'SNRh_full'])
        except Exception as e:
            print(f"  Error while adding Kdp: {e}, trying with filtered Rhohv & SNRh...")
            try:
                add_Kdp(radar, fieldnames=['Rhohv', 'SNRh'])
            except Exception as e:
                print(f"  Adding Kdp didn't work: {e}, skipping...")

    zdr_corr = get_Zdr_corr(os.path.basename(fname))[0]
    zdr = radar.fields['Zdr']['data'].copy() - zdr_corr
    radar.add_field_like('Zdr_corrected', 'Zdr_corrected', zdr, replace_existing=True)

    return radar
    