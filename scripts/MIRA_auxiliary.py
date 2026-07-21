#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:00:27 2025

@author: clerx
"""
__all__ = ['meters_to_km', 'find_nearest', 'interpolate_and_smooth', 
           'rolling_mean', 'generate_date_range', 'preprocess', 'preprocess_mmclx', 
           'get_files_ranges', 'get_files_ranges_mmclx']

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as signal

from scipy import ndimage
from datetime import datetime, timedelta


def meters_to_km(y, _):
    return f"{y / 1000:.1f} km"


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def interpolate_and_smooth(data, max_gap=5, window_length=11, polyorder=3):
    """
    Interpolate NaN values and then apply Savitzky-Golay filter
    
    Parameters:
    -----------
    data : array-like
        Input data with potential NaN values
    max_gap : int, optional
        Maximum number of consecutive NaNs to interpolate
    window_length : int, optional
        Window length for Savitzky-Golay filter
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filter
    
    Returns:
    --------
    smoothed_data : numpy.ndarray
        Smoothed data with interpolated NaNs
    """
    # Create a copy of the data to avoid modifying the original
    data_copy = data.copy()
    
    # Method 1: Interpolation before smoothing
    # Find NaN indices
    nan_mask = np.isnan(data_copy)
    
    # Interpolate NaNs
    # First, try linear interpolation for small gaps
    data_interp = data_copy.copy()
    for i in range(len(data_copy)):
        if nan_mask[i]:
            # Look for nearest non-NaN values
            left = max(0, i - max_gap)
            right = min(len(data_copy), i + max_gap)
            
            # Find non-NaN values around the current point
            non_nan_indices = np.where(~nan_mask[left:right])[0]
            
            if len(non_nan_indices) > 0:
                # Interpolate using nearby non-NaN values
                valid_indices = non_nan_indices + left
                valid_values = data_copy[valid_indices]
                
                # Weighted interpolation based on distance
                weights = 1 / (np.abs(valid_indices - i) + 1)
                data_interp[i] = np.average(valid_values, weights=weights)
    
    # Apply Savitzky-Golay filter to interpolated data
    smoothed_data = signal.savgol_filter(data_interp, 
                                          window_length=window_length, 
                                          polyorder=polyorder)
    
    return smoothed_data


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


def preprocess(ds, chunk_size={'time': 500, 'range':100, 'doppler': 64}):
    """
    Preprocess dataset before combining
    """
    if 'time' in ds.coords:
        _, unique_index = np.unique(ds.indexes['time'], return_index=True)
        ds = ds.isel(time=unique_index)
        if not ds.indexes['time'].is_monotonic_increasing:
            ds = ds.sortby('time')
    
    if 'doppler' in ds.coords and not ds.indexes['doppler'].is_monotonic_increasing:
        ds = ds.sortby('doppler')
        
    if chunk_size:
        ds = ds.chunk(chunk_size)
        
    return ds


def preprocess_mmclx(ds, chunk_size={'time': 500, 'range':100}):
    """
    Preprocess dataset before combining
    """
    if 'time' in ds.coords:
        _, unique_index = np.unique(ds.indexes['time'], return_index=True)
        ds = ds.isel(time=unique_index)
        if not ds.indexes['time'].is_monotonic_increasing:
            ds = ds.sortby('time')
    
    if 'doppler' in ds.coords and not ds.indexes['doppler'].is_monotonic_increasing:
        ds = ds.sortby('doppler')
        
    if chunk_size:
        ds = ds.chunk(chunk_size)
        
    return ds


def get_files_ranges(day, month, year, source):
    """
    Returns a nested dictionary containing file paths of all files with zenith measurements
    and their corresponding start and end times for each hour of the specified date.
    """
    data = {}

    for hour in range(24):
        pth = f"{year}{month:02d}{day:02d}_{hour:02d}*.znc"
        full_path = os.path.join(source, pth)

        # Find all files for this hour and filter out PPI's and RHI's
        hourfiles = glob.glob(full_path)
        valid_files = sorted(f for f in hourfiles if ('.ppi' not in f) and ('.rhi' not in f) and ('.azisectorscan' not in f))

        # Processing files using Dask
        time_ranges = []
        for f in valid_files:
            # Open the dataset and extract the time range
            ds = xr.open_dataset(f, chunks={}, decode_times=False)  # Lazy loading
            
            # Filter out time periods where ds.elv == 90
            msk = ((ds.elv == 90) | (ds.elv == -1000.)).compute()
            ds = ds.where(msk, drop=True)
            msk.close()

            if 'time' in ds.dims and ds.sizes['time'] > 0:
                file_time = pd.to_datetime(ds.coords['time'].values, unit='s')
                time_range = (file_time.min(), file_time.max())
                time_ranges.append(time_range)

        data[hour] = {'files': valid_files, 'timeranges': time_ranges}

    return data


def get_files_ranges_mmclx(day, month, year, source):
    """
    Returns a nested dictionary containing file paths of all files with zenith measurements
    and their corresponding start and end times for each hour of the specified date.
    """
    data = {}

    for hour in range(24):
        pth = f"{year}{month:02d}{day:02d}_{hour:02d}*.mmclx"
        full_path = os.path.join(source, pth)

        # Find all files for this hour and filter out PPI's and RHI's
        hourfiles = glob.glob(full_path)
        valid_files = sorted(f for f in hourfiles if ('.ppi' not in f) and ('.rhi' not in f) and ('.azisectorscan' not in f))

        # Processing files using Dask
        time_ranges = []
        for f in valid_files:
            # Open the dataset and extract the time range
            ds = xr.open_dataset(f, chunks={}, decode_times=False)  # Lazy loading
            
            # Filter out time periods where ds.elv == 90
            msk = ((ds.elv == 90) | (ds.elv == -1000)).compute()
            ds = ds.where(msk, drop=True)
            msk.close()

            if 'time' in ds.dims and ds.sizes['time'] > 0:
                file_time = pd.to_datetime(ds.coords['time'].values, unit='s')
                time_range = (file_time.min(), file_time.max())
                time_ranges.append(time_range)

        data[hour] = {'files': valid_files, 'timeranges': time_ranges}

    return data
