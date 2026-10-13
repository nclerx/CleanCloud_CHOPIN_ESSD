#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:18:25 2025
Copy of "00_cfradial_netcdf_to_vaex_archive.py" from https://github.com/ltelab/Zdr_calibration/

Note: vaex not compatible with python 3.12
@author: clerx
"""

import os
import glob
import vaex
import datetime
import h5py
import numpy as np
from netCDF4 import Dataset

# ------------------------------------------------------
# CONSTANTS
campaign_dir_dic = {'DAVOS_2010': '/ltedata/Davos/2009-2010/MXPOL/Proc_data/',
                    'HYMEX_2012': '/ltedata/HYMEX/SOP_2012/Radar/Proc_data/',
                    'HYMEX_2013': '/ltedata/HYMEX/SOP_2013/Radar/Proc_data/',
                    'PAYERNE_2014': '/ltedata/Payerne_2014/Radar/Proc_data/',
                    'CLACE_2014': '/ltedata/CLACE2014/Radar/Proc_data/',
                    'DAVOS_2014': '/ltedata/Davos/2014/Radar/Proc_data/',
                    'APRES3_2015': '/ltedata/APRES3/2015-16/Radar/Proc_data/',
                    'APRES3': '/ltedata/APRES3/2015-16/Radar/Proc_data/',
                    'VALAIS_2016': '/ltedata/Valais_2016/Radar/Proc_data/',
                    'ICEPOP_2018': '/ltedata/ICEPOP_2018/Radar/Proc_data/',
                    'PLATO_2019': '/ltenas3/PLATO_2019/Radar/Proc_data',
                    'ICEGENESIS_2021':'/ltenas8/users/anneclaire/ICEGENESIS_2021/MXPol/Proc_data_v0_clutter_filter_no_thres_PPI/',
                    'CHOPIN_2024': '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MXPol/proc_data/L0_new/',
                    # 'CHOPIN_2024': '/home/clerx/_ltesrv7_CHOPIN/MXPol/proc_data/L0/',
                    }

campaign_year_dic = {'DAVOS_2010': '201[0-1]',
                    'HYMEX_2012': '2012',
                    'HYMEX_2013': '2013',
                    'PAYERNE_2014': '2014',
                    'CLACE_2014': '2014',
                    'DAVOS_2014': '2014',
                    'APRES3': '201[5-6]',
                    'VALAIS_2016': '201[6-7]',
                    'ICEPOP_2018': '2018',
                    'PLATO_2019': '201[8-9]',
                    'ICEGENESIS_2021':'2021',
                    'CHOPIN_2024': '202[4-5]'}

DEFAULT_CAMPAIGN = 'CHOPIN_2024'#'PLATO_2019'
DEFAULT_START_DIR = campaign_dir_dic[DEFAULT_CAMPAIGN]
DEFAULT_YEAR = campaign_year_dic[DEFAULT_CAMPAIGN]

# For names of variables
PYJACOPO = True # True if the processing was done using pyjacopo

# To exclude values too similar between each others in azimuth
MIN_FRACTION_OF_THEORETICAL_AZIMUTH = 0.25

# Plot
SAVE_PROFILES = False
SAVE_DIR = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/MXPol/Zdr_calib_202606/'
# SAVE_DIR = '/home/clerx/_ltesrv7_CHOPIN/MXPol/Zdr_calib/'
# SAVE_DIR = '/ltenas8/campaigns/ICEGENESIS_2021/MXPol/Zdr_calib/archives_2/'

# Number of scan to be analyzed
DEFAULT_N_SCANS = 'all'
# DEFAULT_N_SCANS = 5

# Error code
ERR = 999

EMPTY_RETURN = [None, None, None, None, None, None, None]

VARIABLES = ['Zh', 'Zdr', 'Rhohv', 'SNRh', 'SNRv']
# ------------------------------------------------------


def search_vscans_no_spectra(dirname=DEFAULT_START_DIR, year=DEFAULT_YEAR):
    '''
    Function for searching the vertical scans in the directory (and
    subdirectories).

    Input:
    - dirname, a string with the name of the directory of the campaign.
                The path should end at "Proc_data", do not include the year.
    - year, a string containing the year to be considered. If unknown or
            if you want to analyze all years in dirname, put "*".
            If you want to analyze a range of years put "[YYYY1, YYYY2]".

    Output:
    - all_vscans, a list of strings (sorted in alphabetical order) of all the
                    filenames of the vertical scans. The filenames contain the
                    full paths.
    '''
    # You may need to adjust this depending on file organization / file naming convention
    # all_vscans = glob.glob(os.path.join(dirname, year, '*', '*', 'MXPol-*-PPI-090_0.nc'))
    all_vscans = [f for f in glob.glob(os.path.join(dirname, '**/*Zdr*.nc'), recursive=True)]

    # Sorting the files, to be sure they are in chronological order
    all_vscans.sort()
    return all_vscans


def check_file(path, variables=VARIABLES):
    """
    Check if file is corrupted using h5py
    ----------
    path : input path of file to check (string)
    variables: list of required variables

    Returns
    -------
    e : error (string)
    path : corrupted file (string)
    """
    try:
        with h5py.File(path, 'r') as f:
            errors = ['Variables not found:']
            for key in f.keys():
                if key in variables:
                    data = f.get(key)
                    if data is not None:
                        continue
                    else:
                        errors.append(key)
                    
            if len(errors) > 1:
                return path, errors
            
    except (OSError, KeyError, RuntimeError) as e:
        # If an error occurs (file is corrupted or other issues)
        return path, [f"Error opening file {path}: {e}"]
    
    return None # return none if no error occurs


def check_file_nc(nc_fid, variables=VARIABLES):
    errors = ['Variables not found:']
    try:
        for var in variables:
            if var in nc_fid.variables:
                data = nc_fid.variables[var][:]
                if data is None:
                    errors.apend(var)
                else:
                    errors.append(var)
        if len(errors) > 1:
            return errors
        
    except (OSError, KeyError, RuntimeError) as e:
        return f"Error accessing variables: {e}"
    
    return None


def read_profiles(all_vscans, valid_fraction_threshold=0.9):
    """
    Function opening all Zdr scans checking whether files are not corrupt (2025 addition).
    If the file is corrupt, append corrupt file name to excluded list and continue. 
    The check for corrupt files slows down the df creating about 3 times 
    (could potentially be optimised at some point but runtime for ~3050 files 
     is still more or less workable - roughly 20 minutes on laptop without saving).
    
    Parameters
    ----------
    all_vscans : list of file paths for all scans
    valid_fraction_threshold : float, optional
        minimum threshold of time steps to be nan-free for the scan to be included
        in the Zdr analysis (default is 0.9)

    Returns
    -------
    vaex dataframe, list of excluded (corrupt) files
    """
    
    # Reducing number if we want to test
    n_scans = DEFAULT_N_SCANS  # len(all_vscans)

    # Limiting num. of profiles if desired
    if DEFAULT_N_SCANS != "all":
        all_vscans = all_vscans[0:n_scans]

    print('Tot. num. scans: %d' % len(all_vscans))

    excluded_files = []
    
    zdr_1d_list = []
    zh_1d_list = []
    rhohv_1d_list = []
    snrh_1d_list = []
    snrv_1d_list = []
    r_matrix_1d_list = []
    az_matrix_1d_list = []
    t_1d_list = []
    event_idx_1d_list = []

    valid_event_idx = 0

    for i_f, fname in enumerate(all_vscans):
        if not i_f % 25:
            print('%d / %d' % (i_f, len(all_vscans)))
            
        check_result = check_file(fname)    
        if check_result is not None:
            path, errors = check_result
            excluded_files.append((path, errors))
            continue    # skip processing file and move to the next one
        
        # Opening the vscan
        with Dataset(fname, 'r') as nc_fid:
            # error = check_file_nc(nc_fid)
            # if error is not None:
            #     excluded_files.append((fname, error))
            #     continue
            
            if PYJACOPO:            
                r = np.array(nc_fid.variables['range'][:], np.float32)
                t = np.array(nc_fid.variables['time'][:], np.float32)
                az = np.array(nc_fid.variables['azimuth'][:], np.float32)
                try:
                    old_zdr_offset = nc_fid.getncattr('ZdrOffset_dB')
                except:
                    old_zdr_offset = 0
                    print('No previous ZdrOffset found, setting to 0...')
            else:
                r = np.array(nc_fid.variables['Range'][:], np.float32)
                t = np.array(nc_fid.variables['Time'][:], np.float32)
                az = np.array(nc_fid.variables['Azimuth'][:], np.float32)
                try:
                    old_zdr_offset = nc_fid.getncattr('ZdrOffset-dB')
                except:
                    old_zdr_offset = 0
                    print('No previous ZdrOffset found, setting to 0...')
                                
            ngates = r.shape[0]
            # Convert azimuth to positive
            az[az < 0.] += 360.
            # If the azimuths were spaced equally
            normal_az_step = 360. / az.shape[0]

            # Difference between 1 azimuth and the next
            az_diff = np.abs(np.roll(az, -1) - az)
            az_diff[-1] = az_diff[-2]
            az_diff[az_diff > 300.] -= 360.
            az_diff = np.abs(az_diff)

            # Accepting only the azimuths whose distance to the next
            # is at least the set fraction of the accepted value (i.e. enough 
            # pedestal movement between consecutive measurements)
            az_corr_idx = az_diff > normal_az_step * MIN_FRACTION_OF_THEORETICAL_AZIMUTH
            az_corr = az[az_corr_idx]
            
            n_valid_timesteps = az_corr_idx.sum()

            # If the radar moved at all
            if np.sum(az_corr_idx) > 0:
                # Loading
                zdr = np.array(-nc_fid.variables['Zdr'][:], dtype=np.float32).T + old_zdr_offset
                zh = np.array(nc_fid.variables['Zh'][:], dtype=np.float32).T
                # zv = np.array(nc_fid.variables['Zv'][:], dtype=np.float32)
                rhohv = np.array(nc_fid.variables['Rhohv'][:], dtype=np.float32).T
                snrh = np.array(nc_fid.variables['SNRh'][:], np.float32).T
                snrv = np.array(nc_fid.variables['SNRv'][:], np.float32).T

                # Selecting the acceptable gates
                zdr[zdr < -998] = np.nan
                zh[zh < -998] = np.nan
                # zv[zv < -998] = np.nan
                rhohv[rhohv < -998] = np.nan
                snrh[snrh < -998] = np.nan
                snrv[snrv < -998] = np.nan

                zdr[zdr > 998] = np.nan
                zh[zh > 998] = np.nan
                # zv[zv > 998] = np.nan
                rhohv[rhohv > 998] = np.nan
                snrh[snrh > 998] = np.nan
                snrv[snrv > 998] = np.nan

                # select measurements with valid azimuths only (moving radar)
                zdr = zdr[:, az_corr_idx]
                zh = zh[:, az_corr_idx]
                # zv = zv[:, az_corr_idx]
                rhohv = rhohv[:, az_corr_idx]
                snrh = snrh[:, az_corr_idx]
                snrv = snrv[:, az_corr_idx]

                # Time of accepted azimuth
                t = t[az_corr_idx]

                # R and az in 2d
                az_matrix, r_matrix = np.meshgrid(az_corr, r)

                # Checking which gates are totally free of nans during the azimuth rotation
#                 nangates_zdr = np.sum(np.isfinite(zdr), axis=1) < az_corr_idx.sum()
#                 nangates_zh = np.sum(np.isfinite(zh), axis=1) < az_corr_idx.sum()
#                 nangates_rhohv = np.sum(np.isfinite(rhohv), axis=1) < az_corr_idx.sum()
#                 nangates_snrh = np.sum(np.isfinite(snrh), axis=1) < az_corr_idx.sum()
#                 nangates_snrv = np.sum(np.isfinite(snrv), axis=1) < az_corr_idx.sum()

                # 2025: the original approach of only selecting totally nan-free gates doesn't 
                # work since there are too many dropped data blocks in all of the CHOPIN-data
                # where none of the range gates contain data (renamed variables for clarity)
                valid_zdr = np.sum(np.isfinite(zdr), axis=1) >= valid_fraction_threshold * n_valid_timesteps
                valid_zh = np.sum(np.isfinite(zh), axis=1) >= valid_fraction_threshold * n_valid_timesteps
                valid_rhohv = np.sum(np.isfinite(rhohv), axis=1) >= valid_fraction_threshold * n_valid_timesteps
                valid_snrh = np.sum(np.isfinite(snrh), axis=1) >= valid_fraction_threshold * n_valid_timesteps
                valid_snrv = np.sum(np.isfinite(snrv), axis=1) >= valid_fraction_threshold * n_valid_timesteps
                
                accepted_gates = np.logical_and.reduce([valid_zdr, valid_zh, valid_rhohv, valid_snrh, valid_snrv])
                nangates = np.logical_not(accepted_gates)
                                
                if nangates.sum() < ngates:
                    # # In case we need to check sizes
                    # print('Sizes:')
                    # print('zdr:   ', zdr.shape)
                    # print('zh:    ', zh.shape)
                    # print('rhohv: ', rhohv.shape)
                    # print('snrh:  ', snrh.shape)
                    # print('snrv:  ', snrv.shape)

                    # print('\nDims:')
                    # print('r:  ', r.shape)
                    # print('az: ', az_corr.shape)
                    
                    # print('\nDims after meshgrid:')
                    # print('r:  ', r_matrix.shape)
                    # print('az: ', az_matrix.shape)

                    # print('\nAccepted gates:')
                    # print(zdr[accepted_gates, :].shape)
                    # print(zh[accepted_gates, :].shape)
                    # print(rhohv[accepted_gates, :].shape)
                    # print(snrh[accepted_gates, :].shape)
                    # print(snrv[accepted_gates, :].shape)
                    # print(r_matrix[accepted_gates, :].shape)
                    # print(az_matrix[accepted_gates, :].shape)
                    
                    zdr_1d = zdr[accepted_gates, :].flatten()
                    zh_1d = zh[accepted_gates, :].flatten()
                    rhohv_1d = rhohv[accepted_gates, :].flatten()
                    snrh_1d = snrh[accepted_gates, :].flatten()
                    snrv_1d = snrv[accepted_gates, :].flatten()
                    r_matrix_1d = r_matrix[accepted_gates, :].flatten()
                    az_matrix_1d = az_matrix[accepted_gates, :].flatten()

                    # Creating a single array for time and scan index
                    t_1d = np.ones(zdr_1d.shape, dtype='float32') * t[0]

                    valid_event_idx += 1
                    event_idx_1d = np.ones(zdr_1d.shape, dtype='float32') * valid_event_idx

                    zdr_1d_list.append(zdr_1d)
                    zh_1d_list.append(zh_1d)
                    rhohv_1d_list.append(rhohv_1d)
                    snrh_1d_list.append(snrh_1d)
                    snrv_1d_list.append(snrv_1d)
                    r_matrix_1d_list.append(r_matrix_1d)
                    az_matrix_1d_list.append(az_matrix_1d)
                    t_1d_list.append(t_1d)
                    event_idx_1d_list.append(event_idx_1d)

    # Concatenating all the 1d arrays
    zdr_1d_concatenated = np.concatenate(zdr_1d_list)
    zh_1d_concatenated = np.concatenate(zh_1d_list)
    rhohv_1d_concatenated = np.concatenate(rhohv_1d_list)
    snrh_1d_concatenated = np.concatenate(snrh_1d_list)
    snrv_1d_concatenated = np.concatenate(snrv_1d_list)
    r_matrix_1d_concatenated = np.concatenate(r_matrix_1d_list)
    az_matrix_1d_concatenated = np.concatenate(az_matrix_1d_list)
    t_1d_concatenated = np.concatenate(t_1d_list)
    event_idx_1d_concatenated = np.concatenate(event_idx_1d_list)

    # To Vaex dataframe
    df = vaex.from_arrays(idx=event_idx_1d_concatenated, t=t_1d_concatenated,
                        r=r_matrix_1d_concatenated, az=az_matrix_1d_concatenated,
                        zdr=zdr_1d_concatenated, zh=zh_1d_concatenated,
                        rhohv=rhohv_1d_concatenated, snr_h=snrh_1d_concatenated,
                        snrv=snrv_1d_concatenated)
    
    return df, excluded_files


def main():
    # dirname = DEFAULT_START_DIR
    # year = DEFAULT_YEAR
    # Getting the list of all the vertical scans in the directory
    all_vscans = search_vscans_no_spectra()
    if len(all_vscans) == 0:
        print("No vertical PPI found in specified directory and year.")
        return ERR
    
    df, excluded_files = read_profiles(all_vscans)

    ini_date = datetime.datetime.fromtimestamp(int(df.data.t[0])).strftime('%Y%m%d-%H%M%S')
    end_date = datetime.datetime.fromtimestamp(int(df.data.t[-1])).strftime('%Y%m%d-%H%M%S')

    out_fname = 'dataframe_%s_from_%s_to_%s.hdf5' % (DEFAULT_CAMPAIGN, ini_date, end_date)
    out_fpath = os.path.join(SAVE_DIR, out_fname)

    print(df)
    df.export(out_fpath, compression='zstd')
    print('Exported to %s ' % out_fname)


if __name__ == '__main__':
    main()