
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:56:46 2025

run pamtra to calculate atmospheric attenuation profiles for clear-sky conditions based on radio sounding data from CLEANCLOUD campaign

@author: clerx
"""


#%% imports
import os 
import glob
import pyPamtra 
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import netCDF4 as nc

from datetime import datetime
from collections import defaultdict
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

#%% constants

base_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024'

freqs = ['X', 'Ka', 'W']

# for more information on additional properties that can be set, see https://pamtra.readthedocs.io/en/latest/settings.html 
radar_properties = {
    'X': {
        'name': 'MXPol',
        'frequency': 9.41,
        'mode': 'spectrum',  # alternative is 'simple' or 'moments'
        'polarisation': "'HH', 'HV', 'VH', 'VV'",
        'n_fft': 512,
        'no_avg': 3,
        't_int': 1,
        'vel_Nyq': 10,
        'beamwidth_deg': 1.3,
        'lin_dB': 'dB',
        'noise_level': 0.1,
    },
    'Ka': {
        'name': 'MIRA',
        'frequency': 35.2,
        'mode': 'spectrum', 
        'polarisation': ['HH', 'HV', 'VH', 'VV'],
        'n_fft': 512, 
        'no_avg': 3,
        't_int': 5,
        'vel_Nyq': 10,
        'beamwidth_deg': 0.5, 
        'lin_dB': 'lin',
        'noise_level': -20,
    },
    'W': { # 25 m mode
        'name': 'BASTA',
        'frequency': 94.95,
        'mode': 'simple',
        'polarisation': ['NN'],
        'n_fft': 4096,
        'no_avg': 5,
        't_int': 3,
        'vel_Nyq': 10,
        'beamwidth_deg': 0.4,
        'lin_dB': 'dB',
        'noise_level': -40, # roughly eyeballed from Fig. 12f from Delanoë et al. 2016 for 8-10 km altitude
    }
}

edr = 1e-4 # eddy dissipation rate


#%% pamtra input constants (hydrometeor characteristics)

# descriptor file:
    # hydro_name: hydrometeor name
    # as_ratio: aspect ratio (< 1 means oblate)
    # liq_ice: hydrometeror phase, liquid (1) or ice (-1)
    # rho_ms:
    # a_ms: a parameter of mass-size relationship
    # b_ms: b parameter mass-size relationship
    # alpha_as: alpha parameter of cross-section area-size relationship
    # beta_as: beta parameter of cross-section area-size relationship
    # moment_in: moment provided in input file
    # nbin: number of discrete bins (internally, nbins+1 is used)
    # dist_name : 
    # p_1
    # p_2
    # p_3
    # p_4
    # d_1
    # d_2
    # scat_name: 
    # vel_size_mod:
    # canting: canting angle of hydrometeors, only for Tmatrix and SSRG [deg]

# descriptorFile = np.array([
#         #['hydro_name' 'as_ratio' 'liq_ice' 'rho_ms' 'a_ms' 'b_ms' 'alpha_as' 'beta_as' 'moment_in' 'nbin' 'dist_name' 'p_1' 'p_2' 'p_3' 'p_4' 'd_1' 'd_2' 'scat_name' 'vel_size_mod' 'canting']
#         ('cwc_q', -99.0,  1, -99.0,   -99.0, -99.0,  -99.0, -99.0, 3, 1, 'mono', -99.0, -99.0,   2.0,    1.0,   2.0e-6,   8.0e-5, 'disabled', 'khvorostyanov01_drops', -99.0),
#         ('iwc_q', 1.0, -1, 700.0, -99.0,  -99.0,  -99.0,   -99.0, 3, 1, 'mono', -99.0, -99.0, 1.564, 0.8547, 1.744e-5, 9.369e-3, 'disabled',   'heymsfield10_particles',  -99.0),
#         ('rwc_q', -99.0, 1, -99.0, -99.0, -99.0,  -99.0, -99.0, 3, 50, 'exp', 0.22, 2.2,   -99.0,    -99.0,  0.00012,   0.006, 'disabled', 'khvorostyanov01_drops',  -99.0),
#         ('swc_q', 1.0, -1, -99.0, 0.069, 2.0, 0.3971,  1.88, 3, 50, 'exp', 2.0e06, 0.,  -99.0, -99.0, 2.0e-04,  0.02, 'disabled',   'heymsfield10_particles', -99.0),
#         ('gwc_q', -99.,-1, -99.0, 169.6, 3.1, -99.0, -99.0, 3, 50, 'exp', -99.0, -99.0, 4.0e6, -99.0, 1.0e-10, 1.0e-2, 'disabled', 'khvorostyanov01_spheres', -99.0)],
#         dtype=[('hydro_name', 'S15'), ('as_ratio', '<f8'), ('liq_ice', '<i8'), ('rho_ms', '<f8'), ('a_ms', '<f8'), ('b_ms', '<f8'), ('alpha_as', '<f8'), ('beta_as', '<f8'), ('moment_in', '<i8'), ('nbin', '<i8'), ('dist_name', 'S15'), ('p_1', '<f8'), ('p_2', '<f8'), ('p_3', '<f8'), ('p_4', '<f8'), ('d_1', '<f8'), ('d_2', '<f8'), ('scat_name', 'S20'), ('vel_size_mod', 'S30'), ('canting', '<f8')] 
#         )


descriptorFile = np.array([
      #['hydro_name' 'as_ratio' 'liq_ice' 'rho_ms' 'a_ms' 'b_ms' 'alpha_as' 'beta_as' 'moment_in' 'nbin' 'dist_name' 'p_1' 'p_2' 'p_3' 'p_4' 'd_1' 'd_2' 'scat_name' 'vel_size_mod' 'canting']
       ('cwc_q', 1.0,  1, -99.0,   -99.0, -99.0,  -99.0, -99.0, 13, 100, 'mgamma', -99.0, -99.0,   2.0,    1.0,   2.0e-6,   8.0e-5, 'mie-sphere', 'corPowerLaw_24388657.6_2.0', -99.0),
       ('iwc_q', 1.0, -1, -99.0, 1.58783,  2.56,  0.684,   2.0, 13, 100, 'mgamma', -99.0, -99.0, 1.564, 0.8547, 1.744e-5, 9.369e-3, 'ssrg-rt3',   'corPowerLaw_30.606_0.5533',  -99.0),
       ('rwc_q', 1.0,  1, -99.0,   -99.0, -99.0,  -99.0, -99.0, 13, 100, 'mgamma', -99.0, -99.0,   2.0,    1.0,  0.00012,   8.2e-3, 'mie-sphere', 'corPowerLaw_494.74_0.7031',  -99.0),
       ('swc_q', 0.6, -1, -99.0,   0.038,   2.0, 0.3971,  1.88, 13, 100, 'mgamma', -99.0, -99.0,   1.0,    1.0,  5.13e-5, 2.294e-2, 'ssrg-rt3',   'corPowerLaw_5.511054_0.25',  -99.0),
       ('gwc_q', 1.0, -1, -99.0,  500.86,  3.18,  -99.0, -99.0, 13, 100, 'mgamma', -99.0, -99.0,  5.37,   1.06,  2.11e-4,   1.3e-2, 'mie-sphere', 'corPowerLaw_406.67_0.85',    -99.0), 
       ('hwc_q', 1.0, -1, -99.0,  392.33,   3.0,  -99.0, -99.0, 13, 100, 'mgamma', -99.0, -99.0,   5.0,    1.0,  1.87e-4,   1.1e-2, 'mie-sphere', 'corPowerLaw_106.33_0.5',     -99.0)],
      dtype=[('hydro_name', 'S15'), ('as_ratio', '<f8'), ('liq_ice', '<i8'), ('rho_ms', '<f8'), ('a_ms', '<f8'), ('b_ms', '<f8'), ('alpha_as', '<f8'), ('beta_as', '<f8'), ('moment_in', '<i8'), ('nbin', '<i8'), ('dist_name', 'S15'), ('p_1', '<f8'), ('p_2', '<f8'), ('p_3', '<f8'), ('p_4', '<f8'), ('d_1', '<f8'), ('d_2', '<f8'), ('scat_name', 'S20'), ('vel_size_mod', 'S30'), ('canting', '<f8')]
      )

# from ACBR "spectralradar-mcmc/simulate_radar/simulate_spectra.ipynb"

order_params = {
    'M0':0,
    'Deff':1, 
    'mu':2, # 3rd parameter of psd
    'a_mass_size':3,
    'b_mass_size':4,
    'alpha_area_size':5,
    'beta_area_size':6,
    
}

Nt = order_params['M0']
Deff = order_params['Deff']
aspect_ratio = 0.6 # < 1 means oblate
canting_angle = 0.

rho_air = 1.2

#%% functions
def meters_to_km(y, _):
    return f"{y / 1000:.1f} km"


def get_RSmetadata(filename, enc='latin1'):
    """
    function to read metadata from first and last lines of radio sounding textfile -- not fully correct/complete!!

    Parameters
    ----------
    filename : string
        file path of the radio sounding data (.txt format)        
    enc : string, optional
        encoding of the file, default is 'latin1'.

    Returns
    -------
    metadata : dictionary
        dictionary containing the metadata extracted from the .txt file according to a standardised format

    """
    metadata = defaultdict(str)
    with open(filename, encoding=enc) as f:
        meta_lines = [next(f).strip() for _ in range(17)]
    
    metadata['Launch Date'] = re.search(r'Launch Date:\s*(.*?)\s+Launch Time:', meta_lines[1]).group(1).strip()
    metadata['Launch Time'] = re.search(r'Launch Time:\s*(.*?)\s+End of Ascent:', meta_lines[1]).group(1).strip()
    metadata['End of Ascent'] = re.search(r'End of Ascent:\s*(.*)', meta_lines[1]).group(1).strip()
    
    metadata['Station Name'] = re.search(r'Station name:\s*(.*?)\s+Serialnumber:', meta_lines[2]).group(1).strip()
    metadata['Serial Number'] = re.search(r'Serialnumber:\s*(.*)', meta_lines[2]).group(1).strip()
    
    metadata['Pressure [hPa]'] = re.search(r'Pressure:\s*(.*?)\s+Temperature:', meta_lines[4]).group(1).strip()
    metadata['Temperature [°C]'] = re.search(r'Temperature:\s*(.*?)\s+Humidity:', meta_lines[4]).group(1).strip()
    metadata['Humidity'] = re.search(r'Humidity:\s*(.*)', meta_lines[4]).group(1).strip()
    
    metadata['Wind Direction'] = re.search(r'Wind direction:\s*(.*?)\s+Wind speed:', meta_lines[5]).group(1).strip()
    metadata['Wind Speed [m/s]'] = re.search(r'Wind speed:\s*(.*?)\s+Cloud group:', meta_lines[5]).group(1).strip()
    metadata['Cloud Group'] = re.search(r'Cloud group:\s*(.*)', meta_lines[5]).group(1).strip()
    
    metadata['Longitude [°]'] = re.search(r'Longitude:\s*(.*?)\s+Latitude:', meta_lines[7]).group(1).strip()
    metadata['Latitude [°]'] = re.search(r'Latitude:\s*(.*?)\s+Altitude:', meta_lines[7]).group(1).strip()
    metadata['Altitude [m a.s.l.]'] = re.search(r'Altitude:\s*(.*)', meta_lines[7]).group(1).strip()
    
    metadata['Institute'] = re.search(r'Company:\s*(.*)', meta_lines[9]).group(1).strip()
    metadata['Data processed by'] = re.search(r'Operator:\s*(.*)', meta_lines[10]).group(1).strip()
    
    metadata['Highest Point (Pressure)'] = re.search(r'Highest Point:\s*(.*?)\s+', meta_lines[12]).group(1).strip()
    metadata['Highest Point (Altitude)'] = re.search(r'\s+([\d]+.*m)', meta_lines[12]).group(1).strip()
    
    # Optional fields
    metadata['Tropopauses'] = meta_lines[13].split(':', 1)[1].strip() if ':' in meta_lines[13] else ''
    metadata['Remarks'] = meta_lines[16]
    
    return metadata


def read_RStxt(filename):  
    """
    function to read radio sounding data and return a dataframe

    Parameters
    ----------
    filename : string
        file path of the radio sounding data (.txt format)

    Returns
    -------
    data_valid : pandas dataframe
        dataframe containing timestamps and numeric data of radio sounding input file

    """
    cols_RSdata = ['time_sec', 'time_UTC', 'P_hPa', 'T_degC', 'RH_perc', 'Wsp_mps', 'Wd_deg', 'lon_deg', 'lat_deg', 'elv_masl', 'Tdew_degC']
    data_full= pd.read_csv(filename, names=cols_RSdata, encoding='latin1', sep='\t', index_col=0)
    
    date = filename[-14:-6]
    cols_to_convert = data_full.columns.difference(['time_UTC'])
    
    data_valid = data_full[pd.to_numeric(data_full.index, errors='coerce').notna()]
    data_valid.loc[:, cols_to_convert] = data_valid[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    data_valid.loc[:, 'time_UTC'] = data_valid['time_UTC'].apply(lambda x: str(x).strip() if isinstance(x, str) else x)
    data_valid.loc[:, 'time_UTC'] = [pd.to_datetime(f"{date} {i}", format='%Y%m%d %H:%M:%S') for i in data_valid['time_UTC']]
    
    return data_valid
  

#%% date & RS file
# date = datetime(2024, 11, 26)
# year, month, day = date.year, date.month, date.day

hour = 15 # adjust this based on radio sounding filename
RSdir = os.path.join(base_dir, f"radiosondes/{year}{month:02}{day:02}{hour}/SOUNDING DATA")
RSfn = glob.glob(f"{RSdir}/*.txt")[0]

# metadata = get_metadata(RSfn)

#%% read RS file
rsdata = read_RStxt(RSfn)

# add radio sounding data to pamtra object (up to maximum elevation to remove data recorded while RS descending)
elvs = pd.to_numeric(rsdata['elv_masl'])[:rsdata['elv_masl'].idxmax()]
temp = (rsdata['T_degC'][:rsdata['elv_masl'].idxmax()] + 273.15).astype(float) # T in Kelvin
relhum = (rsdata['RH_perc'][:rsdata['elv_masl'].idxmax()]).astype(float)
press = (rsdata['P_hPa'][:rsdata['elv_masl'].idxmax()] * 100).astype(float) # P in Pa


#%% resample RS data to match ERA5 levels (and reduce length of pamtra run)

# era_h = [  182.95058,   396.37305,   614.224  ,   836.726  ,  1064.1284 ,
#         1296.6888 ,  1534.6927 ,  1778.4684 ,  2028.3717 ,  2284.7778 ,
#         2548.1975 ,  3098.3103 ,  3683.8257 ,  4308.1665 ,  4975.3115 ,
#         5692.097  ,  6466.543  ,  7309.827  ,  8238.743  ,  9273.333  ,
#        10452.343  , 11120.08   , 11865.504  , 12714.109  , 13697.196  ,
#        14858.614  , 16263.551  , 18486.316  , 20584.055  , 23813.549  ,
#        26419.5    , 31046.248  , 33597.54   , 35942.477  , 39497.023  ,
#        42449.973  , 47663.03   ]

era_h = [  1778.4684 ,  2028.3717 ,  2284.7778 ,
        2548.1975 ,  3098.3103 ,  3683.8257 ,  4308.1665 ,  4975.3115 ,
        5692.097  ,  6466.543  ,  7309.827  ,  8238.743  ,  9273.333  ,
       10452.343  , 11120.08   , 11865.504  , 12714.109  , 13697.196  ,
       14858.614  , 16263.551  , 18486.316  , 20584.055  , 23813.549  ,
       26419.5    , 31046.248  , 33597.54   , 35942.477  , 39497.023  ,
       42449.973  , 47663.03   ]

sonde_h = elvs.values       # radiosonde altitudes in meters
max_h = sonde_h.max()

sonde_T = temp.values       # in Kelvin
sonde_RH = relhum.values    # in percent
sonde_P = press.values      # in Pa

if not np.all(np.diff(sonde_h) > 0):
    sort_idx = np.argsort(sonde_h)
    sonde_h = sonde_h[sort_idx]
    sonde_T = sonde_T[sort_idx]
    sonde_RH = sonde_RH[sort_idx]
    sonde_P = sonde_P[sort_idx]

interp_T = interp1d(sonde_h, sonde_T, kind='linear', bounds_error=False, fill_value=np.nan)
interp_RH = interp1d(sonde_h, sonde_RH, kind='linear', bounds_error=False, fill_value=np.nan)
interp_P = interp1d(sonde_h, sonde_P, kind='linear', bounds_error=False, fill_value=np.nan)

mask = era_h <= max_h
era_h_mask = [i for i in era_h if i <= max_h]
T_on_era = interp_T(era_h)[mask]
RH_on_era = interp_RH(era_h)[mask]
P_on_era = interp_P(era_h)[mask]

#%% pamtra with RS data at ERA5 levels
freqs = [94.95]

pam = pyPamtra.pyPamtra()

pam.set["verbose"] = 1
pam.set["pyVerbose"] = 2

pam.df.addHydrometeor(('cwc_q',-99.,1,-99.,-99.,-99.,-99.,-99.,3,1,'mono',-99.,-99.,-99.,-99.,2e-5,-99.,'mie-sphere', 'khvorostyanov01_drops',-99.))
pam = pyPamtra.importer.createUsStandardProfile(pam, hgt_lev=era_h_mask, temp_lev=T_on_era, relhum_lev=RH_on_era, press_lev=P_on_era)
pam.nmlSet["passive"] = False

#%% run pamtra & plot atmospheric attenuation
pam.runPamtra(freqs)
Att_atmo = pam.r['Att_atmo'].squeeze()
PIA = 2*np.cumsum(Att_atmo) - Att_atmo # 2-way path-integrated attenuation, following formulas of PAMTRA

plt.plot(PIA, era_h_mask[1:])

#%% pamtra with small-scale RS data
freq = [94.95]
pamrs = pyPamtra.pyPamtra()
pamrs.set["verbose"] = 1
pamrs.set["pyVerbose"] = 2

pamrs.df.addHydrometeor(('cwc_q',-99.,1,-99.,-99.,-99.,-99.,-99.,3,1,'mono',-99.,-99.,-99.,-99.,2e-5,-99.,'mie-sphere', 'khvorostyanov01_drops',-99.))
pamrs = pyPamtra.importer.createUsStandardProfile(pamrs, hgt_lev=sonde_h, temp_lev=sonde_T, relhum_lev=sonde_RH, press_lev=sonde_P)
pamrs.nmlSet['passive'] = False

#%% 
pamrs.runPamtra(freq)
Att_atmo_rs = pamrs.r['Att_atmo'].squeeze()
PIA_rs = 2*np.cumsum(Att_atmo_rs)-Att_atmo_rs

plt.plot(PIA_rs, sonde_h[1:])


#%% ERA5 data
t_min = rsdata['time_UTC'].min()
t_max = rsdata['time_UTC'].max()


#%% not per se necessary but additional variables that can be fed into pamtra
pam.p['hgt_lev'] = np.array(era_h_mask, dtype=np.float64)[None, None, :]
pam.p['temp_lev'] = np.array(T_on_era, dtype=np.float64)[None, None, :]
pam.p['press_lev'] = np.array(P_on_era, dtype=np.float64)[None, None, :]
pam.p['relhum_lev'] = np.array(RH_on_era, dtype=np.float64)[None, None, :]
pam.p["hydro_q"][0,0,:,0] = 0
pam.nmlSet["radar_polarisation"] = "NN"
pam.nmlSet["radar_mode"] = "simple"
pam.nmlSet["randomseed"] = 10
pam.p["sfc_temp"] = np.array([[temp[0] + 273.15]], dtype=np.float64)
pam.p["sfc_press"] = np.array([[press[0]]], dtype=np.float64)
pam.p["sfc_relhum"] = np.array([[relhum[0]]], dtype=np.float64)

###################################### OLD ######################################

#%% initialise pamtra object
pam = pyPamtra.pyPamtra()

# add hydrometeor properties
for hyd in descriptorFile: pam.df.addHydrometeor(hyd)

pam = pyPamtra.importer.createUsStandardProfile(pam, hgt_lev=elvs, temp_lev=temp, relhum_lev=relhum, press_lev=press)
pam.p['hgt_lev'] = np.array(elvs, dtype=np.float64)[None, None, :]
pam.p['temp_lev'] = np.array(temp, dtype=np.float64)[None, None, :]
pam.p['press_lev'] = np.array(press, dtype=np.float64)[None, None, :]
pam.p['relhum_lev'] = np.array(relhum, dtype=np.float64)[None, None, :]
pam.p["sfc_temp"] = np.array([[temp[0] + 273.15]], dtype=np.float64)
pam.p["sfc_press"] = np.array([[press[0]]], dtype=np.float64)
pam.p["sfc_relhum"] = np.array([[relhum[0]]], dtype=np.float64)

#%% working cell (in progress) following https://github.com/igmk/pamtra/examples/pyPamTest_radarDualPol.py
pam = pyPamtra.pyPamtra()
# from ACBR https://github.com/ltelab/pyWprof/blob/69a531bff5adb519e2cffbdb85409f8f02484fec/pyWprof/attenuation_correction/01_calc_attenuation_pamtra_wout_lwc.py#L8
pam.df.addHydrometeor(('cwc_q',-99.,1,-99.,-99.,-99.,-99.,-99.,3,1,'mono',-99.,-99.,-99.,-99.,2e-5,-99.,'mie-sphere', 'khvorostyanov01_drops',-99.))
# pam.df.addHydrometeor(('ice', 0.3, -1, 917,917 *  np.pi / 6., 3, np.pi/4., 2, 0, 100, 'exp', 3000, 3e8, -99.0, -99.0, 100e-6,  1000e-6, 'tmatrix', 'heymsfield10_particles', 90.0))

# pam = pyPamtra.importer.createUsStandardProfile(pam, hgt_lev=elvs, temp_lev=temp, relhum_lev=relhum, press_lev=press)
pam = pyPamtra.importer.createUsStandardProfile(pam, hgt_lev=era_h_mask, temp_lev=T_on_era, relhum_lev=RH_on_era, press_lev=P_on_era)
pam.p['hgt_lev'] = np.array(era_h_mask, dtype=np.float64)[None, None, :]
pam.p['temp_lev'] = np.array(T_on_era, dtype=np.float64)[None, None, :]
pam.p['press_lev'] = np.array(P_on_era, dtype=np.float64)[None, None, :]
pam.p['relhum_lev'] = np.array(RH_on_era, dtype=np.float64)[None, None, :]
pam.p["hydro_q"][0,0,:,0] = 0
pam.nmlSet["passive"] = False

pam.p["sfc_temp"] = np.array([[temp[0] + 273.15]], dtype=np.float64)
pam.p["sfc_press"] = np.array([[press[0]]], dtype=np.float64)
pam.p["sfc_relhum"] = np.array([[relhum[0]]], dtype=np.float64)


freqs = [94.95]

pam.set["verbose"] = 1
pam.set["pyVerbose"] = 2
#%%
pam.runPamtra(freqs)
#%%

pam.nmlSet["passive"] = False
# pam.nmlSet["radar_mode"] = "spectrum"
pam.nmlSet["radar_mode"] = "simple"
pam.nmlSet["randomseed"] = 10
#pam.nmlSet["radar_dualpol"] = "both"
pam.nmlSet["radar_polarisation"] = "NN"

# pam.p["hydro_q"][:] = 0.001
pam.p['hydro_q'][0, 0, :, 0] = 0
pam.runPamtra(freqs)

pam.runParallelPamtra(freqs, pp_deltaX=1, pp_deltaY=1, pp_deltaF=10, pp_local_workers="auto")

# this works but takes a long time (1-1.5hr?) - parallel seems a bit faster but unclear
# pam.runPamtra(freqs,checkData=False)

# plt.plot(pam.r["Ze"].ravel())
# plt.plot(pam.r["radar_vel"],pam.r["radar_spectra"][0,0,0,1])

i_p, pol = (0, 'NN')
plt.plot(pam.r["radar_vel"][0,:],pam.r["radar_spectra"][0,0,0,0,i_p,:],label=pam.set["radar_pol"][i_p] + " " + str(pam.r["Ze"][...,i_p,0]))
plt.legend(loc="upper left")  
plt.title(pam.df.data["canting"])
plt.xlabel('velocity [m/s]')
plt.ylabel('reflectivity [dB]')


#
for i_p, pol in enumerate(pam.set['radar_pol']):
    print(pol, pam.r["Ze"][...,i_p,0])
print('ave HH VV ', np.log10(0.5*(10**pam.r["Ze"][...,1,0]+10**pam.r["Ze"][...,2,0])))
plt.figure(11)
plt.clf()
for i_p in range(pam.set["radar_npol"]):
  plt.plot(pam.r["radar_vel"][0,:],pam.r["radar_spectra"][0,0,0,0,i_p,:],label=pam.set["radar_pol"][i_p] + " " + str(pam.r["Ze"][...,i_p,0]))

plt.legend(loc="upper left")  
plt.title(pam.df.data["canting"])

plt.show()

# %% plot radiosounding data 

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# pressure
axes[0].plot(pam.p['press_lev'][0, 0, :] / 100., pam.p['hgt_lev'][0, 0, :])
axes[0].set_xlim([0, 900])
axes[0].set_xlabel('Pressure [hPa]')
axes[0].set_ylim([1960, 17000])
axes[0].set_ylabel('Height [km]')
axes[0].yaxis.set_major_formatter(FuncFormatter(meters_to_km))

# temperature 
axes[1].plot(pam.p['temp_lev'][0, 0, :] - 273.15, pam.p['hgt_lev'][0, 0, :])
axes[1].set_xlim([-80, 20])
axes[1].set_xlabel('Temperature [°C]')
axes[1].set_ylim([1960, 17000])
axes[1].set_ylabel('Height [km]')
axes[1].yaxis.set_major_formatter(FuncFormatter(meters_to_km))

# relative humidity
axes[2].plot(pam.p['relhum_lev'][0, 0, :], pam.p['hgt_lev'][0, 0, :])
axes[2].set_xlim([0, 100])
axes[2].set_ylim([1960, 17000])
axes[2].set_xlabel('Rel. Humidity [%]')
axes[2].set_ylabel('Height [km]')
axes[2].yaxis.set_major_formatter(FuncFormatter(meters_to_km))

tmin = rsdata['time_UTC'].min().strftime('%Y-%m-%d %H:%M')
tmax = rsdata['time_UTC'][elvs.idxmax()].strftime('%Y-%m-%d %H:%M')
plt.suptitle(f"Radio sounding data from {tmin} to {tmax}", fontweight='bold')
plt.tight_layout()

plt.show()


#%% (from example icon_groundbased) adapted for BASTA characteristics (25m mode)

pam.nmlSet['active'] = True
pam.nmlSet['radar_mode'] = 'spectrum'
pam.nmlSet['passive'] = False # Passive is time consuming
pam.set['verbose'] = 1 # set verbosity levels
pam.set['pyVerbose'] = 2 # change to 2 if you want to see job progress number on the output
pam.p['turb_edr'][:] = edr
pam.nmlSet['radar_airmotion'] = True
pam.nmlSet['radar_airmotion_model'] = 'constant'

pam.nmlSet['radar_fwhr_beamwidth_deg'] = 0.4
pam.nmlSet['radar_integration_time'] = 1.0
pam.nmlSet['radar_max_v'] = 5
pam.nmlSet['radar_min_v']= -5
pam.nmlSet['radar_nfft']= 4096
pam.nmlSet['radar_no_ave'] = 1
pam.nmlSet['radar_pnoise0'] = -40

pam.runPamtra(radar_properties['W']['frequency'])
pam.writeResultsToNetCDF('/home/clerx/pamtra_out/20241126_test_BASTA.nc')

pamW = nc.Dataset('/home/clerx/pamtra_out/20241126_test_BASTA.nc')

def readPamtraRadarMoments(ncfile, attVersus=1):
    runVars = ncfile.variables
    H = (runVars['height'][:,0,:])
    ttt = pd.to_datetime(runVars['datatime'][:,0],unit='s')
    tt = (np.tile(ttt,(H.shape[1],1)).T)
    a = 2.0*(runVars['Attenuation_Hydrometeors'][:,0,:,0,0] + runVars['Attenuation_Atmosphere'][:,0,:,0])
    A = a[:,::attVersus].cumsum(axis=1)[:,::attVersus]
    Ze = runVars['Ze'][:,0,:,0,0,0]
    MDV = -runVars['Radar_MeanDopplerVel'][:,0,:,0,0,0]
    SW = runVars['Radar_SpectrumWidth'][:,0,:,0,0,0]
    return H, tt, A, Ze, MDV, SW

versus = 1
Hw, ttw, Aw, Zew, MDVw, SWw = readPamtraRadarMoments(pamW, attVersus=versus)

xfmt = md.DateFormatter('%H')
timelim = [np.datetime64('2015-11-19 06:00'), np.datetime64('2015-11-19 23:59')]
hlim = [0, 10]
vlim = [-8, 8]
splim = [-90, 10]
TlimK = [0, 150]
TlimV = [100, 300]
Zmin, Zmax = -35, 25
Vmin, Vmax = -1, 5
Dmin, Dmax = -5, 20


fig1 = plt.figure(figsize=(12,6)); ax1 = plt.gca()
mesh1 = ax1.pcolormesh(ttw, Hw*0.001, Zew-Aw, cmap='jet', vmin=Zmin, vmax=Zmax)
ax1.set_ylim(hlim)
ax1.set_xlim(timelim)
ax1.set_ylabel('Height    [km]')
ax1.set_xlabel('time')
plt.colorbar(mesh1, ax=ax1, label='Z   [dBZ]')

# Plot Mead Doppler Velocity Ka-band
fig2 = plt.figure(figsize=(12,6)); ax2 = plt.gca()
mesh2 = ax2.pcolormesh(ttw, Hw*0.001, -MDVw, cmap='RdBu', vmin=Vmin, vmax=Vmax)
ax2.set_xlim(timelim)
ax2.set_ylim(hlim)
ax2.set_ylabel('Height    [km]')
ax2.set_xlabel('time')
plt.colorbar(mesh2, ax=ax2, label='MDV   [m/s]')

# %% run pamtra for different radar frequencies

pam.nmlSet["radar_noise_distance_factor"] = 1.0
pam.nmlSet['passive'] = False   # estimate brightness temperatures if set to True (slow)
pam.nmlSet['radar_airmotion'] = True # consider air motion in direction of radar beam
pam.nmlSet['radar_airmotion_model'] = 'constant' # model to describe vertical airmotion

pam.nmlSet["radar_save_noise_corrected_spectra"]=  False
pam.nmlSet['radar_aliasing_nyquist_interv'] = 1 # consider aliasing effects for overspending the nyquist range radar_aliasing_nyquist_interv times
pam.nmlSet['hydro_adaptive_grid'] = False
pam.nmlSet['radar_allow_negative_dD_dU'] = True # allow particle velocity to decrease with size
pam.set['verbose'] = 1
pam.set['pyVerbose'] = 2

pam.p["hydro_q"][:] = 0.00001 # or 0.00001 - unsure what this is but pamtra didn't seem to properly calculate without

for freq in freqs:
    pam.nmlSet['radar_mode'] = radar_properties[freq]['mode']
    pam.nmlSet['radar_fwhr_beamwidth_deg'] = radar_properties[freq]['beamwidth_deg'] # radar full width half radiation beamwidth (required for spectral broadening estimation)
    if radar_properties[freq]['mode'] == 'spectrum':
        pam.nmlSet['radar_nfft'] = radar_properties[freq]['n_fft'] # number of FFT points in the Doppler spectrum
        pam.nmlSet['radar_no_ave'] = radar_properties[freq]['no_avg'] # number of spectral averages
    pam.addSpectralBroadening(edr, 10, radar_properties[freq]['beamwidth_deg'], radar_properties[freq]['t_int'], radar_properties[freq]['frequency'], kolmogorov=0.5)
    pam.nmlSet['radar_integration_time'] = radar_properties[freq]['t_int'] # radar beamwidth (required for spectral broadening estimation)
    pam.nmlSet['radar_max_v'] = radar_properties[freq]['vel_Nyq']
    pam.nmlSet['radar_min_v'] = -radar_properties[freq]['vel_Nyq']
    #pam.nmlSet['radar_polarisation'] = radar_properties[freq]['polarisation']
    vel0 = np.linspace(-radar_properties[freq]['vel_Nyq'], radar_properties[freq]['vel_Nyq'], radar_properties[freq]['n_fft'], endpoint=False)
    dv0 = vel0[1] - vel0[0]
    if radar_properties[freq]['lin_dB'] == 'lin':
        pam.nmlSet['radar_pnoise0'] = 10*np.log10(np.array(radar_properties[freq]['noise_level'])) + 10*np.log10(dv0 * radar_properties[freq]['n_fft'])
    else:
        pam.nmlSet['radar_pnoise0'] = np.array(radar_properties[freq]['noise_level']) + dv0 * radar_properties[freq]['n_fft']
    pam.runPamtra(radar_properties[freq]['frequency'])
    outputpath = f"{base_dir}/pamtra/{rsdata['time_UTC'].min().strftime('%Y%m%d-%H%M')}_{radar_properties[freq]['name']}.nc"
    pam.writeResultsToNetCDF(outputpath)


#%%
diameter = pam.Object.vars_output.out_debug_diameter
sigma_D = pam.fortObject.vars_output.out_debug_back_of_d
eta_wo_turb = pam.fortObject.vars_output.out_debug_radarback
eta_turb = pam.fortObject.vars_output.out_debug_radarback_wturb
eta_turb_noisy = pam.fortObject.vars_output.out_debug_radarback_wturb_wnoise
velocity = pam.fortObject.vars_output.out_debug_radarvel


# %%
# %%

plt.plot(freqs, pam.r['radar_vel'], pam.r['radar_spectra'][0, 0, 1, 0, 0], label='turb')

# %%
