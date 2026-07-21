#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 09:48:11 2026

MXPol hydrometeor classification
based on script https://github.com/ltelab/pyjacopo/blob/main/pyjacopo/example/hydro_classif_example.py

@author: clerx
"""
#%% imports
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src import glob, os, np, pd, xr, gc, plt, pyart, time, datetime, matplotlib as mpl, warnings
from src.constants_input import dirs, base_dir, start_date, radar_altitude, scantimes_MXPol, MIRA_lapserate
from src.utils import generate_date_range
from src.radar_processing import Zdr_corr_pyart, add_Kdp, get_Zdr_corr, filter_radar
import scipy.io as sio


#%% constants & functions from ml_detection script (OBSOLETE)
"""OBSOLETE AND CURRENTLY NOT WORKING BUT KEPT FOR FUTURE REFERENCE"""
from scipy import spatial
from scipy.interpolate import InterpolatedUnivariateSpline, pchip, RegularGridInterpolator
from pyart.config import get_metadata
from copy import deepcopy

MAXTHICKNESS_ML = 1000
MAXHEIGHT_ML = 6000.
MINHEIGHT_ML = 100.
LOWMLBOUND = 0.7
UPMLBOUND = 1.3
SIZEFILT_M = 75
ZH_IM_BOUNDS = (10, 60)
RHOHV_IM_BOUNDS = (0.75, 1)
RHOHV_VALID_BOUNDS = (0.6, 1)
KE = 4 / 3.  # constant in the 4/3 earth radius model
# two extreme earth radii
R_EARTH_MAX = 6378.1370 * 1000
R_EARTH_MIN = 6356.7523 * 1000


def get_earth_radius(latitude):
    """
    Computes the earth radius for a given latitude

    Parameters
    ----------
    latitude: latitude in degrees (WGS84)

    Returns
    -------
    earth_radius : the radius of the earth at the given latitude
    """
    a = R_EARTH_MAX
    b = R_EARTH_MIN
    num = ((a ** 2 * np.cos(latitude)) ** 2 +
           (b ** 2 * np.sin(latitude)) ** 2)
    den = ((a * np.cos(latitude)) ** 2 +
           (b * np.sin(latitude)) ** 2)

    earth_radius = np.sqrt(num / den)

    return earth_radius


def r_to_h(earth_radius, gate_range, gate_theta):
    '''
    Computes the height of radar gates knowing the earth radius at the given
    latitude and the range and elevation angle of the radar gate.

    Inputs:
        earth_radius : the radius of the earth for a given latitude in m.

        gate_range : the range of the gate(s) in m.

        gate_theta : elevation angle of the gate(s) in degrees.

    Outputs:
        height : the height above ground of all specified radar gates
    '''

    height = ((gate_range**2 + (KE * earth_radius)**2 +
               2 * gate_range * KE * earth_radius *
               np.sin(np.deg2rad(gate_theta)))**(0.5) - KE * earth_radius)

    return height


def polar_to_cartesian(radar_sweep, field_name, cart_res=25, max_range=None, mapping=None):

    KE = 4 / 3.  # Constant in the 4/3 earth radius model
    # Two extreme earth radius
    R_EARTH_MAX = 6378.1370 * 1000
    R_EARTH_MIN = 6356.7523 * 1000
    
    # Get data to be interpolated
    pol_data = radar_sweep.get_field(0, field_name)
    is_ppi = False

    if mapping:
        # Check if mapping is usable:
        if is_ppi != mapping['is_ppi']:
            print('Input mapping does not correspond to given scan type, ignoring it')
            mapping = None
        elif mapping['dim_pol'] != pol_data.shape:
            print('Input mapping does not correspond to dimensions of given field'
                  ', ignoring it')
            mapping = None
        else:
            cart_res = mapping['res']
            max_range = mapping['max_range']

    # Get distances of radar data
    r = radar_sweep.range['data']

    if max_range is None:
        max_range = np.max(r)

    # Cut data at max_range
    pol_data_cut = deepcopy(pol_data[:, r < max_range])
    r = r[r < max_range]

    # Set masked pixels to nan
    pol_data = pol_data_cut.filled(np.nan)

    # One specificity of using the kd-tree is that we need to pad the array
    # with nans at large ranges and angles smaller and larger
    pol_data_cut = np.pad(pol_data_cut, pad_width=((1, 1), (0, 1)),
                          mode='constant', constant_values=np.nan)

    # Get angles of radar data
    if is_ppi:
        theta = radar_sweep.azimuth['data']
    else:
        theta = radar_sweep.elevation['data']

    # We need to pad theta and r as well
    theta = np.hstack([np.min(theta) - 0.1, theta, np.max(theta) + 0.1])
    r = np.hstack([r, np.max(r) + 0.1])

    r_grid_p, theta_grid_p = np.meshgrid(r, theta)

    # Generate regular cartesian grid
    if is_ppi:
        x_vec = np.arange(-max_range - cart_res,
                          max_range + cart_res, cart_res)
        y_vec = np.arange(-max_range - cart_res,
                          max_range + cart_res, cart_res)
    else:
        x_vec = np.arange(min(
            [(max_range-cart_res)*np.cos(np.radians(np.max(theta))), 0]),
                          max_range+cart_res, cart_res)

        y_vec = np.arange(0, max_range + cart_res, cart_res)

    x_grid_c, y_grid_c = np.meshgrid(x_vec, y_vec)

    if is_ppi:
        theta_grid_c = np.degrees(np.arctan2(-x_grid_c, -y_grid_c) + np.pi)
        r_grid_c = (np.sqrt(x_grid_c**2 + y_grid_c**2))
    else:
        theta_grid_c = np.degrees(-(np.arctan2(x_grid_c,
                                               y_grid_c) - np.pi / 2))
        E = get_earth_radius(radar_sweep.latitude['data'])
        r_grid_c = (np.sqrt((E * KE * np.sin(np.radians(theta_grid_c)))**2 +
                            2 * E * KE * y_grid_c + y_grid_c ** 2)
                    - E * KE * np.sin(np.radians(theta_grid_c)))

    if not mapping:
        # Kd-tree construction and query
        kdtree = spatial.cKDTree(np.vstack((r_grid_p.ravel(),
                                            theta_grid_p.ravel())).T)
        _, mapping_idx = kdtree.query(np.vstack((r_grid_c.ravel(),
                                                 theta_grid_c.ravel())).T, k=1)

        mapping = {'idx': mapping_idx, 'max_range': max_range, 'res': cart_res,
                   'is_ppi': is_ppi, 'dim_pol': pol_data.shape}

    cart_data = pol_data_cut.ravel()[mapping['idx']]
    cart_data = np.reshape(cart_data, x_grid_c.shape)

    return (x_vec, y_vec), cart_data, mapping


def remap_to_polar(radar_sweep, x, bottom_ml, top_ml, tol=1.5, interp=True):
    '''
    This routine converts the ML in Cartesian coordinates back to polar
    coordinates.

    Inputs:
        radar_sweep : Radar
            A pyart radar instance containing the radar data in polar
            coordinates for a single sweep
        x: array of floats
            The horizontal distance in Cartesian coordinates.
        bottom_ml: array of floats
            Bottom of the ML detected in Cartesian coordinates.
        top_ml: array of floats
            Top of the ML detected on Cartesian coordinates.
        tol : float, optional
            Angular tolerance in degrees that is used when mapping elevation
            angles computed on the Cartesian image to the original angles in
            the polar data.
        interp : bool, optional
            Whether or not to interpolate the ML in polar coordinates (fill holes)

    Outputs:
        (theta, r) : tuple of elevation angle and range corresponding to the
                     polar coordinates
        (bottom_ml, top_ml) : tuple of ml bottom and top ranges for every
                              elevation angle theta
        map_ml_pol : a binary map of the ML in polar coordinates
    '''
    # This routine converts the ML in cartesian coordinates back to polar
    # coordinates

    # Get ranges of radar data
    r = radar_sweep.range['data']
    dr = r[1]-r[0]

    # Get angles of radar data
    theta = radar_sweep.elevation['data']

    # Vectors to store the heights of the ML top and bottom and matrix for the
    # map
    map_ml_pol = np.zeros((len(theta), len(r)))
    bottom_ml_pol = np.zeros(len(map_ml_pol)) + np.nan
    top_ml_pol = np.zeros(len(map_ml_pol)) + np.nan

    if np.sum(np.isfinite(bottom_ml)) > 0:
         # Convert cartesian to polar

        # Get ranges of all pixels located at the top and bottom of cartesian
        # ML
        theta_bottom_ml = np.degrees(-(np.arctan2(x, bottom_ml) - np.pi / 2))
        E = get_earth_radius(radar_sweep.latitude['data'])  # Earth radius
        r_bottom_ml = (np.sqrt((E * KE * np.sin(np.radians(theta_bottom_ml)))**2 +
                               2 * E * KE * bottom_ml + bottom_ml ** 2)
                       - E * KE * np.sin(np.radians(theta_bottom_ml)))

        theta_top_ml = np.degrees(- (np.arctan2(x, top_ml) - np.pi / 2))
        E = get_earth_radius(radar_sweep.latitude['data'])  # Earth radius
        r_top_ml = (np.sqrt((E * KE * np.sin(np.radians(theta_top_ml))) ** 2 +
                            2 * E * KE * top_ml + top_ml ** 2) -
                    E * KE * np.sin(np.radians(theta_top_ml)))

        idx_r_bottom = np.zeros((len(theta))) * np.nan
        idx_r_top = np.zeros((len(theta))) * np.nan

        for i, t in enumerate(theta):
            # Find the pixel at the bottom of the ML with the closest angle
            # to theta
            idx_bot = np.nanargmin(np.abs(theta_bottom_ml - t))

            if np.abs(theta_bottom_ml[idx_bot] - t) < tol:
                # Same with pixel at top of ml
                idx_top = np.nanargmin(np.abs(theta_top_ml - t))
                if np.abs(theta_top_ml[idx_top] - t) < tol:

                    r_bottom = r_bottom_ml[idx_bot]
                    r_top = r_top_ml[idx_top]

                    idx_aux = np.where(r >= r_bottom)[0]
                    if idx_aux.size > 0:
                        idx_r_bottom[i] = idx_aux[0]

                    idx_aux = np.where(r >= r_top)[0]
                    if idx_aux.size > 0:
                        idx_r_top[i] = idx_aux[0]
        if interp:
            if np.sum(np.isfinite(idx_r_bottom)) >= 4:
                idx_valid = np.where(np.isfinite(idx_r_bottom))[0]
                idx_nan = np.where(np.isnan(idx_r_bottom))[0]
                bottom_ml_fill = InterpolatedUnivariateSpline(
                    idx_valid, idx_r_bottom[idx_valid], ext=1)(idx_nan)
                bottom_ml_fill[bottom_ml_fill == 0] = -9999
                idx_r_bottom[idx_nan] = bottom_ml_fill

            if np.sum(np.isfinite(idx_r_top)) >= 4:
                idx_valid = np.where(np.isfinite(idx_r_top))[0]
                idx_nan = np.where(np.isnan(idx_r_top))[0]
                top_ml_fill = InterpolatedUnivariateSpline(
                    idx_valid, idx_r_top[idx_valid], ext=1)(idx_nan)
                top_ml_fill[top_ml_fill == 0] = -9999
                idx_r_top[idx_nan] = top_ml_fill
        else:
            idx_r_bottom[np.isnan(idx_r_bottom)] = -9999
            idx_r_top[np.isnan(idx_r_top)] = -9999

        idx_r_bottom = idx_r_bottom.astype(int)
        idx_r_top = idx_r_top.astype(int)

        for i in range(len(map_ml_pol)):
            if idx_r_bottom[i] != -9999 and idx_r_top[i] != -9999:
                r_bottom_interp = min([len(r), idx_r_bottom[i]])*dr
                bottom_ml_pol[i] = r_to_h(E, r_bottom_interp, theta[i])

                r_top_interp = min([len(r), idx_r_top[i]])*dr
                top_ml_pol[i] = r_to_h(E, r_top_interp, theta[i])

                # check that data has plausible values
                if (bottom_ml_pol[i] > MAXHEIGHT_ML or
                        bottom_ml_pol[i] < MINHEIGHT_ML or
                        top_ml_pol[i] > MAXHEIGHT_ML or
                        top_ml_pol[i] < MINHEIGHT_ML or
                        bottom_ml_pol[i] >= top_ml_pol[i]):
                    bottom_ml_pol[i] = np.nan
                    top_ml_pol[i] = np.nan
                else:
                    map_ml_pol[i, 0:idx_r_bottom[i]] = 1
                    map_ml_pol[i, idx_r_bottom[i]:idx_r_top[i]] = 3
                    map_ml_pol[i, idx_r_top[i]:] = 5

    return (theta, r), (bottom_ml_pol, top_ml_pol), map_ml_pol


def compute_iso0(radar, ml_top, iso0_field='height_over_iso0'):
    """
    Estimates the distance respect to the freezing level of each range gate
    using the melting layer top as a proxy

    Parameters
    ----------
    radar : Radar
        Radar object
    ml_top : 1D array
        The height of the melting layer at each ray
    iso0_field : str
        Name of the iso0 field.

    Returns
    -------
    iso0_dict : dict
        A dictionary containing the distance respect to the melting layer
        and metadata

    """
    iso0_data = np.ma.masked_all((radar.nrays, radar.ngates))
    for ind_ray in range(radar.nrays):
        iso0_data[ind_ray, :] = (
            radar.gate_altitude['data'][ind_ray, :]-ml_top[ind_ray])

    iso0_dict = get_metadata(iso0_field)
    iso0_dict['data'] = iso0_data

    return iso0_dict


def create_ml_obj(radar, ml_pos_field='melting_layer_height'):
    """
    Creates a radar-like object that will be used to contain the melting layer
    top and bottom

    Parameters
    ----------
    radar : Radar
        Radar object
    ml_pos_field : str
        Name of the melting layer height field

    Returns
    -------
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray

    """
    ml_obj = deepcopy(radar)

    # modify original metadata
    ml_obj.range['data'] = np.array([0, 1], dtype='float64')
    ml_obj.ngates = 2

    ml_obj.gate_x = np.zeros((ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_y = np.zeros((ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_z = np.zeros((ml_obj.nrays, ml_obj.ngates), dtype=float)

    ml_obj.gate_longitude = np.zeros(
        (ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_latitude = np.zeros(
        (ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_altitude = np.zeros(
        (ml_obj.nrays, ml_obj.ngates), dtype=float)

    # Create field
    ml_obj.fields = dict()
    ml_dict = get_metadata(ml_pos_field)
    ml_dict['data'] = np.ma.masked_all((ml_obj.nrays, ml_obj.ngates))
    ml_obj.add_field(ml_pos_field, ml_dict)

    return ml_obj


def detect_ml(radar_rhi, refl_field='Zh', rhohv_field='Rhohv', max_range=15, detect_threshold=0.02, check_min_length=True, fill_value=-9999.):
    coords_c, refl_field_c, mapping = polar_to_cartesian(
            radar_rhi, refl_field, max_range=max_range)
    coords_c, rhohv_field_c, _ = polar_to_cartesian(
         radar_rhi, rhohv_field, mapping=mapping)
    cart_res = mapping['res']

    # Get Zh and Rhohv images
    refl_im = pyart.retrieve.detect_ml._normalize_image(refl_field_c, *ZH_IM_BOUNDS)
    rhohv_im = pyart.retrieve.detect_ml._normalize_image(rhohv_field_c, *RHOHV_IM_BOUNDS)

    # Combine images
    comb_im = (1 - rhohv_im) * refl_im
    comb_im[np.isnan(comb_im)] = 0.

    # Get vertical gradient
    size_filt = np.floor(SIZEFILT_M / cart_res).astype(int)
    gradient = pyart.retrieve.ml._gradient_2D(pyart.retrieve.ml._mean_filter(comb_im, (size_filt, size_filt)))
    gradient_z = gradient['Gy']
    gradient_z[np.isnan(rhohv_field_c)] = np.nan

    bottom_ml, top_ml = pyart.retrieve.ml._process_map_ml(
        gradient_z, rhohv_field_c, detect_threshold, *RHOHV_VALID_BOUNDS)

    # Restrict gradient using conditions on medians
    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    if not np.isnan(median_bot_height):
        gradient_z[0:np.floor(LOWMLBOUND *
                              median_bot_height).astype(int), :] = np.nan
    if not np.isnan(median_top_height):
        gradient_z[np.floor(UPMLBOUND *
                            median_top_height).astype(int):, :] = np.nan

    # Identify top and bottom of ML with restricted gradient
    bottom_ml, top_ml = pyart.retrieve.ml._process_map_ml(
        gradient_z, rhohv_field_c, detect_threshold, *RHOHV_VALID_BOUNDS)
    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    thickness = top_ml - bottom_ml
    bad_pixels = ~np.isnan(thickness)
    bad_pixels[bad_pixels] &= (
        bad_pixels[bad_pixels] > MAXTHICKNESS_ML/cart_res)
    top_ml[bad_pixels] = np.nan
    bottom_ml[bad_pixels] = np.nan
    top_ml[np.isnan(bottom_ml)] = np.nan
    bottom_ml[np.isnan(top_ml)] = np.nan

    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    mid_ml = (median_top_height + median_bot_height) / 2

    # Check if ML is valid
    # 1) check if median_bot_height and median_top_height are defined
    if np.isnan(median_bot_height + median_top_height):
        invalid_ml = True
    else:
        invalid_ml = False
        # 2) Check how many values in the data are defined at the height of the
        # ML
        line_val = rhohv_field_c[np.int(mid_ml), :]

        # Check if ML is long enough
        if check_min_length:
            # the condition is that the ml is at least half as
            # long as the length of valid data at the ml height
            if np.logical_and(sum(np.isfinite(top_ml)) < 0.5,
                              sum(np.isfinite(line_val))):
                invalid_ml = True
            
    map_ml = np.zeros(gradient_z.shape)

    # 1 = below ML, 3 = in ML, 5 =  above ML
    mdata_ml = {'BELOW':1, 'INSIDE':3, 'ABOVE':5}
    # If ML is invalid, just fill top_ml and bottom_ml with NaNs
    if invalid_ml: 
        top_ml = np.nan * np.zeros((gradient_z.shape[1]))
        bottom_ml = np.nan * np.zeros((gradient_z.shape[1]))
    else:
        for j in range(0, len(top_ml) - 1):
            if(not np.isnan(top_ml[j]) and not np.isnan(bottom_ml[j])):
                map_ml[np.int(top_ml[j]):, j] = mdata_ml['BELOW']
                map_ml[np.int(bottom_ml[j]):np.int(top_ml[j]), j] = mdata_ml['INSIDE']
                map_ml[0:np.int(bottom_ml[j]), j] = mdata_ml['ABOVE']

    # create dictionary of output ml

    # Cartesian coordinates
    ml_cart = {}
    ml_cart['data'] = np.array(map_ml)
    ml_cart['x'] = coords_c[0]
    ml_cart['z'] = coords_c[1]

    ml_cart['bottom_ml'] = np.array((bottom_ml) * cart_res)
    ml_cart['top_ml'] = np.array((top_ml) * cart_res)

    # Polar coordinates
    (theta, r), (bottom_ml, top_ml), map_ml = remap_to_polar(
        radar_rhi, ml_cart['x'], ml_cart['bottom_ml'], ml_cart['top_ml'],
        interp=True)
    map_ml = np.ma.array(map_ml, mask=map_ml == 0, fill_value=fill_value)
    bottom_ml = np.ma.masked_invalid(bottom_ml)
    top_ml = np.ma.masked_invalid(top_ml)

    ml_pol = {}
    ml_pol['data'] = map_ml
    ml_pol['theta'] = theta
    ml_pol['range'] = r
    ml_pol['bottom_ml'] = bottom_ml
    ml_pol['top_ml'] = top_ml

    output = {}
    output['ml_cart'] = ml_cart
    output['ml_pol'] = ml_pol
    output['ml_exists'] = not invalid_ml
    
    
    all_ml = [output]
    ml_field = 'ML'
    ml_pos_field = 'MLHeight'
    
    ml_dict = get_metadata(ml_field)
    ml_dict.update({'_FillValue': 0})
    ml_obj = create_ml_obj(radar_rhi, ml_pos_field)

    ml_data = np.ma.masked_all(
        (radar_rhi.nrays, radar_rhi.ngates), dtype=np.uint8)
    for sweep in range(radar_rhi.nsweeps):
        sweep_start = radar_rhi.sweep_start_ray_index['data'][sweep]
        sweep_end = radar_rhi.sweep_end_ray_index['data'][sweep]
        ml_obj.fields[ml_pos_field]['data'][sweep_start:sweep_end+1, 0] = (
            all_ml[sweep]['ml_pol']['bottom_ml'])
        ml_obj.fields[ml_pos_field]['data'][sweep_start:sweep_end+1, 1] = (
            all_ml[sweep]['ml_pol']['top_ml'])
        ml_data[sweep_start:sweep_end+1, :] = all_ml[sweep]['ml_pol']['data']
    ml_dict['data'] = ml_data

    valid_values = ml_obj.fields[ml_pos_field]['data'][:, 1].compressed()

    get_iso0 = True
    iso0_field='iso0'
    # get the iso0
    iso0_dict = None
    if get_iso0:
        iso0_dict = compute_iso0(
            radar_rhi, ml_obj.fields[ml_pos_field]['data'][:, 1],
            iso0_field=iso0_field)
    
    return output, ml_obj, iso0_dict


#%% constants & functions for hydrometeor classification
SNR_THR = -10.
RHOHV_THR = 0.6
ZDR_BIAS = 0.

# load centroids needed for hydrometeor classification
centroids = sio.loadmat('/ltedata/0_Software/Prog_com/Lib_python/hydro_classif/MXPol_centroids/Centroids_sqeuc_averaged_X_c2.mat')['Centroid']	# updated path Feb 2026 
dummy_centroids = [99999, 99999, 99999, 99999, 99999]
centroids = np.insert(centroids,5,dummy_centroids,0)
perm = [1,0,2,4,3,5,6,8,7] # change the order to make it corresponds to Jordi
centroids = centroids[perm]  


def classify_hm(fname, scantype='sector', rain_flag=False, tempdata=None):
    warnings.filterwarnings('ignore', message='.*TBB.*')

    """function to run hydrometeor classification with demixing (Besic et al. 2018) using the implementation in pyart-mch"""
    if type(fname) == str:
        radar = pyart.io.read_cfradial(fname)
    else:
        radar = fname

    # check if length of file is correct (81 data points in time)
    t_scan = len(radar.time['data'])
    if t_scan > scantimes_MXPol[scantype]*1.5:
        print(f"too many data points (time is {int(t_scan)} s instead of {scantimes_MXPol[scantype]}) - corrupted file")
        return

    try:
        filter_radar(radar)
    except Exception as e:
        print(f"Error while processing {fname.split('/')[-1].split('.')[0]}: {e}, skipping...")
        return

    # add Zdr correction if not already present
    if not 'Zdr_corrected' in radar.fields.keys():
        try:
            radar = Zdr_corr_pyart(radar)
        except Exception as e:
            print(f"  Error while adding Zdr correction: {e}, skipping...")
            return

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
                return

    # correct Zdr
    if not 'Zdr_corrected' in radar.fields.keys():
        zdr_corr = get_Zdr_corr(os.path.basename(fname))[0]
        zdr = radar.fields['Zdr']['data'].copy() - zdr_corr
        radar.add_field_like('Zdr_corrected', 'Zdr_corrected', zdr, replace_existing=True)
        
    # define iso0 depending on the rain flag based on MIRA-temperature data in the future)
    if rain_flag:
        if tempdata is None:
            print(f"MIRA/temperature data required if rain flag == True")
            return
        try:
            time_radar = pd.to_datetime(radar.time['data'], unit='s')
            T_surface = (tempdata.TEMP[:, 0]) - tempdata.range[0] * MIRA_lapserate
            zerodeg_height = (T_surface / -MIRA_lapserate).interpolate_na(dim='time', method='linear').ffill(dim='time').bfill(dim='time')
            zerodeg_altitude = zerodeg_height + radar_altitude
            iso0 = zerodeg_altitude.sel(time=time_radar, method='nearest').values.reshape(-1, 1) # reshape to (n_times, 1) for broadcasting
        except Exception as e:
            print(f"Error while calculating 0deg-isotherm: {e}")
            return

    # if rain_flag and (scantype == 'RHI'): 
    #     # if it rains, define iso0 as the top of the melting layer (detected using Wolfensberger and Berne 2018, see ml_detection.py script)
    #     _, ml_obj, _ = detect_ml(radar)
    #     iso0 = np.nanmean(ml_obj.fields['MLHeight']['data'][:,1])
    # else: 
    #     # if it snows, set the iso0 below the ground
    #     iso0 = radar_altitude - 10
        
        # create the field containing height over iso0
        height_over_iso0 = radar.gate_altitude['data'] - iso0 # relative height above iso0, + means above, - below
        height_over_iso0_dic = {'long_name':'height above the zero degree celsius isotherm', 'units':'m','valid_min': -30000, 'valid_max': 30000, 'data': height_over_iso0}
        radar.add_field('height_over_iso0',height_over_iso0_dic)

    # create gatefilter to exclude gates with low SNR and Rhohv for the classification
    gf = pyart.correct.GateFilter(radar)
    gf.exclude_below('SNRh_full', SNR_THR)
    gf.exclude_below('Rhohv_full', RHOHV_THR)

    # set values below SNR and Rhohv threshold to nans
    for v in radar.fields.keys() - {'sCC', 'sPowH', 'sPowV', 'Zdr_full'}:
        if not v.endswith("_full"):
            continue
        else:
            V = radar.fields[v]['data'].copy()
            fieldname = v.split('_full')[0] + '_HC'
            V = np.ma.masked_where(gf.gate_excluded, V)
            radar.add_field_like(v, fieldname, V, replace_existing=True)
        
    # create empty container for entropy
    radar.add_field('hydroclass_entropy', {'long_name':'Hydroclass entropy', 'units': '','valid_min':-1, 'valid_max':1, 'data':radar.fields['Zdr']['data']*np.nan})

    # semisupervised classification from PyArt - MCH
    hydro_classif = pyart.retrieve.hydroclass_semisupervised(radar, refl_field='Zh', zdr_field='Zdr_corrected', rhv_field='Rhohv', 
                                                            kdp_field='Kdp', iso0_field='height_over_iso0', entropy_field='hydroclass_entropy', compute_entropy=True,
                                                            output_distances=True, mass_centers=centroids, temp_ref='height_over_iso0')
    
    # add hydrometeor centroids to metadata
    radar.metadata['CentroidsAG'] = centroids[0,:]
    radar.metadata['CentroidsCR'] = centroids[1,:]
    radar.metadata['CentroidsLR'] = centroids[2,:]
    radar.metadata['CentroidsRP'] = centroids[3,:]
    radar.metadata['CentroidsRN'] = centroids[4,:]
    radar.metadata['CentroidsVI'] = centroids[5,:]
    radar.metadata['CentroidsWS'] = centroids[6,:]
    radar.metadata['CentroidsMH'] = centroids[7,:]
    radar.metadata['CentroidsIH'] = centroids[8,:]
    radar.metadata['CentroidsColumnNames'] = 'ZH, ZDR, KDP, RhoHV, height_above_iso0'

    # information on the method
    radar.metadata['ClassificationMethod'] = 'Besic et al. AMT 2016 with demixing of Besic et al. AMT 2018'

    # adding the hydrometeor classification to the radar object
    for _, key in enumerate(hydro_classif.keys()):
        radar.add_field(key, hydro_classif[key])

    for key in ['proportion_AG', 'proportion_CR', 'proportion_LR', 'proportion_RP', 'proportion_RN', 'proportion_VI', 'proportion_WS', 'proportion_MH', 'proportion_IH']:
        radar.fields[key]['units']='%'
    
    return radar


def plot_classification(radar, hcname='hydro', savepath='tmp', scantype='sector', xlim=None, ylim=None):
    """small function to plot the results - dominant hydrometeor class """
    if scantype == 'sector':
        xlim = [-6, 16]
        ylim = [-3, 16]
    if scantype == 'RHI':
        xlim = [-15, 15]
        ylim = [0, 12]

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    display = pyart.graph.RadarDisplay(radar)
    cmap = mpl.colors.ListedColormap(["white", "blue", "deepskyblue", "green", "gold", "orange", "red", "magenta", "purple","brown"])
    norm = mpl.colors.BoundaryNorm(np.arange(0,11), cmap.N) 
    cbarticks = np.arange(0,10)+.5
    legend = ['', 'AG', 'CR', 'LR', 'RP', 'RN', 'VI', 'WS', 'MH', 'IH'] 
    # AG: aggregates, CR: crystals, LR: light rain, RP: rimed particles, RN: rain, VI: vertically-aligned ice, WS: wet snow, MH: melting hail, IH: ice hail
    if scantype == 'sector':
        display.plot_ppi('Zh', vmin=-15, vmax=45, ax=axs[0], cmap='viridis_r')
        display.plot_ppi(hcname, cmap=cmap, norm=norm, ticks=cbarticks, ticklabs=legend, ax=axs[1])
    if scantype == 'RHI':
        display.plot_rhi('Zh', vmin=-15, vmax=45, ax=axs[0], cmap='viridis_r')
        display.plot_rhi(hcname, cmap=cmap, norm=norm, ticks=cbarticks, ticklabs=legend, ax=axs[1])
    [ax.set_xlim(xlim) for ax in axs]
    [ax.set_ylim(ylim) for ax in axs]
    plt.tight_layout()
    if savepath != 'tmp':
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
    

# %% create plots for sector scans
if __name__ == '__main__':
    dates = generate_date_range(start_date, '2024-12-25')
    # dates = generate_date_range('2024-12-10', '2024-12-25')
    overwrite = False
    for date in dates:
        rain_flag = True
        mira = None
        t_date = time.time()
        year, month, day = date.year, date.month, date.day
        files_all = sorted(glob.iglob(os.path.join(base_dir, dirs['MXPol'], f"{year:04d}/{month:02d}/{day:02d}/XPOL-{year:04d}{month:02d}{day:02d}*.nc")))
        files_sector = [f for f in files_all if 'sector' in f]
        files_RHI = [f for f in files_all if 'RHI' in f]

        try:
            mira = xr.open_dataset(f"{dirs['MIRA']}/{year}{month:02}{day:02}_small.nc")
        except Exception as e:
            print(f"no MIRA data for {year}-{month:02}-{day:02}: {e}\nskipping this day...")
            continue

        if not files_sector and not files_RHI:
            continue
        
        if files_sector:
            print(f"\nMaking {len(files_sector)} sector scan plots for {year}-{month:02}-{day:02}, starting {pd.to_datetime(time.time(), unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
            for fname in files_sector:
                try:
                    name = fname.split('/')[-1].split('.')[0]
                    outname = os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/{year:04d}/{month:02d}/{day:02d}/{name}.nc")
                    figname = os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/plots/{year:04d}/{month:02d}/{day:02d}/{name}_hydroclassif.png")
                    os.makedirs(os.path.dirname(outname), exist_ok=True)
                    os.makedirs(os.path.dirname(figname), exist_ok=True)
                except Exception as e:
                    print(e)
            
                if os.path.exists(outname) and not overwrite:
                    print(f"HC file for {os.path.basename(outname)} already exists, skipping...")
                    continue

                try:
                    radar_hc = classify_hm(fname, rain_flag=True, tempdata=mira)
                except Exception as e:
                    print(f"Error occurred while processing {os.path.basename(fname)}: {e}, skipping...")
                    continue
                if radar_hc is None:
                    continue

                plot_classification(radar_hc, savepath=figname, scantype='sector')
                pyart.io.write_cfradial(outname, radar_hc)
                del radar_hc
                gc.collect()
            t_end = time.time()
            print(f"Finished making sector scan plots & .nc-generation for {year}-{month:02}-{day:02}, time taken: {t_end-t_date:.2f} seconds\n")

        if files_RHI:
            print(f"Making {len(files_RHI)} RHI plots for {year}-{month:02}-{day:02}, starting {pd.to_datetime(time.time(), unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
            for fname in files_RHI:
                try:
                    name = fname.split('/')[-1].split('.')[0]
                    outname = os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/{year:04d}/{month:02d}/{day:02d}/{name}.nc")
                    figname = os.path.join(base_dir, dirs['MXPol'], f"hydroclassif/plots/{year:04d}/{month:02d}/{day:02d}/{name}_hydroclassif.png")
                except Exception as e:
                    print(e)

                if os.path.exists(outname) and not overwrite:
                    print(f"HC file for {os.path.basename(outname)} already exists, skipping...")
                    continue
                try:
                    radar_hc = classify_hm(fname, rain_flag=True, tempdata=mira, scantype='RHI')
                except Exception as e:
                    print(f"Error occurred while processing {os.path.basename(fname)}: {e}, skipping...")
                    continue
                if radar_hc is None:
                    continue

                plot_classification(radar_hc, savepath=figname, scantype='RHI')
                pyart.io.write_cfradial(outname, radar_hc)
                del radar_hc
                gc.collect()
            t_end = time.time()
            print(f"Finished making RHI plots & .nc-generation for {year}-{month:02}-{day:02}, time taken: {t_end-t_date:.2f} seconds\n")
