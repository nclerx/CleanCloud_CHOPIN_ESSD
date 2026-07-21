#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 15 Oct 09:30:15 2025
"""
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src import os, np, pd, da, zarr, xr, plt, LogNorm, time
from src.constants_input import default_variable_maps, radar_altitude, zarr_variables, doppler_dim_map, doppler_dims, time_bins_radars, ranges
from src.radar_processing import load_zenithdata
from src.utils import find_temperature_altitude_xr, format_elapsed, align_times_to_grid, safe_reindex, safe_reindex_dim

from typing import List, Dict, Optional
import getpass

def generate_zarr_encodings(variables):
    """create dictionary for zarr encodings"""
    encodings = {}
    for name, props in variables.items():
        if props.get("dtype") == bool:
            encodings[name] = {"dtype": bool, "_FillValue": False}
            continue

        if props.get("dtype") == "datetime":
            encodings[name] = {"dtype": np.int32, 
                               "units": props.get("units", "seconds since 1970-01-01"), 
                               "calendar": "proleptic_gregorian",
                               "_FillValue": np.iinfo(np.int32).min,
                               }
            continue

        if "precision" in props:
            vmin, vmax = props["range"]
            precision = props["precision"]
            n_levels = int(np.ceil((vmax - vmin) / precision)) + 1

            # choose integer type
            if n_levels <= 256:
                dtype = np.uint8
            elif n_levels <= 65536:
                dtype = np.uint16
            else:
                dtype = np.uint32

            # reserve the top integer value as a fill sentinel for integer encodings
            fillval = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 0
            encodings[name] = {
                "dtype": dtype,
                "scale_factor": precision,
                "add_offset": vmin,
                "_FillValue": int(fillval),
            }

        else:
            dtype = props.get("dtype", np.float32)
            # choose a sensible default fill value for integer types
            fillval = 0
            if np.issubdtype(dtype, np.integer):
                fillval = np.iinfo(dtype).max
            encodings[name] = {"dtype": dtype, "_FillValue": int(fillval)}

    return encodings


def create_empty_zarr(path, radar, time_bins, range_bins, encodings, drop_spectral=False):
    """create empty zarr dataset with specified dimensions and encodings"""
    coords = {"time": time_bins, "range": range_bins}
    doppler_dim = doppler_dim_map.get(radar)
    if doppler_dim and not drop_spectral:
        coords["doppler"] = np.array(doppler_dims[radar])

    data_vars = {}
    for var, props in zarr_variables[radar].items():
        dims = [("doppler" if d == doppler_dim_map[radar] else d) for d in props["dims"]]
        if drop_spectral and "doppler" in dims:
            continue

        shape = tuple(len(coords[d]) for d in dims)
        enc = encodings.get(var, {})
        dtype = enc.get("dtype", np.float32)
        fill_value = enc.get("_FillValue", np.nan if np.issubdtype(dtype, np.floating) else 0)
        data_vars[var] = (dims, da.full(shape, fill_value, dtype=dtype, chunks="auto"))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "campaign": "CLEANCLOUD_CHOPIN_2024",
            "creation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "created_by": getpass.getuser(),
            "radar": radar
        },
    )

    enc_to_zarr = {}
    for var, enc in encodings.items():
        if var not in data_vars:
            continue
        enc_to_zarr[var] = {
            "dtype": enc.get("dtype", np.float32),
            "_FillValue": enc.get("_FillValue"),
        }

        if "scale_factor" in enc:
            ds[var].attrs["scale_factor"] = enc["scale_factor"]
        if "add_offset" in enc:
            ds[var].attrs["add_offset"] = enc["add_offset"]

    ds.to_zarr(path, mode="w", encoding=enc_to_zarr, consolidated=True)
    return ds


def preprocess_for_zarr(ds: xr.Dataset, encodings: Dict[str, Dict]) -> xr.Dataset:
    """
    clip, round, and cast dataset variables according to encoding specifications.
    - applies scale_factor and add_offset
    - clips to representable range
    - converts to the target dtype
    """
    ds_proc = ds.copy()
    for var, enc in encodings.items():
        if var not in ds_proc:
            continue
        
        dtype = enc.get("dtype", np.float32)

        if dtype == bool:
            ds_proc[var] = ds_proc[var].fillna(False).astype(bool)
            continue

        da = ds_proc[var]
        fill = enc.get('_FillValue', 0)
        nan_mask = da.isnull()

        if 'scale_factor' in enc and 'add_offset' in enc and np.issubdtype(dtype, np.integer):
            scale = float(enc['scale_factor'])
            offset = float(enc['add_offset'])
            
            # Apply encoding transformation first
            da = (da - offset) / scale
            da = da.round()
            
            # Then clip in ENCODED space (0 to max-1, reserve max for fill)
            max_val = np.iinfo(dtype).max - 1
            da = da.clip(0, max_val)

        # Convert to target dtype
        da = da.astype(dtype)
        
        # Apply fill value where NaNs were
        da = da.where(~nan_mask, fill)

        ds_proc[var] = da

    return ds_proc


def campaign_to_zarr(
    dates: List[pd.Timestamp],
    precip_times: pd.DatetimeIndex,
    dirs: Dict[str, str],
    DFRs,
    att_fn: str,
    zarrdir: str,
    encodings: Dict[str, Dict],
    BASTAmode: str = '25m',
    radars: List[str] = ['MIRA', 'BASTA', 'MXPol', 'peaktree'],
    resample_time: bool = True,
    resample_time_interval: str = '30s',
    resample_range: bool = True,
    interp_method='nearest',
    ERA: bool = True,
    drop_spectral=False
):
    """process/prepare campaign data and append to zarr datasets for each radar"""
    if not resample_time and not resample_range:
        print('Skipping DFR calculations, datasets not aligned', flush=True)

    for date in dates:
        ts_start = pd.Timestamp.now()
        day_start = pd.Timestamp(date).normalize()
        day_end = day_start + pd.Timedelta(days=1)

        if drop_spectral:
            periods = [(day_start, day_end)]
        else:
            periods = [(day_start + pd.Timedelta(hours=h), day_start + pd.Timedelta(hours=h+1)) for h in range(24)]
        
        for period_start, period_end in periods:
            print(f"\nProcessing data for {date.strftime('%Y-%m-%d')} between {period_start.strftime('%H:%M')} and {period_end.strftime('%H:%M')}", flush=True)

            try:
                datasets = load_zenithdata(
                    date, dirs, DFRs, att_fn, BASTAmode,
                    useMXPol=('MXPol' in radars),
                    useMIRA=('MIRA' in radars),
                    useBASTA=('BASTA' in radars),
                    peakTree=('peaktree' in radars),
                    resample_time=resample_time,
                    target_time=resample_time_interval,
                    resample_range=resample_range,
                    interp_method=interp_method,
                    ERA=True,
                    drop_spectral=drop_spectral,
                    start_time=period_start,
                    end_time=period_end
                )
            except Exception as e:
                print(f"Failed to load data for {date.strftime('%Y-%m-%d')}: {e}", flush=True)
                continue

            # correctly add SNR-mask (naming is inversed at the moment i.e. _full contains the filtered data)
            if 'MXPol' in datasets:
                full_vars = [var for var in datasets['MXPol'].data_vars if '_full' in var]
                SNRmask = (datasets['MXPol']['SNRh'] > -5) & (datasets['MXPol']['SNRv'] > -5)
                vars_to_mask = [var for var in datasets['MXPol'].data_vars if var not in full_vars and datasets['MXPol'][var].dims == ('time', 'range')]
                for var in vars_to_mask:
                    datasets['MXPol'][var] = datasets['MXPol'][var].where(SNRmask)

            # correct MIRA reflectivity and LDR values (following MBP mail 28.05.2026) 
            if 'MIRA' in datasets: 
                datasets['MIRA']['Z'] = datasets['MIRA']['Z'] + 0.89 
                datasets['MIRA']['LDR'] = datasets['MIRA']['LDRg'] - 0.89 

            if 'BASTA' in datasets:
                datasets['BASTA']['reflectivity_attn_DFR_corrected'] = datasets['BASTA']['reflectivity_attn_DFR_corrected'] + 0.89

            # compute DFRs (if datasets have the same time & range resolutions only)
            DFR_XKa = DFR_XW = DFR_KaW = None
            if resample_time and resample_range:
                try:
                    if 'MXPol' in datasets and 'MIRA' in datasets:
                        DFR_XKa = datasets['MXPol']['Zh_full'] - datasets['MIRA']['Z']
                        datasets['MIRA']['DFR_XKa'] = datasets['MXPol']['DFR_XKa'] = DFR_XKa
                    if 'MIRA' in datasets and 'BASTA' in datasets:
                        DFR_KaW = datasets['MIRA']['Z'] - datasets['BASTA']['reflectivity_attn_DFR_corrected']
                        datasets['MIRA']['DFR_KaW'] = datasets['BASTA']['DFR_KaW'] = DFR_KaW
                    if 'MXPol' in datasets and 'BASTA' in datasets:
                        DFR_XW = datasets['MXPol']['Zh_full'] - datasets['BASTA']['reflectivity_attn_DFR_corrected']
                        datasets['MXPol']['DFR_XW'] = datasets['BASTA']['DFR_XW'] = DFR_XW
                except Exception as e:
                    print(f"    Failed to compute DFRs: {e}", flush=True)

            for radar in radars:
                if radar not in datasets:
                    print(f"No data for {radar} between {period_start.strftime('%Y-%m-%d %H:%M')} and {period_end.strftime('%Y-%m-%d %H:%M')}")
                    continue

                print(f"\n--- Preparing {radar} data for {period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%Y-%m-%d %H:%M')} at {ts_start.strftime('%H:%M:%S (%Y-%m-%d)')} ---", flush=True)
                ts_zarr = pd.Timestamp.now()
                ds = datasets[radar]
                ds = ds.chunk({'time': 3600, 'range': -1})  # rechunk for more efficient processing
                variables_map = default_variable_maps.get(radar, {})

                # peaktree dimension reduction
                if radar == 'peaktree':
                    keep_vars = list(variables_map.values())
                    ds = ds[keep_vars]
                    for dim in ['mode', 'nodes']:
                        if dim in ds.dims:
                            ds = ds.isel({dim: 0})
                    if 'vel' in ds.dims:
                        ds = ds.max(dim='vel', keep_attrs=True).compute()

                selected_vars = {}
                for canon, varname in variables_map.items():
                    if varname in ds:
                        da_var = ds[varname]
                        if 'clean_mask' in ds and {'time', 'range'} <= set(da_var.dims):
                            da_var = da_var.where(ds['clean_mask'])
                        selected_vars[canon] = da_var

                # add DFRs
                if DFR_XKa is not None and radar in ['MXPol', 'MIRA']:
                    selected_vars['DFR_XKa'] = DFR_XKa
                if DFR_XW is not None and radar in ['MXPol', 'BASTA']:
                    selected_vars['DFR_XW'] = DFR_XW
                if DFR_KaW is not None and radar in ['MIRA', 'BASTA']:
                    selected_vars['DFR_KaW'] = DFR_KaW

                # precipitation flag
                radar_times = pd.to_datetime(ds.time.values)
                if hasattr(precip_times, 'dtype') and precip_times.dtype == 'bool':
                    precip_times = precip_times.index
                nearest_precip = precip_times.searchsorted(radar_times)
                nearest_precip[nearest_precip == len(precip_times)] = len(precip_times) - 1
                diff = np.abs((radar_times - precip_times[nearest_precip]).astype('timedelta64[s]').astype(int))
                try:
                    window_seconds = int(pd.to_timedelta(resample_time_interval).total_seconds())
                except Exception:
                    window_seconds = pd.to_timedelta(resample_time_interval).seconds
                is_precip = diff <= (window_seconds / 2)
                selected_vars['is_precip'] = xr.DataArray(
                    is_precip,
                    coords={'time': ds.time},
                    dims=['time'],
                    name='is_precip'
                )

                if not selected_vars:
                    print(f"    No variables to write for {radar}, skipping.", flush=True)
                    continue

                daily_ds = xr.Dataset(selected_vars).sel(time=slice(period_start, period_end))

                # ERA zdi interpolation
                if ERA and 'ERA' in datasets:
                    try:
                        zdi = find_temperature_altitude_xr(datasets['ERA'], alt_range=(0, 15e3),
                                                        radar_altitude=radar_altitude, temps=[0])
                        zdi = zdi.interp(time=daily_ds['time'], method='nearest')
                        daily_ds['zdi'] = zdi['z0degC']
                    except Exception as e:
                        print(f"    Failed to compute 0°C isotherm altitude: {e}", flush=True)

                # remove duplicate times
                if 'time' in daily_ds:
                    daily_ds = daily_ds.sortby('time')
                    _, idx = np.unique(daily_ds['time'], return_index=True)
                    daily_ds = daily_ds.isel(time=idx)

                if not resample_time: # make sure to map times to zarr-times
                    full_target = time_bins_radars[radar]
                    
                    interval_mask = (full_target >= period_start) & (full_target <= period_end)
                    target_time = full_target[interval_mask]

                    if len(target_time) == 0:
                        print(f"    Warning: no target_time for {radar} on {date}", flush=True)
                    else:
                        time_diff = np.abs(daily_ds.time.values.astype("datetime64[ns]") - target_time.values.astype("datetime64[ns]")[:len(daily_ds.time)])
                        if len(daily_ds.time) != len(target_time) or np.any(time_diff > np.timedelta64(1, 'ms')):
                            daily_ds = align_times_to_grid(daily_ds, target_time)
                        else:
                            pass
                        
                # open zarr
                zarr_path = os.path.join(zarrdir, f"{radar}.zarr") if not radar == 'BASTA' else os.path.join(zarrdir, f"{radar}_{BASTAmode}.zarr")
                if not os.path.exists(zarr_path):
                    print(f"    Zarr file missing for {radar}, skipping.", flush=True)
                    continue
                zarr_ds = xr.open_zarr(zarr_path)

                zarr_times = zarr_ds.time.values
                overlapping_times = np.intersect1d(daily_ds.time.values, zarr_times)
                if len(overlapping_times) == 0:
                    print(f"    No overlapping times for {radar}, skipping.", flush=True)
                    zarr_ds.close()
                    continue
                
                # compute write region in zarr
                daily_ds = daily_ds.sel(time=overlapping_times)
                zarr_time_mask = np.isin(zarr_times, overlapping_times)
                if not zarr_time_mask.any():
                    print(f"    No matching times in Zarr store!", flush=True)
                    continue

                true_indices = np.where(zarr_time_mask)[0]
                start, stop = true_indices[0], true_indices[-1] + 1

                if (stop - start) != len(daily_ds.time):
                    print(f"    ERROR: Region size mismatch for {radar} on {date.strftime('%Y-%m-%d')}:", flush=True)
                    print(f"Region: slice({start}, {stop}) = {stop - start} steps", flush=True)
                    print(f"Data to write: {len(daily_ds.time)} steps", flush=True)
                    zarr_ds.close()
                    continue
                
                # drop non-time variables and missing variables
                drop_non_time = [
                    v for v in daily_ds.variables
                    if 'time' not in daily_ds[v].dims
                    and v not in ['range', 'doppler', 'time']
                ]
                ds_to_write = daily_ds.drop_vars(drop_non_time, errors='ignore')

                # reindex ranges to ensure match
                zarr_ranges = zarr_ds['range'].values
                if 'range' in ds_to_write.dims and radar in ranges:
                    tol = ranges[radar].get('tol', None)
                    ds_to_write = ds_to_write.reindex({'range': zarr_ranges}, method='nearest', tolerance=tol)

                missing_vars = set(ds_to_write.data_vars) - set(zarr_ds.data_vars)
                if missing_vars:
                    print(f"    Dropping variables not in zarr for {radar}: {missing_vars}", flush=True)
                    ds_to_write = ds_to_write.drop_vars(missing_vars)

                # doppler dimension renaming
                doppler_dim = doppler_dim_map.get(radar)
                if doppler_dim and doppler_dim != 'doppler':
                    vars_with_doppler = [var for var in ds_to_write.data_vars if doppler_dim in ds_to_write[var].dims]
                    if vars_with_doppler:
                        for var in vars_with_doppler:
                            ds_to_write[var] = ds_to_write[var].rename({doppler_dim: 'doppler'})
                        if doppler_dim in ds_to_write.coords:
                            ds_to_write = ds_to_write.rename({doppler_dim: 'doppler'})

                # clip variables to defined ranges
                for var in ds_to_write.data_vars:
                    if var in encodings[radar] and 'range' in zarr_variables[radar][var]:
                        vmin, vmax = zarr_variables[radar][var]['range']
                        data = ds_to_write[var].astype(float).where(np.isfinite(ds_to_write[var]))
                        data = data.clip(min=vmin, max=vmax)
                        ds_to_write[var] = data.compute()

                # rechunk if specified
                rechunk_dict = {}
                for var in ds_to_write.data_vars:
                    if 'chunks' in encodings[radar].get(var, {}):
                        dims = ds_to_write[var].dims
                        rechunk_dict[var] = dict(zip(dims, encodings[radar][var]['chunks']))
                if rechunk_dict:
                    ds_to_write = ds_to_write.chunk(rechunk_dict)

                # write to zarr
                for coord in ['range', 'doppler']:
                    if coord in ds_to_write.data_vars and not coord in ds_to_write.coords:
                        ds_to_write = ds_to_write.set_coords(coord)

                vars_to_write = [v for v in ds_to_write.data_vars if 'time' in ds_to_write[v].dims]
                coords_to_drop = ['range'] if drop_spectral else ['range', 'doppler']
                ds_to_write = ds_to_write[vars_to_write].drop_vars(coords_to_drop, errors='ignore')
                ds_encoded = preprocess_for_zarr(ds_to_write, encodings[radar]).compute()

                z_store = zarr.open_group(zarr_path, mode='r+')
                try:
                    for var in ds_encoded.data_vars:
                        if var not in z_store:
                            print(f"Warning: {var} not in zarr store, skipping")
                            continue
                        
                        data = ds_encoded[var].values                    
                        # write directly to the zarr array at the specified region
                        if data.ndim == 2:  # (time, range)
                            z_store[var][start:stop, :] = data
                        elif data.ndim == 1:  # (time,)
                            z_store[var][start:stop] = data
                        elif data.ndim == 3:  # (time, range, doppler)
                            z_store[var][start:stop, :, :] = data
                    
                    zarr.consolidate_metadata(zarr_path)
                except Exception as e:
                    print(f"    ERROR writing to zarr for {radar}: {e}", flush=True)
                    zarr_ds.close()
                    continue
                
                zarr_ds.close()
                ts_end = pd.Timestamp.now()
                print(f"    Wrote {len(overlapping_times)} timesteps for {radar} on {ts_end.strftime('%H:%M:%S (%Y-%m-%d)')}", flush=True)
                print(f"    (elapsed time for loading & appending data to zarr: {format_elapsed((ts_end - ts_zarr).total_seconds())}", flush=True)


def decode_zarr(ds: xr.Dataset, encodings: Dict[str, Dict]) -> xr.Dataset:
    ds_decoded = ds.copy()

    for var, enc in encodings.items():
        if var not in ds_decoded:
            continue

        da = ds_decoded[var]
        dtype = enc.get("dtype", np.float32)
        fill = enc.get("_FillValue", None)

        # handle boolean separately (no scaling)
        if dtype == bool:
            da = da.astype(bool)
            ds_decoded[var] = da
            continue

        # mask fill values BEFORE scaling (fill is in encoded space)
        if fill is not None:
            da = da.where(da != fill)  # Mask encoded fill values as NaN
        
        # convert to float for math operations
        da = da.astype(np.float32)

        # apply scale and offset
        scale = enc.get("scale_factor", 1.0)
        offset = enc.get("add_offset", 0.0)
        da = da * scale + offset

        ds_decoded[var] = da

    return ds_decoded


def plot_temp_stratified_density(ds, var_x, var_y, temp_bins=None, 
                                  cloud_type=None, precip_only=False, save_path=None):
    """
    create density plots stratified by temperature zone.

    ds : xarray.Dataset
    var_x, var_y : str, variable names (e.g., 'Z_X', 'width_X')
    temp_bins : list of tuples, optional, temperature ranges, e.g., [(-8, -3), (-15, -8), (-25, -15)] (if None, uses predefined SIP zones)
    cloud_type : str, optional, 'orographic' or 'frontal' to filter data
    precip_only : bool, if True, only use precipitation periods
    save_path : str, optional
    """
    if temp_bins is None:
        # Use predefined zones
        zones = [('HM_zone', 'Hallett-Mossop\n-3 to -8°C'),
                 ('droplet_shatter_zone', 'droplet shattering\n-8 to -15°C'),
                 ('ice_collision_zone', 'ice collision\n-15 to -25°C')]
    else:
        zones = [(None, f'{t[0]} to {t[1]}°C') for t in temp_bins]
    
    # Filter by cloud type if specified
    ds_filtered = ds.copy()
    if cloud_type and 'cloud_type' in ds:
        ds_filtered = ds_filtered.where(ds_filtered['cloud_type'] == cloud_type)
    
    # Filter by precipitation if specified
    if precip_only and 'is_precip' in ds:
        ds_filtered = ds_filtered.where(ds_filtered['is_precip'])
    
    n_zones = len(zones)
    fig, axes = plt.subplots(1, n_zones, figsize=(5*n_zones, 4))
    if n_zones == 1:
        axes = [axes]
    
    for i, (zone_mask, zone_label) in enumerate(zones):
        if zone_mask:
            # use predefined mask
            mask = ds_filtered[zone_mask]
        else:
            # create mask from temperature bins
            t_min, t_max = temp_bins[i]
            mask = (ds_filtered['temperature_approx'] >= t_min) & (ds_filtered['temperature_approx'] < t_max)
        
        x_data = ds_filtered[var_x].where(mask).values.flatten()
        y_data = ds_filtered[var_y].where(mask).values.flatten()
        
        valid = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[valid]
        y_data = y_data[valid]
        
        if len(x_data) > 0:
            # 2D histogram
            h, xedges, yedges = np.histogram2d(x_data, y_data, bins=50)
            h = np.ma.masked_where(h == 0, h)
            
            im = axes[i].pcolormesh(xedges, yedges, h.T, 
                                    norm=LogNorm(vmin=1, vmax=h.max()), 
                                    cmap='viridis')
            axes[i].set_xlabel(var_x)
            if i == 0:
                axes[i].set_ylabel(var_y)
            axes[i].set_title(zone_label)
            axes[i].grid(True, alpha=0.3)
            plt.colorbar(im, ax=axes[i], label='Count')
            
            # add sample count
            axes[i].text(0.05, 0.95, f'N={len(x_data):,}', 
                        transform=axes[i].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
    
    title_suffix = ''
    if cloud_type:
        title_suffix += f' ({cloud_type} clouds)'
    if precip_only:
        title_suffix += ' [precipitating inte only]'
    
    fig.suptitle(f'{var_y} vs {var_x}{title_suffix}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_orographic_vs_frontal_comparison(ds, radar='X', precip_only=False, save_path=None):
    """
    compare SIP signatures between orographic and frontal clouds
    
    ds : xarray.Dataset (must contain 'cloud_type' classification()
    radar : str
    save_path : str, optional
    """
    if 'cloud_type' not in ds:
        print("Error: Dataset must contain 'cloud_type' variable")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    cloud_types = ['orographic', 'frontal']
    if precip_only and 'is_precip' in ds:
        time_mask = ds['is_precip']
        ds = ds.sel(time=time_mask)

    colors = ['steelblue', 'coral']
    
    # cloud top height
    for ctype, color in zip(cloud_types, colors):
        if 'cloudtop' in ds:
            data = ds['cloudtop'].where(ds['cloud_type'] == ctype).values.flatten()
            data = data[~np.isnan(data)]
            if len(data) > 0:
                axes[0, 0].hist(data, bins=30, alpha=0.6, label=ctype.capitalize(), 
                               color=color, edgecolor='black', density=True)
    axes[0, 0].set_xlabel('Cloud top height (m above radar)')
    axes[0, 0].set_ylabel('probability density')
    axes[0, 0].set_title('cloud top height distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # multi-modal frequency in HM zone
    peak_var = f'numpeaks_{radar}'
    if 'HM_zone' in ds and peak_var in ds:
        zone_data = {}
        for ctype, color in zip(cloud_types, colors):
            mask = ds['HM_zone'] & (ds['cloud_type'] == ctype)
            peaks = ds[peak_var].where(mask).values.flatten()
            peaks = peaks[~np.isnan(peaks)]
            
            if len(peaks) > 0:
                multi_modal_frac = np.sum(peaks > 1) / len(peaks)
                zone_data[ctype] = {'multi': multi_modal_frac, 'single': 1 - multi_modal_frac, 'n': len(peaks)}
        
        if zone_data:
            x = np.arange(len(zone_data))
            width = 0.35
            single_vals = [zone_data[ct]['single'] for ct in cloud_types if ct in zone_data]
            multi_vals = [zone_data[ct]['multi'] for ct in cloud_types if ct in zone_data]
            labels = [ct.capitalize() for ct in cloud_types if ct in zone_data]
            
            axes[0, 1].bar(x - width/2, single_vals, width, label='single peak', color='lightblue')
            axes[0, 1].bar(x + width/2, multi_vals, width, label='multi-modal', color='salmon')
            axes[0, 1].set_ylabel('fraction of observations')
            axes[0, 1].set_title('peak modality in HM zone')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(labels)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Add sample sizes
            for i, (ct, color) in enumerate(zip([c for c in cloud_types if c in zone_data], colors)):
                axes[0, 1].text(i, 0.95, f"N={zone_data[ct]['n']:,}", 
                               ha='center', va='top', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # mean reflectivity profile by cloud type
    z_var = f'Z_{radar}'
    if z_var in ds and 'height_rel_to_0C' in ds:
        height_bins = np.arange(-500, 5000, 100)
        
        for ctype, color in zip(cloud_types, colors):
            ds_type = ds.where(ds['cloud_type'] == ctype)
            
            height = ds_type['height_rel_to_0C'].values.flatten()
            z = ds_type[z_var].values.flatten()
            
            valid = ~(np.isnan(height) | np.isnan(z))
            height = height[valid]
            z = z[valid]
            
            if len(height) > 0:
                # Calculate mean Z in height bins
                z_binned = []
                for i in range(len(height_bins) - 1):
                    mask = (height >= height_bins[i]) & (height < height_bins[i+1])
                    if np.sum(mask) > 10:
                        z_binned.append(np.mean(z[mask]))
                    else:
                        z_binned.append(np.nan)
                
                axes[0, 2].plot(z_binned, height_bins[:-1], 
                               label=ctype.capitalize(), color=color, linewidth=2)
        
        axes[0, 2].set_xlabel(f'mean {z_var} (dBZ)')
        axes[0, 2].set_ylabel('height relative to 0°C (m)')
        axes[0, 2].set_title('mean reflectivity profile')
        axes[0, 2].axhline(460, color='red', linestyle='--', alpha=0.3, linewidth=1)
        axes[0, 2].axhline(1230, color='red', linestyle='--', alpha=0.3, linewidth=1)
        axes[0, 2].axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # DFR in HM zone
    if 'DFR_KaW' in ds and 'HM_zone' in ds:
        for ctype, color in zip(cloud_types, colors):
            mask = ds['HM_zone'] & (ds['cloud_type'] == ctype)
            dfr = ds['DFR_KaW'].where(mask).values.flatten()
            dfr = dfr[~np.isnan(dfr)]
            
            if len(dfr) > 0:
                axes[1, 0].hist(dfr, bins=50, alpha=0.6, label=ctype.capitalize(),
                               color=color, edgecolor='black', density=True)
        
        axes[1, 0].set_xlabel('DFR Ka-W (dB)')
        axes[1, 0].set_ylabel('probability density')
        axes[1, 0].set_title('DFR distribution in HM zone')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # spectral width in HM zone
    width_var = f'width_{radar}'
    if width_var in ds and 'HM_zone' in ds:
        for ctype, color in zip(cloud_types, colors):
            mask = ds['HM_zone'] & (ds['cloud_type'] == ctype)
            width = ds[width_var].where(mask).values.flatten()
            width = width[~np.isnan(width)]
            
            if len(width) > 0:
                axes[1, 1].hist(width, bins=50, range=(0, 2), alpha=0.6, 
                               label=ctype.capitalize(), color=color, edgecolor='black', density=True)
        
        axes[1, 1].set_xlabel('spectral width (m/s)')
        axes[1, 1].set_ylabel('probability density')
        axes[1, 1].set_title('spectral width in HM zone')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # vertical gradient statistics by cloud type and zone
    dz_var = f'dZ_dh_{radar}'
    if dz_var in ds:
        zones = [('HM_zone', 'HM'), ('droplet_shatter_zone', 'droplet'), ('ice_collision_zone', 'ice')]
        
        oro_means = []
        frontal_means = []
        zone_labels = []
        
        for zone_mask, zone_label in zones:
            if zone_mask in ds:
                mask_oro = ds[zone_mask] & (ds['cloud_type'] == 'orographic')
                dz_oro = ds[dz_var].where(mask_oro).values.flatten()
                dz_oro = dz_oro[~np.isnan(dz_oro)]
                
                mask_frontal = ds[zone_mask] & (ds['cloud_type'] == 'frontal')
                dz_frontal = ds[dz_var].where(mask_frontal).values.flatten()
                dz_frontal = dz_frontal[~np.isnan(dz_frontal)]
                
                if len(dz_oro) > 0 and len(dz_frontal) > 0:
                    oro_means.append(np.mean(dz_oro) * 1000)  # Convert to dB/km
                    frontal_means.append(np.mean(dz_frontal) * 1000)
                    zone_labels.append(zone_label)
        
        if zone_labels:
            x = np.arange(len(zone_labels))
            width = 0.35
            axes[1, 2].bar(x - width/2, oro_means, width, label='orographic', color='steelblue')
            axes[1, 2].bar(x + width/2, frontal_means, width, label='frontal', color='coral')
            axes[1, 2].set_ylabel('mean dZ/dh (dB/km)')
            axes[1, 2].set_title('reflectivity gradient by zone')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(zone_labels)
            axes[1, 2].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Orographic vs frontal cloud comparison - {radar}-band', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_temp_stratified_density(ds, var_x, var_y, temp_bins=None, 
                                  cloud_type=None, precip_only=False, save_path=None):
    """
    Create density plots stratified by temperature zone.
    
    Parameters:
    -----------
    ds : xarray.Dataset
    var_x, var_y : str
        Variable names (e.g., 'Z_X', 'width_X')
    temp_bins : list of tuples, optional
        Temperature ranges, e.g., [(-8, -3), (-15, -8), (-25, -15)]
        If None, uses predefined SIP zones
    cloud_type : str, optional
        'orographic' or 'frontal' to filter data
    precip_only : bool
        If True, only use precipitation periods
    save_path : str, optional
    """
    if temp_bins is None:
        zones = [('HM_zone', 'Hallett-Mossop\n-3 to -8°C'),
                 ('droplet_shatter_zone', 'droplet shatter\n-8 to -15°C'),
                 ('ice_collision_zone', 'ice collision\n-15 to -25°C')]
    else:
        zones = [(None, f'{t[0]} to {t[1]}°C') for t in temp_bins]
    
    ds_filtered = ds.copy()
    if cloud_type and 'cloud_type' in ds:
        # cloud_type is 1D (time only), so select times where condition is true
        time_mask = ds_filtered['cloud_type'] == cloud_type
        ds_filtered = ds_filtered.sel(time=time_mask)
    
    if precip_only and 'is_precip' in ds:
        time_mask = ds_filtered['is_precip']
        ds_filtered = ds_filtered.sel(time=time_mask)
    
    n_zones = len(zones)
    fig, axes = plt.subplots(1, n_zones, figsize=(5*n_zones, 4))
    if n_zones == 1:
        axes = [axes]
    
    for i, (zone_mask, zone_label) in enumerate(zones):
        if zone_mask:
            mask = ds_filtered[zone_mask]
        else:
            t_min, t_max = temp_bins[i]
            mask = (ds_filtered['temperature_approx'] >= t_min) & (ds_filtered['temperature_approx'] < t_max)
        
        x_data = ds_filtered[var_x].where(mask).values.flatten()
        y_data = ds_filtered[var_y].where(mask).values.flatten()
        
        valid = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[valid]
        y_data = y_data[valid]
        
        if len(x_data) > 0:
            # 2D histogram
            h, xedges, yedges = np.histogram2d(x_data, y_data, bins=50)
            h = np.ma.masked_where(h == 0, h)
            
            im = axes[i].pcolormesh(xedges, yedges, h.T, 
                                    norm=LogNorm(vmin=1, vmax=h.max()), 
                                    cmap='viridis')
            axes[i].set_xlabel(var_x)
            if i == 0:
                axes[i].set_ylabel(var_y)
            axes[i].set_title(zone_label)
            axes[i].grid(True, alpha=0.3)
            plt.colorbar(im, ax=axes[i], label='Count')
            
            # Add sample count
            axes[i].text(0.05, 0.95, f'N={len(x_data):,}', 
                        transform=axes[i].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
    
    title_suffix = ''
    if cloud_type:
        title_suffix += f' ({cloud_type} clouds)'
    if precip_only:
        title_suffix += ' [precip only]'
    
    fig.suptitle(f'{var_y} vs {var_x}{title_suffix}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def decode_and_combine_radars(zarrdir, radars):
    """
    Load, decode, and combine radar datasets with proper variable renaming.
    
    zarrdir : str, directory containing zarr files
    radars : list of radar names (e.g., ['MXPol', 'MIRA', 'BASTA'])
    encodings : dict containing encoding of zarr_variables
    
    Returns:
    ds_combined : xarray.Dataset
    dsets : dict of individual radar datasets (for reference)
    """
    dsets = {}
    
    for radar in radars:
        # print(f"Opening {radar}.zarr...")
        t_radar = time.time()
        d = xr.open_zarr(f"{zarrdir}/{radar}.zarr", consolidated=True)
        # d = decode_zarr(d, radar, encodings) # decoding not necessary - xarray automatically handles this
        # print(f"  Opened in {time.time()-t_radar:.2f}s")
        # try:
        #     print(f"  Chunks: {d.chunks}")
        # except ValueError as e:
        #     print(f"  Chunks: inconsistent - {e}")
        d = d.unify_chunks()
        
        if radar == 'MXPol':
            d = d.rename({
                'reflectivity': 'Z_X',
                'reflectivity_raw': 'Z_X_full',
                'velocity': 'velocity_X',
                'spectrumwidth': 'width_X',
                'zdr': 'zdr_X',
                'rhohv': 'rhohv_X',
                'snrh': 'SNRh_X',
                'snrv': 'SNRv_X',
            })
            if 'skewness' in d:
                d = d.rename({'skewness': 'skewness_X'})
            if 'doppler' in d.dims:
                d = d.rename({'sZh': 'sZh_X',
                              'sZv': 'sZv_X'})
            dsets['X'] = d
            
        elif radar == 'MIRA':
            d = d.rename({
                'reflectivity': 'Z_Ka',
                'velocity': 'velocity_Ka',
                'skewness': 'skewness_Ka',
                'spectrumwidth': 'width_Ka',
                'LDR': 'LDR_Ka',
                'numpeaks': 'numpeaks_Ka',
                'snrh': 'SNRh_Ka',
                'snrv': 'SNRv_Ka'  
            })
            if 'doppler' in d.dims:
                d = d.rename({'sZh': 'sZh_Ka',
                              'sZv': 'sZv_Ka'})
            dsets['Ka'] = d
            
        elif radar == 'BASTA':
            d = d.rename({
                'reflectivity': 'Z_W',
                'velocity': 'velocity_W',
                'reflectivity_raw': 'Z_W_raw',
                'mask_artefacts_loose': 'mask_artefacts_W',
            })
            if 'mask_artefacts_loose' in d.data_vars:
                d = d.rename({'mask_artefacts_loose': 'mask_artefacts_W'})
            dsets['W'] = d
            
        elif radar == 'peaktree':
            d = d.rename({
                'reflectivity': 'Z_Ka_PT',
                'velocity': 'velocity_Ka_PT',
                'skewness': 'skewness_Ka_PT',
                'spectrumwidth': 'width_Ka_PT',
                'LDR': 'LDR_Ka_PT',
                'numpeaks': 'numpeaks_Ka_PT'            
            })
            dsets['Ka_PT'] = d
            print(f"  Renamed in {time.time()-t_radar:.2f}s")

    print("Combining datasets...")
    t_combine = time.time()
    ds_combined = dsets['Ka']
    data_vars = {}

    for var in ds_combined.data_vars:
        data_vars[var] = ds_combined[var]
    
    if 'X' in dsets:
        x_vars = ['Z_X', 'velocity_X', 'skewness_X', 'width_X', 'zdr_X', 'rhohv_X', 'sZh_X', 'sZv_X']
        for var in x_vars:
            if var in dsets['X']:
                data_vars[var] = dsets['X'][var]
        
        if 'DFR_XW' in dsets['X']:
            data_vars['DFR_XW'] = dsets['X']['DFR_XW']

    if 'W' in dsets:
        w_vars = ['Z_W', 'Z_W_raw', 'velocity_W']
        for var in w_vars:
            if var in dsets['W']:
                data_vars[var] = dsets['W'][var]
        
        if 'cloudtop' in dsets['W']:
            data_vars['cloudtop'] = dsets['W']['cloudtop']

    if 'Ka_PT' in dsets:
        pt_vars = ['Z_Ka_PT', 'velocity_Ka_PT', 'skewness_Ka_PT', 'width_Ka_PT', 
                    'LDR_Ka_PT', 'numpeaks_Ka_PT']
        for var in pt_vars:
            if var in dsets['Ka_PT']:
                data_vars[var] = dsets['Ka_PT'][var]

    # create new dataset from collected variables
    ds_combined = xr.Dataset(
        data_vars=data_vars,
        coords=dsets['Ka'].coords,
        attrs=dsets['Ka'].attrs
    )

    print(f"Combined in {time.time()-t_combine:.2f}s")

    for v in ds_combined.data_vars:
        if np.issubdtype(ds_combined[v].dtype, np.floating):
            ds_combined[v] = ds_combined[v].astype('float32')

    # print(f"\nCombined dataset variables:")
    # for var in sorted(ds_combined.data_vars):
    #     print(f"  {var}: {ds_combined[var].dims}")
    
    return ds_combined, dsets


def decode_and_combine_modes(dir, modes):
    """
    load, decode, and combine datasets of different modes (BASTA)
    
    zarrdir : str, directory containing zarr files
    radars : list of radar names (e.g., ['MXPol', 'MIRA', 'BASTA'])
    encodings : dict containing encoding of zarr_variables
    
    Returns:
    ds_combined : xarray.Dataset with all modes
    dsets : dict of individual radar datasets (for reference)
    """
    dsets = {}
    
    for mode in modes:
        # print(f"Opening {radar}.zarr...")
        t_radar = time.time()
        d = xr.open_zarr(f"{dir}/BASTA_{mode}.zarr", consolidated=True)
        # d = decode_zarr(d, radar, encodings) # decoding not necessary - xarray automatically handles this
        # print(f"  Opened in {time.time()-t_radar:.2f}s")
        # try:
        #     print(f"  Chunks: {d.chunks}")
        # except ValueError as e:
        #     print(f"  Chunks: inconsistent - {e}")
        d = d.unify_chunks()
        
        dsets[mode] = d
        
    return dsets


def define_temperature_zones(ds, lapse_rate=6.5):
    """
    Define temperature zones relative to 0°C isotherm.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset with 'range' and 'zdi' (0°C isotherm height above radar)
    lapse_rate : float
        Temperature lapse rate in K/km (default: 6.5)
    
    Returns:
    --------
    ds : xarray.Dataset with added temperature and zone masks
    """
    # calculate height relative to 0°C
    # zdi and cloudtop are heights above radar
    ds['height_rel_to_0C'] = ds['range'] - ds['zdi']
    
    # approximate temperature (°C)
    ds['temperature_approx'] = -1 * (ds['height_rel_to_0C'] / 1000) * lapse_rate
    
    # define SIP zones (height in meters above 0°C)
    ds['HM_zone'] = (ds['height_rel_to_0C'] >= 460) & (ds['height_rel_to_0C'] <= 1230)  # -3 to -8°C
    ds['droplet_shatter_zone'] = (ds['height_rel_to_0C'] > 1230) & (ds['height_rel_to_0C'] <= 2300)  # -8 to -15°C
    ds['ice_collision_zone'] = (ds['height_rel_to_0C'] > 2300) & (ds['height_rel_to_0C'] <= 3850)  # -15 to -25°C
    
    return ds


def classify_cloud_type(ds, orographic_threshold=3000):
    """
    Classify clouds as orographic vs. frontal/developed based on cloud top height.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Must contain 'cloudtop_height'
    orographic_threshold : float
        Height threshold in meters above radar (default: 3000m)
    
    Returns:
    --------
    ds : xarray.Dataset with 'cloud_type' variable
    """
    ds['cloud_type'] = xr.where(ds['cloudtop'] < orographic_threshold, 
                                  'orographic', 'frontal')
    return ds


def calculate_vertical_gradients(ds, radar='X'):
    """
    Calculate vertical gradients of reflectivity and spectral width.
    
    Parameters:
    -----------
    ds : xarray.Dataset
    radar : str
        Radar to use ('X' or 'Ka')
    
    Returns:
    --------
    ds : xarray.Dataset with added gradient variables
    """
    z_var = f'Z_{radar}'
    width_var = f'width_{radar}'
    
    # calculate gradients (centered difference)
    if z_var in ds:
        ds[f'dZ_dh_{radar}'] = ds[z_var].differentiate('range')
    
    if width_var in ds:
        ds[f'dW_dh_{radar}'] = ds[width_var].differentiate('range')
    
    return ds


def resample_native(data, target_time, target_range, time_tol, range_tol, linear_vars, flag_vars, method='linear'):
    cont = data[linear_vars]
    cont = safe_reindex(cont, 'time', target_time, time_tol, method=method)
    cont = safe_reindex(cont, 'range', target_range, range_tol, method=method)

    flags_all = data[flag_vars]
    time_only_flags = [v for v in flag_vars if 'range' not in flags_all[v].dims]
    range_flags = [v for v in flag_vars if 'range' in flags_all[v].dims]

    flags_parts = []
    if time_only_flags:
        f = safe_reindex_dim(data[time_only_flags], 'time', target_time, time_tol)
        flags_parts.append(f)
    if range_flags:
        f = safe_reindex_dim(data[range_flags], 'time', target_time, time_tol)
        f = safe_reindex_dim(f, 'range', target_range, range_tol)
        flags_parts.append(f)

    flags = xr.merge(flags_parts) if flags_parts else xr.Dataset()
    flags = (flags == 1)

    return xr.merge([cont, flags])


def update_resampled_zarr(radar_name, native_ds, resampled_zarr_path, resampled_ds, time_tol, range_tol, linear_vars, flag_vars, freq='1D', save=True):
    """loop over the resampled dataset's time grid in chunks (freq), resample the corresponding
    native data window, accumulate results, then overwrite each variable in the resampled
    zarr store in one full-array write (avoiding partial-region chunk alignment issues)"""

    target_time_full = resampled_ds.time
    target_range = resampled_ds.range

    # build period boundaries over the campaign, aligned to the resampled store's own time grid
    period_starts = pd.date_range(target_time_full.values[0], target_time_full.values[-1], freq=freq)

    for period_start in period_starts:
        period_end = period_start + pd.tseries.frequencies.to_offset(freq)
        target_mask = (target_time_full >= period_start) & (target_time_full < period_end)
        if not target_mask.any():
            continue
        target_time_period = target_time_full.where(target_mask, drop=True)

        pad = pd.Timedelta(time_tol) * 2
        native_slice = native_ds.sel(time=slice(period_start - pad, period_end + pad))
        if native_slice.sizes.get('time', 0) == 0:
            print(f"--- No native data for {radar_name} {period_start} to {period_end}, skipping ---")
            continue

        print(f"--- Resampling {radar_name} {period_start} to {period_end} ---")
        updated_period = resample_native(
            native_slice, target_time_period, target_range,
            time_tol=time_tol, range_tol=range_tol,
            linear_vars=linear_vars, flag_vars=flag_vars,
        ).compute()

        start_idx = int(np.searchsorted(target_time_full.values, target_time_period.values[0]))
        end_idx = start_idx + len(target_time_period)

        updated_period.drop_vars('range').to_zarr(
            resampled_zarr_path,
            mode="r+",
            region={"time": slice(start_idx, end_idx)},
        )

    zarr.consolidate_metadata(resampled_zarr_path)
