#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Provides a generic `aggregate_campaign` function that concatenates per-day
radar DataArrays (MIRA/BASTA/MXPol), computes per-range statistics and
saves a NetCDF summary and a small CSV summary. This module is intentionally
conservative: it only reads datasets via the existing `load_zenithdata`
helper and respects masks when present.

Usage (example):
    from src.aggregator import aggregate_campaign, default_variable_maps
    aggregate_campaign(
        radar='MIRA',
        dates=dates,
        dirs=dirs,
        DFRs=DFRs,
        att_fn=att_fn,
        BASTAmode=BASTAmode,
        variables_map=default_variable_maps['MIRA'],
        range_grid=np.arange(400, 10e3, 25),
        output_dir=f"{base_dir}/campaign_stats",
    )
"""
from __future__ import annotations

from src import os, np, pd, xr, FuncFormatter
from src.constants_input import default_variable_maps
from src.radar_processing import load_zenithdata
from src.utils import safe_reindex

from typing import List, Dict, Optional


def _compute_stat(da: xr.DataArray, stat: str) -> xr.DataArray:
    """Compute a single statistic on DataArray along time dimension.

    Returns a DataArray aligned with the 'range' coordinate.
    """
    if stat == 'mean':
        return da.mean(dim='time', skipna=True)
    if stat == 'min':
        return da.min(dim='time', skipna=True)
    if stat == 'max':
        return da.max(dim='time', skipna=True)
    if stat == 'std':
        return da.std(dim='time', skipna=True)
    if stat == 'median':
        return da.median(dim='time', skipna=True)
    if stat == 'count':
        # count of non-NaN samples
        return da.count(dim='time')
    raise ValueError(f"Unsupported stat: {stat}")


def aggregate_campaign(
    dates: List[pd.Timestamp],
    dirs: Dict[str, str],
    DFRs,
    att_fn: str,
    BASTAmode: str,
    radars: List[str] = ['MIRA', 'BASTA', 'MXPol'],
    resample_time: bool = False,
    range_grid: Optional[np.ndarray] = None,
    stats: List[str] = None,
    output_dir: Optional[str] = None,
    peakTree: bool = True,
) -> Dict[str, str]:
    """Aggregate a radar campaign and compute per-range statistics, including below-cloudtop (_belowct) stats.

    Parameters
    ----------
    dates: list of datetimes to process
    dirs, DFRs, att_fn, BASTAmode: same parameters as used by load_zenithdata
    variables_map: mapping canonical->actual variable names in each dataset
    range_grid: optional common range grid to reindex range coordinate
    resample_time: pass to load_zenithdata (if True, each day's data is reindexed to a full_time grid)
    stats: list of statistics to compute (default: ['mean','min','max','std','median','count'])
    output_dir: where to write NetCDF and CSV. if None, uses current working dir.
    peakTree: whether to load peaktree files in load_zenithdata

    Returns a dict with paths of written files.
    """
    if stats is None:
        stats = ['mean', 'min', 'max', 'std', 'median', 'count']

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'campaign_stats')
    os.makedirs(output_dir, exist_ok=True)

    all_collected: Dict[str, Dict[str, List[xr.DataArray]]] = {}
    all_processed_dates: Dict[str, List[pd.Timestamp]] = {}

    for radar in radars:
        variables_map = default_variable_maps.get(radar, {})
        all_collected[radar] = {k: [] for k in variables_map.keys()}
        all_processed_dates[radar] = []
    
    for date in dates:
        print(f"\n=== Loading data for {date.strftime('%Y-%m-%d')} ===")
        try:
            datasets = load_zenithdata(date, dirs, DFRs, att_fn, BASTAmode, 
                                    useMXPol = ('MXPol' in radars), useMIRA = ('MIRA' in radars), useBASTA = ('BASTA' in radars),
                                    resample_time=resample_time, ERA=False, peakTree= ('peaktree' in radars))
        except Exception as e:
            print(f"failed to load data for {date.strftime('%Y-%m-%d')}: {e}")
            continue

        for radar in radars:
            if radar not in datasets:
                print(f"  no {radar} data for {date.strftime('%Y-%m-%d')}, skipping")
                continue
            print(f"\n=== Processing {radar} for {date.strftime('%Y-%m-%d')} ===")
            
            variables_map = default_variable_maps.get(radar, {})

            # storage for DataArrays
            collected: Dict[str, List[xr.DataArray]] = {k: [] for k in variables_map.keys()}
            processed_dates: List[pd.Timestamp] = []
            
            ds = datasets[radar]

            # reindex range if requested
            if range_grid is not None and 'range' in ds.coords:
                try:
                    ds = safe_reindex(ds, 'range', range_grid, tolerance=25, name=radar)
                except Exception:
                    # safe_reindex prints warnings; continue with original ds
                    pass

            # extract variables
            for canon, varname in variables_map.items():
                if varname in ds:
                    da = ds[varname]
                    # apply clean_mask if it exists
                    if 'clean_mask' in ds and set(da.dims) >= set(['time', 'range']):
                        mask = ds['clean_mask']
                        try:
                            da = da.where(mask)
                        except Exception:
                            pass

                    # fix dimension handling for 1D cloudtop array
                    if da.dims != ('time', 'range') and all_collected[radar].get('reflectivity'):
                        da = da.expand_dims({'range': all_collected[radar]['reflectivity'][0].range}, axis=1).transpose('time', 'range')
                    all_collected[radar][canon].append(da)
                else:
                    print(f"  variable {varname} (canonical: {canon}) not in dataset for {date.strftime('%Y-%m-%d')}")

            all_processed_dates[radar].append(pd.to_datetime(date))

    results = {}
    for radar in radars:
        print(f"\n=== Computing statistics for {radar} ===")

        collected = all_collected[radar]
        processed_dates = sorted(all_processed_dates[radar])

        if len(processed_dates) == 0:
            print(f"No {radar} data found for the specified dates, skipping.")
            continue

        # concat and compute stats
        out_ds = xr.Dataset()

        for canon, da_list in collected.items():
            if not da_list:
                continue
            try:
                full = xr.concat(da_list, dim='time', join='outer')
            except Exception:
                print(f"  concat failed for {canon}: {e}")

            # compute requested stats
            for st in stats:
                try:
                    stat_da = _compute_stat(full, st)
                    out_ds[f"{canon}_{st}"] = stat_da
                except Exception as e:
                    print(f"  failed to compute {st} for {canon}: {e}")

            if radar == 'MIRA' and collected.get('num_peaks'):
                try:
                    npkg_list = collected['num_peaks']
                    npkg_full = xr.concat(npkg_list, dim='time', join='outer')
                    no_peaks = np.ceil(npkg_full / 2.).astype(int)

                    unique_npeaks = np.sort(np.unique(no_peaks.values[~np.isnan(no_peaks.values)])).astype(int)
                    unique_npeaks = unique_npeaks[unique_npeaks <= 5]

                    for canon, da_list in collected.items():
                        if not da_list:
                            continue
                        try:
                            full = xr.concat(da_list, dim='time', join='outer')
                            for st in stats:
                                stat_by_npeaks = []

                                for npeak_val in unique_npeaks:
                                    mask = (no_peaks == npeak_val)
                                    da_masked = full.where(mask)
                                    
                                    try:
                                        stat_da = _compute_stat(da_masked, st)
                                        if 'range' in stat_da.dims:
                                            if st == 'count':
                                                stat_scalar = stat_da.sum(dim='range', skipna=True)
                                            else:
                                                stat_scalar = stat_da.mean(dim='range', skipna=True)
                                        else:
                                            stat_scalar = stat_da
                                        stat_by_npeaks.append(stat_scalar)
                                    except Exception as e:
                                        print(f"  failed to compute {st} for {canon} with no_peaks={npeak_val}: {e}")
                                
                                if stat_by_npeaks:
                                    combined = xr.concat(stat_by_npeaks, dim='no_peaks')
                                    combined = combined.assign_coords({'no_peaks': unique_npeaks})
                                    out_ds[f"{canon}_npeaks_{st}"] = combined
                                    
                        except Exception as e:
                            print(f"  failed to compute by_no_peaks stats for {canon}: {e}")

                except Exception as e:
                    print(f"  failed to compute by_no_peaks stats: {e}")

            # compute _belowct stats if cloudtop exists
            if collected.get('cloudtop'):
                try:
                    cloudtop_full = xr.concat(collected['cloudtop'], dim='time', join='outer')
                    below_mask = xr.DataArray(
                        cloudtop_full.coords['range'].values[None, :] < cloudtop_full.values[:, 0, None],
                        dims=('time', 'range'),
                        coords={'time': cloudtop_full.coords['time'], 'range': cloudtop_full.coords['range']}
                        )
                    
                    for canon, da_list in collected.items():
                        if not da_list or canon == 'cloudtop':
                            continue

                        try:
                            full = xr.concat(da_list, dim='time', join='outer')
                            da_below = full.where(below_mask)

                            for st in stats:
                                try:
                                    stat_da = _compute_stat(da_below, st)
                                    out_ds[f"{canon}_belowct_{st}"] = stat_da
                                except Exception as e:
                                    print(f"  failed to compute belowct {st} for {canon}: {e}")
                        except Exception as e:
                            print(f"  failed to compute belowct stats for {canon}: {e}")

                except Exception as e:
                    print(f"  failed to create below-cloudtop mask: {e}")
        
        # cloudtop campaign stats
        if collected.get('cloudtop'):
            try:
                cloudtop_full = xr.concat(collected['cloudtop'], dim='time', join='outer')
                cloudtop_1d = cloudtop_full.isel(range=0) if 'range' in cloudtop_full.dims else cloudtop_full

                for st in stats:
                    try:
                        if st == 'count':
                            val = cloudtop_1d.count(dim='time')
                        else:
                            val = getattr(cloudtop_1d, st)(dim='time', skipna=True)
                        out_ds.attrs[f'cloudtop_{st}'] = float(val.values)
                    except Exception as e:
                        print(f"  failed to compute cloudtop {st}: {e}")
            except Exception as e:
                print(f"  failed to compute cloudtop campaign stats: {e}")
                
            if radar == 'MIRA' and collected.get('num_peaks'):
                try:
                    npkg_list = collected['num_peaks']
                    npkg_full = xr.concat(npkg_list, dim='time', join='outer')
                    
                    if 'range' in npkg_full.dims:
                        no_peaks_1d = npkg_full.isel(range=0)
                    else:
                        no_peaks_1d = npkg_full
                    no_peaks = np.ceil(no_peaks_1d / 2.).astype(int)

                    unique_npeaks = np.sort(np.unique(no_peaks.values[~np.isnan(no_peaks.values)])).astype(int)
                    unique_npeaks = unique_npeaks[unique_npeaks <= 5]

                    for st in stats:
                        stat_by_npeaks = []

                    for npeak_val in unique_npeaks:
                        mask = (no_peaks == npeak_val)
                        cloudtop_masked = cloudtop_1d.where(mask)
                        
                        try:
                            if st == 'count':
                                val = cloudtop_masked.count(dim='time')
                            else:
                                val = getattr(cloudtop_masked, st)(dim='time', skipna=True)
                            
                            if hasattr(val, 'values'):
                                val = float(val.values)
                            stat_by_npeaks.append(val)

                        except Exception as e:
                            print(f"  failed to compute cloudtop {st} for no_peaks={npeak_val}: {e}")

                    if stat_by_npeaks:
                        combined = xr.DataArray(
                            stat_by_npeaks,
                            dims=['no_peaks'],
                            coords={'no_peaks': unique_npeaks}
                        )
                        out_ds[f"cloudtop_npeaks_{st}"] = combined

                except Exception as e:
                    print(f"  failed to compute cloudtop by_no_peaks stats: {e}")

        # global attributes
        out_ds.attrs['radar'] = radar
        out_ds.attrs['date_start'] = processed_dates[0].strftime('%Y-%m-%d')
        out_ds.attrs['date_end'] = processed_dates[-1].strftime('%Y-%m-%d')

        # write NetCDF
        start_str = processed_dates[0].strftime('%Y%m%d')
        end_str = processed_dates[-1].strftime('%Y%m%d')
        ncname = os.path.join(output_dir, f"{radar}_campaign_stats_{start_str}-{end_str}.nc")
        enc = {var: {'zlib': True, 'complevel': 4} for var in out_ds.data_vars}
        out_ds.to_netcdf(ncname, encoding=enc)
        print(f"Wrote campaign NetCDF: {ncname}")

        results[radar] = ncname

    return results

def aggregate_campaign_multidim(
        dates: List[pd.Timestamp],
        dirs: Dict[str, str],
        DFRs,
        att_fn: str,
        BASTAmode: str,
        radars: List[str] = ['MIRA', 'BASTA', 'MXPol', 'peaktree'],
        resample_time: bool = False,
        range_grid: Optional[np.ndarray] = None,
        stats: List[str] = None,
        output_dir: Optional[str] = None
        ) -> str:
    """Aggregate radar campaign statistics into a multidimensional Dataset (radar, stat, range/npeaks/time)."""
    if stats is None:
        stats = ['mean', 'min', 'max', 'std', 'median', 'count']

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'campaign_stats')
    os.makedirs(output_dir, exist_ok=True)

    all_collected = {}
    all_processed_dates = {}

    # 1. Load / collect data
    for radar in radars:
        variables_map = default_variable_maps.get(radar, {})
        all_collected[radar] = {k: [] for k in variables_map.keys()}
        all_processed_dates[radar] = []

    for date in dates:
        print(f"\n=== Loading data for {date.strftime('%Y-%m-%d')} ===")
        try:
            datasets = load_zenithdata(date, dirs, DFRs, att_fn, BASTAmode, 
                                    useMXPol = ('MXPol' in radars), useMIRA = ('MIRA' in radars), useBASTA = ('BASTA' in radars),
                                    resample_time=resample_time, ERA=False, peakTree= ('peaktree' in radars))
        except Exception as e:
            print(f"failed to load data for {date.strftime('%Y-%m-%d')}: {e}")
            continue

        for radar in radars:
            if radar not in datasets:
                print(f"  no {radar} data for {date.strftime('%Y-%m-%d')}, skipping")
                continue

            ds = datasets[radar]
            variables_map = default_variable_maps.get(radar, {})

            if range_grid is not None and 'range' in ds.coords:
                try:
                    ds = safe_reindex(ds, 'range', range_grid, tolerance=25, name=radar)
                except Exception:
                    pass

            for canon, varname in variables_map.items():
                if varname not in ds:
                    continue
                da = ds[varname]

                if radar == 'peaktree':
                    if 'mode' in da.dims:
                        da = da.isel(mode=0)  # select first mode if multiple
                    if 'vel' in da.dims:
                        da = da.isel(vel=da.argmax(dim='vel'))  # max over vel if present
                if 'clean_mask' in ds and set(da.dims) >= set(['time', 'range']):
                    da = da.where(ds['clean_mask'])
                all_collected[radar][canon].append(da)

            all_processed_dates[radar].append(pd.to_datetime(date))

    # Step 2. Compute statistics and assemble multidim dataset
    ds_out_list = []

    for radar in radars:
        if not all_processed_dates[radar]:
            continue
        print(f"\n=== Computing statistics for {radar} ===")

        radar_ds = xr.Dataset()
        collected = all_collected[radar]

    numpeaks_var = None
    if radar in ['MIRA', 'peaktree']:
        numpeaks_var = collected.get('numpeaks', [])
        if numpeaks_var:
            try:
                numpeaks_full = xr.concat(numpeaks_var, dim='time', join='outer')
                if radar == 'peaktree':
                    numpeaks_full = np.ceil(numpeaks_full / 2.).astype(int)
                numpeaks_full = numpeaks_full.where(numpeaks_full <= 5)
                unique_npeaks = np.sort(np.unique(numpeaks_full.values[~np.isnan(numpeaks_full.values)])).astype(int)
            except Exception as e:
                print(f"  failed to process numpeaks for {radar}: {e}")
                numpeaks_var = None
                unique_peaks = []        
        else:
            numpeaks_var = None
            unique_npeaks = []
    else:
        numpeaks_full = None
        unique_npeaks = []
    # --- Regular stats (per range) ---
    for canon, da_list in collected.items():
        if not da_list:
            continue
        try:
            full = xr.concat(da_list, dim='time', join='outer')
        except Exception as e:
            print(f"  concat failed for {canon}: {e}")
            continue

        if 'nodes' in full.dims:
            full_agg = full.max(dim='nodes', skipna=True)
        else:
            full_agg = full

        stat_dict = {}
        for st in stats:
            try:
                stat_da = _compute_stat(full_agg, st)
                stat_dict[st] = stat_da
            except Exception as e:
                print(f"  failed {st} for {canon}: {e}")
                continue

        if stat_dict:
            stat_results = [stat_dict[st] for st in stats if st in stat_dict]
            combined = xr.concat(stat_results, dim='stat')
            combined = combined.assign_coords({'stat': [st for st in stats if st in stat_dict]})
            radar_ds[canon] = combined

    # --- Additional per-numpeaks stats (for MIRA and peaktree only) ---
    if numpeaks_full is not None and len(unique_peaks) > 0:
        print(f"  Computing per-numpeaks statistics for {radar}...")
        for canon, da_list in collected.items():
            if canon == 'numpeaks' or not da_list:
                continue

            try:
                full = xr.concat(da_list, dim='time', join='outer')
            except Exception as e:
                print(f"    concat failed for {canon}: {e}")
                continue

            stats_by_peak = []
            for npv in unique_peaks:
                mask = numpeaks_full == npv
                masked = full.where(mask)

                per_peak_stats = []
                for st in stats:
                    try:
                        stat_da = _compute_stat(masked, st)
                        # Collapse range for compactness (optional)
                        if 'range' in stat_da.dims:
                            if st == 'count':
                                stat_scalar = stat_da.sum(dim='range', skipna=True)
                            else:
                                stat_scalar = stat_da.mean(dim='range', skipna=True)
                        else:
                            stat_scalar = stat_da
                        per_peak_stats.append(stat_scalar)
                    except Exception as e:
                        print(f"      failed {st} for {canon} (numpeaks={npv}): {e}")

                if per_peak_stats:
                    combined = xr.concat(per_peak_stats, dim='stat')
                    combined = combined.assign_coords({'stat': stats})
                    combined = combined.expand_dims({'numpeaks': [npv]})
                    stats_by_peak.append(combined)

            if stats_by_peak:
                combined_all = xr.concat(stats_by_peak, dim='numpeaks')
                radar_ds[f"{canon}_by_numpeaks"] = combined_all

    radar_ds = radar_ds.expand_dims({'radar': [radar]})
    ds_out_list.append(radar_ds)
    
    # Step 3. Merge all radars
    if ds_out_list:
        final_ds = xr.concat(ds_out_list, dim='radar')
    else:
        print("No radar data found, aborting.")
        return ""

    # Step 4. Add metadata and save
    final_ds.attrs['stats'] = ', '.join(stats)
    final_ds.attrs['radars'] = ', '.join(radars)
    final_ds.attrs['description'] = 'Campaign-long radar statistics (multidimensional format)'

    start_str = min(min(v) for v in all_processed_dates.values() if v).strftime('%Y%m%d')
    end_str = max(max(v) for v in all_processed_dates.values() if v).strftime('%Y%m%d')

    ncname = os.path.join(output_dir, f"campaign_stats_multidim_{start_str}-{end_str}.nc")
    enc = {var: {'zlib': True, 'complevel': 4} for var in final_ds.data_vars}
    final_ds.to_netcdf(ncname, encoding=enc)
    print(f"\n Wrote multidimensional campaign NetCDF: {ncname}")

    return ncname