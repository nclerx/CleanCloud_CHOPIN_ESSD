# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:29:32 2026

script to make synoptic ERA plots

@author: clerx
"""
# %% imports
import sys
project_path = '/home/clerx/scripts/chopin_processing'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src import os, glob, np, pd, xr, plt, sns, datetime, mcolors, LogNorm, mdates, time, scipy, cartopy, cm, cfeature, rasterio
from src import crs as ccrs
from src.constants_input import base_dir, dirs, zarr_variables, df_precip, pltConfig, plotdates_weekly, start_date, end_date, DFR_files, date_ticks, latlon_Helmos, radar_altitude
from src.utils import calc_alt
from src.plotting import plot_density_histogram, plot_density, FuncFormatter, meters_to_km, meters_to_km_num, weekly_plot, plot_weekly_HALO
from src.zarr_utils import generate_zarr_encodings, decode_and_combine_radars
from src.radar_processing import DFRcorrection_array as add_DFRcorrection
from src.radar_processing import cloudtop_height, find_continuous_cloudtop
from src.utils import determine_nbins, read_ERA_data, get_DFRdata, mask_to_intervals

import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap, BoundaryNorm

from metpy.calc import geopotential_to_height
from metpy.units import units
from scipy.interpolate import griddata
from rasterio.warp import transform
from rasterio.windows import from_bounds

#%% load all ERA data
start, end = datetime(2025, 1, 1), datetime(2025, 1, 24)

# ERA = []
# # for date in pd.date_range(start_date, end_date):
# print(f"Loading ERA data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
# for date in pd.date_range(start, end):
#     ds = read_ERA_data(date, dirs['ERA'], large_file=True, reduce_dims=False)
#     ERA.append(ds)
# ERA_data = xr.concat(ERA, dim='time')
# del ERA
# print(f"Finished loading ERA data with dimensions: {ERA_data.dims}")

# load only specific section of ERA data (zoom-in around Mt. Helmos)
ERA = []
print(f"Loading ERA data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
for date in pd.date_range(start, end):
    ds = read_ERA_data(date, dirs['ERA'], large_file=True, reduce_dims=True, latlon=([35, 40], [20, 25]) )
    ERA.append(ds)
ERA_data = xr.concat(ERA, dim='time')
del ERA
print(f"Finished loading ERA data with dimensions: {ERA_data.dims}")

ERA_data = ERA_data.assign_coords(altitude=calc_alt(ERA_data['z']))

#%% constants/inputs
levels_to_plot = [850, 700, 500, 300]  # hPa - sea level, ~3km altitude, mid-troposphere, jet stream
lat_h, lon_h = latlon_Helmos
site_altitude_km = radar_altitude / 1000.0

lon_min, lon_max = 20, 25
lat_min, lat_max = 35, 40
dem_path = dirs['DEM']

#%% functions
def get_wind_at_level(ds, level, time):
    """
    Select u and v wind components at a given pressure level and time.
    
    Parameters
    ----------
    ds : xarray.Dataset
    level : int or float
        Pressure level in hPa (e.g. 850, 500, 250)
    time : datetime-like
        datetime.datetime or pandas.Timestamp
    """
    u = ds['u'].sel(isobaricInhPa=level, time=time, method='nearest')
    v = ds['v'].sel(isobaricInhPa=level, time=time, method='nearest')
    return u, v

def thickness_1000_500(ds, level_unused, time):
    z500 = ds.z.sel(time=time, isobaricInhPa=500)
    z1000 = ds.z.sel(time=time, isobaricInhPa=1000)
    return (z500 - z1000) / 9.81

def geopotential_height(ds, level, time):
    return ds.z.sel(isobaricInhPa=level, time=time, method='nearest') / 9.81

def windspeed(ds, level, time):
    """
    Get wind speed at a given pressure level and time.
    """
    u, v = get_wind_at_level(ds, level, time)
    wind_speed = np.sqrt(u**2 + v**2)
    return wind_speed

def relhum(ds, level, time):
    """
    Get relative humidity at a given pressure level and time.
    """
    rh = ds['r'].sel(isobaricInhPa=level, time=time, method='nearest')
    return rh

def getvar(ds, varname, level, time):
    da = ds[varname].sel(isobaricInhPa=level, time=time, method='nearest')
    return da

def temperature(ds, level, time):
    """
    Get temperature at a given pressure level and time.
    """
    t = ds['t'].sel(isobaricInhPa=level, time=time, method='nearest')
    return t

def Xsection_elevation_from_dem(xcoord, coord_unique, coord_slice, dem_path):
    """Extract elevation for a cross-section from a DEM in ETRS89 (arcseconds)"""
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        dem_bounds = src.bounds
        
        if xcoord == 'longitude':
            lat, lon_min, lon_max = coord_unique, coord_slice[0], coord_slice[1]
            lons = np.linspace(lon_min, lon_max, 1000) 
            lats = np.full_like(lons, lat)
        elif xcoord == 'latitude':
            lon, lat_min, lat_max = coord_unique, coord_slice[0], coord_slice[1]
            lats = np.linspace(lat_min, lat_max, 1000)
            lons = np.full_like(lats, lon)
        else:
            raise ValueError("xcoord must be 'latitude' or 'longitude'")
            
        lons_arcsec = lons * 3600
        lats_arcsec = lats * 3600

        pixel_width = dem_transform.a    # 2.0 arcsec
        pixel_height = -dem_transform.e  # 2.0 arcsec (stored as -2.0)
        left = dem_transform.c
        top = dem_transform.f

        cols = ((lons_arcsec - left) / pixel_width).astype(int)
        rows = ((top - lats_arcsec) / pixel_height).astype(int)

        n_rows, n_cols = src.height, src.width
        buf = 5
        row_min = max(rows.min() - buf, 0)
        row_max = min(rows.max() + buf, n_rows - 1)
        col_min = max(cols.min() - buf, 0)
        col_max = min(cols.max() + buf, n_cols - 1)

        window = rasterio.windows.Window(
            col_off=col_min,
            row_off=row_min,
            width=col_max - col_min + 1,
            height=row_max - row_min + 1,
        )
        dem_crop = src.read(1, window=window)
        dem_crop = np.where(dem_crop == 3.2767e+04, 0, dem_crop)

        # Offset row/col indices to be relative to the crop
        rows_crop = rows - row_min
        cols_crop = cols - col_min

        valid = (
            (rows_crop >= 0) & (rows_crop < dem_crop.shape[0]) &
            (cols_crop >= 0) & (cols_crop < dem_crop.shape[1])
        )

        elevations = np.full(len(lons), np.nan)
        elevations[valid] = dem_crop[rows_crop[valid], cols_crop[valid]]

    if xcoord == 'longitude':
        coord_range = lons
    else:
        coord_range = lats

    return coord_range, elevations

def plot_winds_level(u, v, level, time, extent=None, add_streamlines=False, add_contours=True):
    """
    Plot synoptic-scale wind vectors with color-coded wind speed.
    
    Parameters
    ----------
    u, v : xarray.DataArray
        Wind components
    level : int
        Pressure level in hPa
    time : datetime-like
    extent : list, optional
        [lon_min, lon_max, lat_min, lat_max]
    add_streamlines : bool
        Add streamlines instead of/in addition to quivers
    add_contours : bool
        Add wind speed contours
    """
    wind_speed = np.sqrt(u**2 + v**2)
    
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    ax.coastlines(linewidth=0.8, color='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', zorder=3)
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', alpha=0.5, zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', alpha=0.3, zorder=1)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                      alpha=0.5, linestyle='--', zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    
    levels_speed = np.arange(0, 41, 5)
    cf = ax.contourf(u.longitude, u.latitude, wind_speed,
                     levels=levels_speed,
                     cmap='magma_r',
                    #  cmap='Blues',
                     alpha=0.7,
                     transform=ccrs.PlateCarree(),
                     extend='max',
                     zorder=2)
    
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', 
                        pad=0.05, shrink=0.8)
    cbar.set_label('Wind Speed (m s$^{-1}$)', fontsize=11, weight='bold')
    
    if add_contours:
        cs = ax.contour(u.longitude, u.latitude, wind_speed,
                       levels=levels_speed[::2],
                       colors='black',
                       linewidths=0.5,
                       alpha=0.4,
                       transform=ccrs.PlateCarree(),
                       zorder=4)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
    
    if add_streamlines:
        ax.streamplot(u.longitude.values, u.latitude.values,
                     u.values, v.values,
                     density=2,
                     color='black',
                     linewidth=1,
                     arrowsize=1.5,
                     transform=ccrs.PlateCarree(),
                     zorder=5)
    else:
        skip = (slice(None, None, 3), slice(None, None, 3))
        
        q = ax.quiver(
            u.longitude[skip[1]],
            u.latitude[skip[0]],
            u.values[skip],
            v.values[skip],
            # wind_speed.values[skip],
            scale=500,
            scale_units='width',
            width=0.003,
            headwidth=3,
            headlength=5,
            color='black',
            # cmap='Blues',
            transform=ccrs.PlateCarree(),
            zorder=5,
            alpha=0.8
        )
        
        # Quiver key
        qk = ax.quiverkey(q, 0.92, 0.05, 20, "20 m s$^{-1}$",
                         labelpos='E', coordinates='axes',
                         fontproperties={'size': 10, 'weight': 'bold'})
    
    time_str = pd.to_datetime(time).strftime('%Y-%m-%d %H:%M UTC')
    ax.set_title(
        f"ERA5 {level} hPa wind field",
        fontsize=14, weight='bold', loc='left'
    )
    ax.set_title(
        f"{time_str}",
        fontsize=12, loc='right', weight='bold',#style='italic'
    )    
    fig.tight_layout()
    return fig, ax

def plot_multilevel_winds(ds, levels, time, extent=None):
    """
    Plot wind fields at multiple pressure levels side by side.
    """
    n_levels = len(levels)
    fig = plt.figure(figsize=(7*n_levels, 6))
    
    # Create a list to store contourf objects for shared colorbar
    cfs = []
    
    for i, level in enumerate(levels, 1):
        u, v = get_wind_at_level(ds, level, time)
        wind_speed = np.sqrt(u**2 + v**2)
        
        ax = fig.add_subplot(1, n_levels, i, projection=ccrs.PlateCarree())
        
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        ax.coastlines(linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', alpha=0.5)
        
        cf = ax.contourf(u.longitude, u.latitude, wind_speed,
                        levels=np.arange(0, 41, 5),
                        cmap='magma_r',
                        alpha=0.7,
                        transform=ccrs.PlateCarree(),
                        extend='max')
        cfs.append(cf)
        
        skip = (slice(None, None, 4), slice(None, None, 4))
        ax.quiver(
            u.longitude[skip[1]], u.latitude[skip[0]],
            u.values[skip], v.values[skip],
            scale=400, width=0.003,
            transform=ccrs.PlateCarree(),
            alpha=0.7
        )
        
        ax.set_title(f"{level} hPa", fontsize=12, weight='bold')
    
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cfs[0], cax=cbar_ax, orientation='vertical')
    cbar.set_label('Wind Speed (m s$^{-1}$)', fontsize=11, weight='bold')
    
    time_str = pd.to_datetime(time).strftime('%Y-%m-%d %H:%M UTC')
    fig.suptitle(f"ERA5 wind fields {time_str}", fontsize=14, weight='bold', y=0.85)    
    fig.tight_layout(rect=[0, 0, 0.90, 0.96])
    return fig

def plot_multilevel_synoptic(ds, levels, time, scalar_func, scalar_label, scalar_levels, scalar_title, 
                             cmap, cmap_norm=False, wind=True, wind_level=None, extent=None,
                             contour_func=None, contour_level=None, contour_levels=None):
    """
    Plot wind fields at multiple pressure levels side by side.
    """
    n_levels = len(levels)
    fig = plt.figure(figsize=(7*n_levels, 6), constrained_layout=True)
    cfs = []
    
    if cmap_norm:
        norm = mcolors.TwoSlopeNorm(
            vmin=scalar_levels.min(),
            vcenter=0.0,
            vmax=scalar_levels.max()
        )
    else:
        norm = None

    for i, level in enumerate(levels, 1):
        scalar = scalar_func(ds, level, time)
        
        ax = fig.add_subplot(1, n_levels, i, projection=ccrs.PlateCarree())
        
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        ax.coastlines(linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', alpha=0.5)
        
        cf = ax.contourf(scalar.longitude, scalar.latitude, scalar,
                        levels=scalar_levels,
                        cmap=cmap,
                        norm=norm,
                        extend='both',
                        transform=ccrs.PlateCarree(),)
        if contour_func is not None and contour_level is not None:
            contour_field = contour_func(ds, contour_level, time)
            cs = ax.contour(
                contour_field.longitude,
                contour_field.latitude,
                contour_field,
                levels=contour_levels,
                colors='black',
                linewidths=1,
                transform=ccrs.PlateCarree()
            )
            ax.clabel(cs, inline=True, fontsize=8)

        cfs.append(cf)
        
        if wind:
            w_level = wind_level or level
            u, v = get_wind_at_level(ds, w_level, time)
            
            skip = (slice(None, None, 4), slice(None, None, 4))
            ax.quiver(
                u.longitude[skip[1]], u.latitude[skip[0]],
                u.values[skip], v.values[skip],
                scale=400, width=0.003,
                transform=ccrs.PlateCarree(),
                alpha=0.7
            )
        
        if not n_levels == 1:
            ax.set_title(f"{level} hPa", fontsize=12, weight='bold')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.6, color='gray', alpha=0.4, linestyle='--')
        gl.top_labels=True
        gl.right_labels=True
        gl.left_labels=True
        gl.bottom_labels=True
        
        ax.scatter(
            lon_h, lat_h,
            s=250,
            marker='*',
            facecolor='red',
            edgecolor='black',
            linewidth=1.2,
            transform=ccrs.PlateCarree(),
            zorder=10
        )

    # fig.subplots_adjust(right=0.90)
    #cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.8])
    cbar = fig.colorbar(cfs[0], ax=fig.axes, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label(scalar_label, fontsize=11, weight='bold')
    
    time_str = pd.to_datetime(time).strftime('%Y-%m-%d %H:%M UTC')
    fig.suptitle(f"ERA5 {scalar_title} {time_str}", fontsize=14, weight='bold')
    # fig.tight_layout(rect=[0, 0, 0.90, 0.96])
    return fig

def compute_height_pressure_profile(cross, pressure_limits=(1000, 200)):
    """Compute mean geometric height profile and pressure levels for axis labelling."""
    cross_p = cross.isel(time=0).sel(isobaricInhPa=slice(*pressure_limits))
    pressure_levels = cross_p.isobaricInhPa.values.squeeze()
    
    geopotential_height = cross_p['z']
    geometric_height = geopotential_to_height(geopotential_height.metpy.unit_array)
    mid = geometric_height.shape[-1] // 2
    mean_height_km = geometric_height.magnitude[:, mid].squeeze() / 1000.0
    
    return pressure_levels, mean_height_km

def plot_vertical_cross_section(cross, t, scalar_func, scalar_label, scalar_levels, scalar_title, cmap, dem_input, height_profile, cmap_norm=False, wind=False, wind_components=None, xcoord='longitude', pressure_limits=(1000, 200)):
    """
    Plot a vertical cross-section (pressure vs longitude/latitude).
    """
    site_coord = lon_h if xcoord == 'longitude' else lat_h
    coords, surface_elevations = dem_input
    surface_km = surface_elevations / 1000.0
    pressure_levels, mean_height_km = height_profile

    z_min = 0.0
    z_max = 12.0
    dz = 0.25   # 250 m resolution (adjust as desired)
    height_grid_km = np.arange(z_min, z_max + dz, dz)

    cross_t = cross.sel(time=t).sel(isobaricInhPa=slice(*pressure_limits))
    pressure_levels = cross_t.isobaricInhPa.values
    era_coords = cross_t[xcoord].values
    era_surface_km = np.interp(era_coords, coords, surface_km)

    extra_dims = [d for d in cross_t['altitude'].dims if d != 'isobaricInhPa']
    alt_profile_km = cross_t['altitude'].mean(dim=extra_dims).values / 1000.0
    
    sort_idx = np.argsort(alt_profile_km)
    heights_sorted_km = alt_profile_km[sort_idx]
    pressures_sorted = pressure_levels[sort_idx]
    
    cross_t = cross_t.assign_coords(height_km=('isobaricInhPa', alt_profile_km))
    cross_t = cross_t.swap_dims({'isobaricInhPa': 'height_km'})
    cross_t = cross_t.sortby('height_km')
    
    height_grid_km = np.arange(alt_profile_km.min(), z_max + dz, dz)
    cross_t = cross_t.interp(height_km=height_grid_km)
    
    time_str = pd.to_datetime(t).strftime('%Y-%m-%d %H:%M UTC')
    label = 'E-W' if xcoord == 'longitude' else 'N-S'

    scalar = scalar_func(cross_t)  # (pressure, lon/lat)

    height_ticks_km = np.arange(0, mean_height_km.max() + 1, 2)

    if cmap_norm:
        norm = mcolors.TwoSlopeNorm(vmin=scalar_levels.min(), vcenter=0.0, vmax=scalar_levels.max())
    else:
        norm = None
 
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    cf = ax.contourf(
        cross_t[xcoord],
        height_grid_km,
        scalar,
        levels=scalar_levels,
        cmap=cmap,
        norm=norm,
        extend='both'
    )

    if wind and wind_components is not None:
        comp1, comp2 = wind_components
        skip = (slice(None, None, 2), slice(None, None, 3))
        q = ax.quiver(
            cross_t[xcoord].values[skip[1]],
            height_grid_km[skip[0]],
            cross_t[comp1].values[skip],
            cross_t[comp2].values[skip],
            scale=200,
            width=0.003
        )
        ax.quiverkey(q, X=0.1, Y=1.05, U=10, label='10 m/s', labelpos='E', coordinates='axes')

    ax.plot(coords, surface_km, color='saddlebrown', linewidth=2)
    ax.fill_between(coords, 0, surface_km, color='saddlebrown', alpha=0.9)
    ax.plot(era_coords, era_surface_km, color='gray', linewidth=2, linestyle='--', label='ERA topography')
    ax.plot(site_coord, site_altitude_km, marker='*', markersize=12, color='red', markeredgecolor='black', markeredgewidth=0.5, label='Mt. Helmos')

    ax.set_ylim(0, z_max)
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel(f"{xcoord.capitalize()} (°)")
    ax.grid(True, linestyle='--', alpha=0.4)

    # Secondary y-axis: pressure ticks at corresponding heights
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    pressure_ticks = [1000, 850, 700, 500, 300, 200]
    tick_heights_km = np.interp(pressure_ticks, pressures_sorted[::-1], heights_sorted_km[::-1])
    ax2.set_yticks(tick_heights_km)
    ax2.set_yticklabels([f"{p}" for p in pressure_ticks])
    ax2.set_ylabel("Pressure (hPa)")

    cbar = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.08)
    cbar.set_label(scalar_label, fontsize=11, weight='bold')

    time_str = pd.to_datetime(t).strftime('%Y-%m-%d %H:%M UTC')
    label = 'E-W' if xcoord == 'longitude' else 'N-S'
    ax.set_title(f"{label} {scalar_title} cross-section\n{time_str}", weight='bold')
    return fig
    
#%% plot synoptic ERA data (single level + multilevel winds)
# variables_to_plot = ['T', 'RH', 'U', 'V', 'omega']

# ttime = pd.Timestamp('2025-01-01 10:00')
# level = 850

# for t in pd.date_range('2025-01-01', '2025-01-24', freq='1H'):
#     date_dir = f"{os.path.dirname(dirs['ERA'])}/plots/{t.strftime('%Y%m%d')}"
#     os.makedirs(date_dir, exist_ok=True)
#     if t.hour % 6 == 0:
#         print(f"Plotting ERA wind field data for {t.strftime('%Y-%m-%d %H:%M')}")
#     outname_multilevel = f"{date_dir}/wind_multilevel_{t.strftime('%Y%m%d_%H%M')}.png"
#     if os.path.exists(outname_multilevel):
#         print(f"{outname_multilevel} already exists, skipping...")
#     else:
#         fig_multi = plot_multilevel_winds(ERA_data, levels_to_plot, t)
#         plt.savefig(outname_multilevel, dpi=150)
#         plt.close(fig_multi)
#     for level in levels_to_plot:
#         outname_level = f"{date_dir}/wind_{level}hPa_{t.strftime('%Y%m%d_%H%M')}.png"
#         if os.path.exists(outname_level):
#             print(f"{outname_level} already exists, skipping...")
#             continue
#         else:
#             u, v = get_wind_at_level(ERA_data, level, t)
#             fig_level, ax_level = plot_winds_level(u, v, level, t)
#             plt.savefig(outname_level, dpi=150)
#             plt.close(fig_level)


# #%% make multilevel plots (only) 
# for t in pd.date_range('2025-01-01', '2025-01-24', freq='1H'):
#     date_dir = f"{os.path.dirname(dirs['ERA'])}/plots/small/{t.strftime('%Y%m%d')}"
#     os.makedirs(date_dir, exist_ok=True)
#     if t.hour % 6 == 0:
#         print(f"Plotting ERA synoptic data for {t.strftime('%Y-%m-%d %H:%M')}")
        
#     # plot relative humidity
#     outname_relhum = f"{date_dir}/relhum_{t.strftime('%Y%m%d_%H%M')}.png"
#     if os.path.exists(outname_relhum):
#         print(f"{outname_relhum} already exists, skipping...")
#         continue
#     fig = plot_multilevel_synoptic(
#         ERA_data,
#         levels=levels_to_plot,
#         time=t,
#         scalar_func=relhum,
#         scalar_label="Relative Humidity (%)",
#         scalar_levels=np.arange(0, 101, 10),
#         scalar_title='relative humidity',
#         cmap="YlGnBu",
#         wind=True,
#         wind_level=700
#     )
#     plt.savefig(outname_relhum, dpi=150, bbox_inches='tight', facecolor='w')
#     plt.close(fig)

#     # plot temperature
#     outname_temp = f"{date_dir}/temperature_{t.strftime('%Y%m%d_%H%M')}.png"
#     if os.path.exists(outname_temp):
#         print(f"{outname_temp} already exists, skipping...")
#         continue
#     fig = plot_multilevel_synoptic(
#         ERA_data,
#         levels=[850, 700, 500],
#         time=t,
#         scalar_func=temperature,    
#         scalar_label="Temperature (°C)",
#         scalar_levels=np.arange(-40, 21, 5),
#         scalar_title='temperature',
#         cmap="RdBu_r",
#         wind=True
#     )
#     plt.savefig(outname_temp, dpi=150, bbox_inches='tight', facecolor='w')
#     plt.close(fig)

#     # plot cloud cover
#     outname_cc = f"{date_dir}/cloudcover_{t.strftime('%Y%m%d_%H%M')}.png"
#     if os.path.exists(outname_cc):
#         print(f"{outname_cc} already exists, skipping...")
#         continue
#     fig = plot_multilevel_synoptic(
#         ERA_data,
#         levels=levels_to_plot,
#         time=t,
#         scalar_func=lambda ds, level, time: getvar(ds, 'cc', level, time),
#         scalar_label="Cloud cover (%)",
#         scalar_levels=np.arange(0, 101, 5),
#         scalar_title='cloud cover',
#         cmap="YlGnBu",
#     )
#     plt.savefig(outname_cc, dpi=150, bbox_inches='tight', facecolor='w')
#     plt.close(fig)

#     # plot vertical wind
#     outname_vwind = f"{date_dir}/verticalwind_{t.strftime('%Y%m%d_%H%M')}.png"
#     if os.path.exists(outname_vwind):
#         print(f"{outname_vwind} already exists, skipping...")
#         continue
#     fig = plot_multilevel_synoptic(
#         ERA_data,
#         levels=levels_to_plot,
#         time=t,
#         scalar_func=lambda ds, level, time: getvar(ds, 'w', level, time),
#         scalar_label="Vertical wind (m s$^{-1}$)",
#         scalar_levels=np.arange(-1.0, 1.05, 0.1),
#         scalar_title='vertical wind',
#         cmap="RdBu_r",
#         cmap_norm=True,
#     )
#     plt.savefig(outname_vwind, dpi=150, bbox_inches='tight', facecolor='w')
#     plt.close(fig)

#     # # plot pressure
#     # outname_pressure = f"{os.path.dirname(dirs['ERA'])}/plots/{t.strftime('%Y%m%d')}/pressure_{t.strftime('%Y%m%d_%H%M')}.png"
#     # fig = plot_multilevel_synoptic(
#     #     ERA_data,
#     #     levels=levels_to_plot,
#     #     time=t,
#     #     scalar_func=geopotential_height,
#     #     scalar_label="Geopotential height",
#     #     scalar_levels=np.arange(1000, 6000, 100),
#     #     scalar_title='geopotential height',
#     #     cmap="viridis",
#     #     wind=True
#     # )

#     # plot 500 hPa vorticitiy
#     outname_vo = f"{date_dir}/vorticity_{t.strftime('%Y%m%d_%H%M')}.png"
#     if os.path.exists(outname_vo):
#         print(f"{outname_vo} already exists, skipping...")
#         continue
#     fig = plot_multilevel_synoptic(
#         ERA_data,
#         levels=levels_to_plot,
#         time=t,
#         scalar_func=lambda ds, level, time: ds.vo.sel(time=time, isobaricInhPa=level)*1e5,
#         scalar_label="Relative Vorticity (1e-5 s$^{-1}$)",
#         scalar_levels=np.arange(-20, 21, 2),
#         scalar_title='relative vorticity',
#         cmap="RdBu_r",
#         cmap_norm=True,
#         wind=True
#     )
#     plt.savefig(outname_vo, dpi=150, bbox_inches='tight', facecolor='w')
#     plt.close(fig)

#     # plot geopotential thickness
#     outname_frontal = f"{date_dir}/frontal_{t.strftime('%Y%m%d_%H%M')}.png"
#     if os.path.exists(outname_frontal):
#         print(f"{outname_frontal} already exists, skipping...")
#         continue
#     fig = plot_multilevel_synoptic(
#         ERA_data,
#         levels=[500],  # just to create one subplot
#         time=t,
#         scalar_func=thickness_1000_500,
#         scalar_label="1000–500 hPa Thickness (m)",
#         scalar_levels=np.arange(5280, 5640, 20),
#         scalar_title="1000–500 hPa Thickness",
#         cmap="viridis",
#         wind=True,
#         wind_level=850
#     )
#     plt.savefig(outname_frontal, dpi=150, bbox_inches='tight', facecolor='w')
#     plt.close(fig)


#%% cross-sections
if ERA_data.latitude[0] > ERA_data.latitude[-1]:
    lat_slice = slice(lat_max, lat_min)
else:
    lat_slice = slice(lat_min, lat_max)

if ERA_data.longitude[0] > ERA_data.longitude[-1]:
    lon_slice = slice(lon_max, lon_min)
else:
    lon_slice = slice(lon_min, lon_max)

cross_lat = ERA_data.sel(longitude=lon_h, method='nearest').sel(latitude=lat_slice)
cross_lon = ERA_data.sel(latitude=lat_h, method='nearest').sel(longitude=lon_slice)

dem_lat = Xsection_elevation_from_dem('latitude', lon_h, (lat_min, lat_max), dem_path)
dem_lon = Xsection_elevation_from_dem('longitude', lat_h, (lon_min, lon_max), dem_path)

height_profile = compute_height_pressure_profile(cross_lat, pressure_limits=(1000, 200))

for t in pd.date_range('2025-01-01', '2025-01-24', freq='1H'):
    date_dir = f"{os.path.dirname(dirs['ERA'])}/plots/small/{t.strftime('%Y%m%d')}"
    os.makedirs(date_dir, exist_ok=True)
    if t.hour % 6 == 0:
        print(f"Plotting ERA synoptic data for {t.strftime('%Y-%m-%d %H:%M')}")

    # vertical winds
    outname_vw_NS = f"{date_dir}/NS_verticalwind_{t.strftime('%Y%m%d_%H%M')}.png"
    if os.path.exists(outname_vw_NS):
        print(f"{outname_vw_NS} already exists, skipping...")
        continue
    fig = plot_vertical_cross_section(
        cross_lon,
        t,
        scalar_func=lambda ds: ds['w'],
        scalar_label="Vertical wind (m s$^{-1}$)",
        scalar_levels=np.arange(-1.0, 1.05, 0.1),
        scalar_title='vertical wind',
        cmap="RdBu_r",
        dem_input=dem_lon,
        height_profile=height_profile,
        cmap_norm=True,
        wind=True,
        wind_components=('u', 'w'),
        xcoord='longitude'
    )
    plt.savefig(outname_vw_NS, dpi=150, bbox_inches='tight', facecolor='w')
    plt.close(fig)
    
    outname_vw_EW = f"{date_dir}/EW_verticalwind_{t.strftime('%Y%m%d_%H%M')}.png"
    if os.path.exists(outname_vw_EW):
        print(f"{outname_vw_EW} already exists, skipping...")
        continue
    fig = plot_vertical_cross_section(
        cross_lat,
        t,
        scalar_func=lambda ds: ds['w'],
        scalar_label="Vertical wind (m s$^{-1}$)",
        scalar_levels=np.arange(-1.0, 1.05, 0.1),
        scalar_title='vertical wind',
        cmap="RdBu_r",
        dem_input=dem_lat,
        height_profile=height_profile,
        cmap_norm=True,
        wind=True,
        wind_components=('u', 'w'),
        xcoord='latitude'
    )
    plt.savefig(outname_vw_EW, dpi=150, bbox_inches='tight', facecolor='w')
    plt.close(fig)

    # relative humidity
    outname_relhum_NS = f"{date_dir}/NS_relhum_{t.strftime('%Y%m%d_%H%M')}.png"
    if os.path.exists(outname_relhum_NS):
        print(f"{outname_relhum_NS} already exists, skipping...")
        continue
    fig = plot_vertical_cross_section(
        cross_lat,
        t,
        scalar_func=lambda ds: ds['r'],
        scalar_label="Relative Humidity (%)",
        scalar_levels=np.arange(0, 101, 10),
        scalar_title='relative humidity',
        cmap="YlGnBu",
        dem_input=dem_lat,
        height_profile=height_profile,
        wind=True,
        wind_components=('u', 'w'),
        xcoord='latitude'
    )
    plt.savefig(outname_relhum_NS, dpi=150, bbox_inches='tight', facecolor='w')
    plt.close(fig)

    outname_relhum_EW = f"{date_dir}/EW_relhum_{t.strftime('%Y%m%d_%H%M')}.png"
    if os.path.exists(outname_relhum_EW):
        print(f"{outname_relhum_EW} already exists, skipping...")
        continue
    fig = plot_vertical_cross_section(
        cross_lon,
        t,
        scalar_func=lambda ds: ds['r'],
        scalar_label="Relative Humidity (%)",
        scalar_levels=np.arange(0, 101, 10),
        scalar_title='relative humidity',
        cmap="YlGnBu",
        dem_input=dem_lon,
        height_profile=height_profile,
        wind=True,
        wind_components=('u', 'w'),
        xcoord='longitude'
    )
    plt.savefig(outname_relhum_EW, dpi=150, bbox_inches='tight', facecolor='w')
    plt.close(fig)

    # plot temperature
    outname_temp_NS = f"{date_dir}/NS_temperature_{t.strftime('%Y%m%d_%H%M')}.png"
    if os.path.exists(outname_temp_NS):
        print(f"{outname_temp_NS} already exists, skipping...")
        continue
    fig = plot_vertical_cross_section(
        cross_lat,
        t,
        scalar_func=lambda ds: ds['t'],
        scalar_label="Temperature (°C)",
        scalar_levels=np.arange(-40, 21, 5),
        scalar_title='temperature',
        cmap="RdBu_r",
        dem_input=dem_lat,
        height_profile=height_profile,
        wind=True,
        wind_components=('u', 'w'),
        xcoord='latitude'
    )
    plt.savefig(outname_temp_NS, dpi=150, bbox_inches='tight', facecolor='w')
    plt.close(fig)

    outname_temp_EW = f"{date_dir}/EW_temperature_{t.strftime('%Y%m%d_%H%M')}.png"
    if os.path.exists(outname_temp_EW):
        print(f"{outname_temp_EW} already exists, skipping...")
        continue
    fig = plot_vertical_cross_section(
        cross_lon,
        t,
        scalar_func=lambda ds: ds['t'],
        scalar_label="Temperature (°C)",
        scalar_levels=np.arange(-40, 21, 5),
        scalar_title='temperature',
        cmap="RdBu_r",
        dem_input=dem_lon,
        height_profile=height_profile,
        wind=True,
        wind_components=('u', 'w'),
        xcoord='longitude'
    )
    plt.savefig(outname_temp_EW, dpi=150, bbox_inches='tight', facecolor='w')
    plt.close(fig)