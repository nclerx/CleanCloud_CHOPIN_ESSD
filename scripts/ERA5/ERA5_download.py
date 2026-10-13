#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:43:29 2025

To be able to connect to the Copernicus CDS in this script, the folder from which it is run needs to contain a file called ".cdsapirc", with the following details:
url: https://cds.climate.copernicus.eu/api
key: <your-personal-access-token>
The token is on your profile page at cds.climate.copernicus.eu, after you log in. For more details/information, see https://cds.climate.copernicus.eu/how-to-api. 

@author: clerx
"""
#%% imports 
import cdsapi

from src import os, datetime, time

#%% date range & directory definitions
starttime = datetime(2025, 1, 1)
endtime = datetime(2025, 1, 24)

base_dir = '/t5500/ltenas7/campaigns/CLEANCLOUD_CHOPIN_2024/ERA5/grib'

#%% download ERA5 at pressure levels
client = cdsapi.Client()
# Static request components
dataset = "reanalysis-era5-pressure-levels"
variables = [
    "geopotential", "fraction_of_cloud_cover", "relative_humidity",
    "specific_humidity", "temperature", "u_component_of_wind",
    "v_component_of_wind", "vertical_velocity", "divergence", "vorticity"
]   # corresponding short names: [z, cc, r, q, t, u, v, w, d, vo] table 9 of https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Parameterlistings
times = [
    f"{hour:02d}:00" for hour in range(24)
]
pressure_levels = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100",
    "125", "150", "175", "200", "225", "250", "300", "350", "400",
    "450", "500", "550", "600", "650", "700", "750", "775", "800",
    "825", "850", "875", "900", "925", "950", "975", "1000"
]
area = [45, 10, 30, 35]  # North, West, South, East

# Iterate through each day in the date range
current_date = starttime
while current_date <= endtime:
    year = str(current_date.year)
    month = f"{current_date.month:02d}"
    day = f"{current_date.day:02d}"

    request = {
        "product_type": "reanalysis",
        "variable": variables,
        "pressure_level": pressure_levels,
        "year": year,
        "month": month,
        "day": day,
        "time": times,
        "data_format": "grib",
        "area": area
    }

    filename = f"{base_dir}/era5_pressurelevels_{year}{month}{day}_large.grib"
    if not os.path.exists(filename):
        print(f"Requesting data for {year}-{month}-{day}...")
        client.retrieve(dataset, request).download(filename)
        print(f"Saved to {filename}")
    else:
        print(f"GRIB file {os.path.basename(filename)} already exists, skipping.")

    current_date += timedelta(days=1)

