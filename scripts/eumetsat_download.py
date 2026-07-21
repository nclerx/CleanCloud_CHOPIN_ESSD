#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 10:36:34 2026
download eumetsat geostationary geotiffs

@author: clerx
"""

import os, requests, rasterio
from datetime import datetime, timedelta
from eumdac import DataStore

consumer_key = '4BJ6UoVWi1fP06ou9C9UPr0JC0oa'
consumer_secret = '_bTz9YzwYYMK0RueYViwnuq96UEa'

credentials = (consumer_key, consumer_secret)

token = eumdac.AccessToken(credentials)

print(f"This token '{token}' expires {token.expiration}")
API_KEY = "YOUR_EUMETSAT_API_KEY"  # Your EUMETSAT API key
DOWNLOAD_DIR = "./meteosat"
SATELLITE_PRODUCT = "MSG"           # Meteosat Second Generation
CHANNELS = ["IR_108", "WV_062"]     # IR 10.8 µm, Water Vapour 6.2 µm
START_DATE = "2025-01-06"
END_DATE = "2025-01-09"
TIME_STEP_HOURS = 1                  # hourly steps
# Peloponnese bounding box
LAT_MIN = 30
LAT_MAX = 45
LON_MIN = 10
LON_MAX = 35

# Create download folder
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def daterange(start_date, end_date, step_hours=1):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(hours=step_hours)

def search_file_url(channel, date_time):
    """
    Search EUMETSAT catalog for the file URL of a specific datetime and channel.
    """
    search_url = "https://eoapi.eumetsat.int/datastore/search"  # EUMETSAT API endpoint
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {
        "dataset": SATELLITE_PRODUCT,
        "channel": channel,
        "startTime": date_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": date_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "GeoTIFF"
    }
    
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    results = response.json()
    if results.get("count", 0) > 0:
        return results["features"][0]["properties"]["download_url"]
    else:
        print(f"No file found for {channel} at {date_time}")
        return None

def download_and_clip(url, channel, date_time):
    """
    Download GeoTIFF, clip to bounding box, and save.
    """
    local_path = os.path.join(DOWNLOAD_DIR, f"{channel}_{date_time.strftime('%Y%m%d_%H%M')}.tif")
    
    # Download
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Clip to bounding box using rasterio
    with rasterio.open(local_path) as src:
        window = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
        subset = src.read(window=window)
        profile = src.profile
        profile.update({
            "height": subset.shape[1],
            "width": subset.shape[2],
            "transform": rasterio.windows.transform(window, src.transform)
        })

        clipped_path = local_path.replace(".tif", "_peloponnese.tif")
        with rasterio.open(clipped_path, "w", **profile) as dst:
            dst.write(subset)
    
    os.remove(local_path)  # remove full disk to save space
    return clipped_path

# -------------------------------
# MAIN LOOP
# -------------------------------
start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

for dt in tqdm(list(daterange(start_dt, end_dt, TIME_STEP_HOURS))):
    for channel in CHANNELS:
        try:
            file_url = search_file_url(channel, dt)
            if file_url:
                clipped_file = download_and_clip(file_url, channel, dt)
                print(f"Saved: {clipped_file}")
        except Exception as e:
            print(f"Error for {channel} at {dt}: {e}")

print("All downloads completed!")
