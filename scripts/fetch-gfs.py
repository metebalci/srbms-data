#!/usr/bin/env python3
# Copyright (c) 2026 Mete Balci. All Rights Reserved.
"""
Fetch full-atmosphere wind data from NOAA GFS model (surface to ~80km).

Data source: NOAA Global Forecast System (GFS) 0.25° resolution
Available on AWS Open Data: https://registry.opendata.aws/noaa-gfs-bdp-pds/

Pressure levels: 1000 hPa (surface) to 0.01 hPa (~80km)
This replaces ERA5 reanalysis with real-time forecast data.

Output: gfs-wind.json + gfs-wind.bin with full atmosphere wind grid
"""

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
import requests
import numpy as np
from pyproj import Transformer

# EPSG:3035 (ETRS89-LAEA) transformer for European projection
wgs84_to_laea = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

# GFS data on AWS S3
GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# Europe bounds (WGS84) - covers DACH and future expansion
# Roughly: Portugal to Poland, Sicily to Norway
EUROPE_BOUNDS = {
    "west": -11.0,
    "east": 25.0,
    "south": 35.0,
    "north": 60.0
}

# Scale factor for Int16 encoding
WIND_SCALE = 100  # 0.01 m/s precision

# GFS pressure levels covering full atmosphere (hPa) with approximate altitudes (m)
# Standard atmosphere: h = -8500 * ln(p/1013.25)
# Selected to match ERA5 coverage but with GFS real-time data
GFS_PRESSURE_LEVELS = [
    # Troposphere (surface to ~12km)
    (1000, 110),    # Near surface
    (925, 760),     # Lower troposphere
    (850, 1500),
    (700, 3000),
    (500, 5500),    # Mid troposphere
    (300, 9200),    # Upper troposphere
    # Tropopause / jet stream (~10-12km)
    (250, 10400),
    (200, 11800),
    # Stratosphere (12-50km)
    (150, 13600),
    (100, 16200),
    (70, 18600),
    (50, 21000),
    (30, 24000),
    (20, 27000),
    (10, 31000),
    (7, 34000),
    (5, 37000),
    (3, 40000),
    (2, 43000),
    (1, 48000),
    # Upper stratosphere / mesosphere (50-80km)
    # GFS extends to 0.01 hPa with levels: 0.7, 0.4, 0.2, 0.1, 0.01
    # Note: Using available GFS levels, some may need adjustment
]

# Extended levels for upper atmosphere (GFS provides these)
GFS_UPPER_LEVELS = [
    (0.7, 52000),   # 0.7 hPa ≈ 52 km
    (0.4, 56000),   # 0.4 hPa ≈ 56 km
    (0.2, 60000),   # 0.2 hPa ≈ 60 km
    # 0.1 hPa may not be available, skip to 0.01
    (0.01, 80000),  # 0.01 hPa ≈ 80 km (mesopause)
]

# All levels combined
ALL_PRESSURE_LEVELS = GFS_PRESSURE_LEVELS + GFS_UPPER_LEVELS


def get_latest_gfs_cycle() -> Tuple[str, str, datetime]:
    """Get the latest available GFS cycle.

    GFS runs 4 times daily: 00, 06, 12, 18 UTC
    Data becomes available ~3.5-4 hours after cycle time.

    Returns:
        Tuple of (date_str YYYYMMDD, cycle_str HH, cycle_datetime)
    """
    now = datetime.now(timezone.utc)

    # Try cycles from most recent to oldest
    for hours_ago in range(0, 24, 6):
        check_time = now - timedelta(hours=hours_ago)
        cycle_hour = (check_time.hour // 6) * 6
        cycle_time = check_time.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

        # Check if this cycle's data is likely available (4+ hours after cycle)
        if now - cycle_time >= timedelta(hours=4):
            date_str = cycle_time.strftime("%Y%m%d")
            cycle_str = f"{cycle_hour:02d}"
            return date_str, cycle_str, cycle_time

    # Fallback to 12 hours ago
    fallback = now - timedelta(hours=12)
    cycle_hour = (fallback.hour // 6) * 6
    cycle_time = fallback.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    return cycle_time.strftime("%Y%m%d"), f"{cycle_hour:02d}", cycle_time


def get_target_forecast_hour(cycle_time: datetime) -> Tuple[int, datetime]:
    """Calculate the forecast hour to fetch for the next full hour.

    We want the forecast valid for the next full hour from now.
    E.g., if running at 09:05 UTC with 00Z cycle, we want f010 for 10:00 UTC.

    Returns:
        Tuple of (forecast_hour, target_datetime)
    """
    now = datetime.now(timezone.utc)

    # Target is the next full hour
    target_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # Calculate hours between cycle and target
    delta = target_time - cycle_time
    forecast_hour = int(delta.total_seconds() // 3600)

    # Clamp to valid GFS forecast range (0-384 hours)
    forecast_hour = max(0, min(384, forecast_hour))

    return forecast_hour, target_time


def get_gfs_file_url(date_str: str, cycle: str, forecast_hour: int = 0) -> str:
    """Construct GFS file URL on AWS S3."""
    return f"{GFS_BASE_URL}/gfs.{date_str}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"


def get_gfs_index_url(gfs_url: str) -> str:
    """Get the .idx index file URL for a GFS file."""
    return gfs_url + ".idx"


def parse_gfs_index(idx_content: str) -> List[dict]:
    """Parse GFS .idx file to get byte ranges for specific variables."""
    entries = []
    lines = idx_content.strip().split('\n')

    for i, line in enumerate(lines):
        parts = line.split(':')
        if len(parts) >= 5:
            entry = {
                'num': int(parts[0]),
                'start': int(parts[1]),
                'var': parts[3],
                'level': parts[4],
                'end': None
            }
            entries.append(entry)

    # Fill in end byte positions
    for i in range(len(entries) - 1):
        entries[i]['end'] = entries[i + 1]['start'] - 1

    if entries:
        entries[-1]['end'] = entries[-1]['start'] + 500000

    return entries


def find_variable_range(entries: List[dict], var_name: str, level_mb: float) -> Optional[Tuple[int, int]]:
    """Find byte range for a specific variable and level."""
    # Handle fractional pressure levels
    if level_mb < 1:
        level_str = f"{level_mb} mb"
    else:
        level_str = f"{int(level_mb)} mb"

    for entry in entries:
        if entry['var'] == var_name and entry['level'] == level_str:
            return entry['start'], entry['end']

    return None


def download_byte_range(url: str, start: int, end: int) -> Optional[bytes]:
    """Download a specific byte range from a URL."""
    headers = {'Range': f'bytes={start}-{end}'}

    try:
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code in (200, 206):
            return response.content
        else:
            print(f"  Download failed: HTTP {response.status_code}", flush=True)
            return None
    except Exception as e:
        print(f"  Download error: {e}", flush=True)
        return None


def extract_grib_data(grib_bytes: bytes, bounds: dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract data from GRIB2 bytes for the Europe region."""
    import eccodes

    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as f:
        f.write(grib_bytes)
        temp_path = f.name

    try:
        with open(temp_path, 'rb') as f:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                return None

            try:
                ni = eccodes.codes_get(gid, 'Ni')
                nj = eccodes.codes_get(gid, 'Nj')
                lat_first = eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
                lat_last = eccodes.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
                lon_first = eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
                lon_last = eccodes.codes_get(gid, 'longitudeOfLastGridPointInDegrees')

                values = eccodes.codes_get_values(gid)
                data = values.reshape((nj, ni))

                lats = np.linspace(lat_first, lat_last, nj)
                lons = np.linspace(lon_first, lon_last, ni)

                # Handle longitude wrapping (GFS uses 0-360)
                if lons[-1] > 180:
                    lons = np.where(lons > 180, lons - 360, lons)
                    sort_idx = np.argsort(lons)
                    lons = lons[sort_idx]
                    data = data[:, sort_idx]

                # Extract Europe region with margin
                lat_mask = (lats >= bounds['south'] - 0.5) & (lats <= bounds['north'] + 0.5)
                lon_mask = (lons >= bounds['west'] - 0.5) & (lons <= bounds['east'] + 0.5)

                lat_indices = np.where(lat_mask)[0]
                lon_indices = np.where(lon_mask)[0]

                if len(lat_indices) == 0 or len(lon_indices) == 0:
                    print(f"  Warning: No data in Europe bounds", flush=True)
                    return None

                data_bounded = data[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
                lats_bounded = lats[lat_indices]
                lons_bounded = lons[lon_indices]

                return data_bounded, lats_bounded, lons_bounded

            finally:
                eccodes.codes_release(gid)

    except Exception as e:
        print(f"  GRIB extraction error: {e}", flush=True)
        return None
    finally:
        os.unlink(temp_path)


def compute_laea_bounds(lons: np.ndarray, lats: np.ndarray) -> dict:
    """Compute LAEA (EPSG:3035) bounds from WGS84 grid coordinates.

    Returns bounds that can be used for direct scene coordinate lookups.
    """
    # Convert corner points to LAEA
    corners_lon = [lons[0], lons[-1], lons[0], lons[-1]]
    corners_lat = [lats[0], lats[0], lats[-1], lats[-1]]

    laea_x, laea_y = wgs84_to_laea.transform(corners_lon, corners_lat)

    return {
        "minX": float(min(laea_x)),
        "maxX": float(max(laea_x)),
        "minY": float(min(laea_y)),
        "maxY": float(max(laea_y))
    }


def fetch_gfs_wind() -> Optional[dict]:
    """Fetch full-atmosphere wind data from GFS."""
    print("Fetching NOAA GFS full-atmosphere wind data...", flush=True)

    # Get latest GFS cycle
    date_str, cycle, cycle_time = get_latest_gfs_cycle()
    print(f"  GFS cycle: {date_str} {cycle}Z", flush=True)

    # Get target forecast hour (next full hour)
    forecast_hour, target_time = get_target_forecast_hour(cycle_time)
    print(f"  Forecast hour: f{forecast_hour:03d} (valid {target_time.strftime('%Y-%m-%d %H:%M UTC')})", flush=True)

    # Get file URL and index
    gfs_url = get_gfs_file_url(date_str, cycle, forecast_hour=forecast_hour)
    idx_url = get_gfs_index_url(gfs_url)

    print(f"  Fetching index...", flush=True)

    try:
        idx_response = requests.get(idx_url, timeout=30)
        idx_response.raise_for_status()
        idx_content = idx_response.text
    except Exception as e:
        print(f"  Failed to fetch index: {e}", flush=True)
        return None

    entries = parse_gfs_index(idx_content)
    print(f"  Index has {len(entries)} entries", flush=True)

    # First pass: fetch one level to determine grid dimensions
    test_pressure = 1000
    u_range = find_variable_range(entries, 'UGRD', test_pressure)
    if u_range is None:
        print(f"  Could not find test level {test_pressure} hPa", flush=True)
        return None

    u_bytes = download_byte_range(gfs_url, u_range[0], u_range[1])
    if u_bytes is None:
        return None

    test_result = extract_grib_data(u_bytes, EUROPE_BOUNDS)
    if test_result is None:
        return None

    _, lats, lons = test_result
    rows = len(lats)
    cols = len(lons)

    print(f"  Grid dimensions: {cols}x{rows} (native GFS 0.25°)", flush=True)

    # Build output arrays
    altitude_levels = []
    successful_levels = []

    # First, check which levels are available
    print(f"  Checking available pressure levels...", flush=True)
    for pressure_hpa, altitude_m in ALL_PRESSURE_LEVELS:
        u_range = find_variable_range(entries, 'UGRD', pressure_hpa)
        v_range = find_variable_range(entries, 'VGRD', pressure_hpa)
        if u_range and v_range:
            successful_levels.append((pressure_hpa, altitude_m))

    print(f"  Found {len(successful_levels)} available levels", flush=True)

    num_levels = len(successful_levels)
    total_cells = rows * cols * num_levels

    wind_u = np.zeros(total_cells, dtype=np.float32)
    wind_v = np.zeros(total_cells, dtype=np.float32)

    # Fetch each level
    for level_idx, (pressure_hpa, altitude_m) in enumerate(successful_levels):
        altitude_levels.append(altitude_m)

        # Format pressure for display
        if pressure_hpa < 1:
            pres_str = f"{pressure_hpa} hPa"
        else:
            pres_str = f"{int(pressure_hpa)} hPa"

        print(f"  Fetching {pres_str} (~{altitude_m/1000:.0f} km)...", flush=True)

        u_range = find_variable_range(entries, 'UGRD', pressure_hpa)
        v_range = find_variable_range(entries, 'VGRD', pressure_hpa)

        # Download U component
        u_bytes = download_byte_range(gfs_url, u_range[0], u_range[1])
        if u_bytes is None:
            print(f"    Failed to download UGRD", flush=True)
            continue

        # Download V component
        v_bytes = download_byte_range(gfs_url, v_range[0], v_range[1])
        if v_bytes is None:
            print(f"    Failed to download VGRD", flush=True)
            continue

        # Extract U
        u_result = extract_grib_data(u_bytes, EUROPE_BOUNDS)
        if u_result is None:
            continue
        u_data, _, _ = u_result

        # Extract V
        v_result = extract_grib_data(v_bytes, EUROPE_BOUNDS)
        if v_result is None:
            continue
        v_data, _, _ = v_result

        # Store in output arrays (row-major with levels as innermost dimension)
        for row in range(rows):
            for col in range(cols):
                idx = row * cols * num_levels + col * num_levels + level_idx
                wind_u[idx] = u_data[row, col]
                wind_v[idx] = v_data[row, col]

        # Log wind stats
        speed = np.sqrt(u_data**2 + v_data**2)
        print(f"    Wind speed: {speed.min():.1f} - {speed.max():.1f} m/s", flush=True)

    # Compute bounds from actual extracted data
    lat_south = min(lats[0], lats[-1])
    lat_north = max(lats[0], lats[-1])
    lon_west = min(lons[0], lons[-1])
    lon_east = max(lons[0], lons[-1])

    # Compute LAEA bounds for direct scene coordinate lookup
    laea_bounds = compute_laea_bounds(lons, lats)
    print(f"  LAEA bounds: X={laea_bounds['minX']:.0f} to {laea_bounds['maxX']:.0f}, Y={laea_bounds['minY']:.0f} to {laea_bounds['maxY']:.0f}", flush=True)

    return {
        "forecast_time": target_time,
        "cycle_time": cycle_time,
        "forecast_hour": forecast_hour,
        "bounds": {
            "west": float(lon_west),
            "east": float(lon_east),
            "south": float(lat_south),
            "north": float(lat_north)
        },
        "bounds_laea": laea_bounds,
        "cols": cols,
        "rows": rows,
        "levels_m": altitude_levels,
        "wind_u": wind_u.tolist(),
        "wind_v": wind_v.tolist()
    }


def main():
    output_json = Path("wind.json")
    output_bin = Path("wind.bin")

    print("=" * 60)
    print("NOAA GFS Full-Atmosphere Wind Fetcher")
    print("=" * 60)
    print()

    print(f"Configuration:")
    print(f"  Data source: NOAA GFS 0.25° (real-time forecast)")
    print(f"  Coverage: Europe ({EUROPE_BOUNDS['west']}°W to {EUROPE_BOUNDS['east']}°E, {EUROPE_BOUNDS['south']}°N to {EUROPE_BOUNDS['north']}°N)")
    print(f"  Pressure levels: {len(ALL_PRESSURE_LEVELS)} (1000 hPa to 0.01 hPa)")
    print(f"  Altitude range: surface to ~80 km")
    print()

    try:
        result = fetch_gfs_wind()

        if not result:
            print("\nERROR: Failed to fetch GFS data", flush=True)
            sys.exit(1)

        # Extract data
        forecast_time = result.pop("forecast_time")
        cycle_time = result.pop("cycle_time")
        forecast_hour = result.pop("forecast_hour")
        timestamp = forecast_time.isoformat()

        wind_u = np.array(result.pop("wind_u"), dtype=np.float32)
        wind_v = np.array(result.pop("wind_v"), dtype=np.float32)

        # Convert to Int16
        wind_u_i16 = np.clip(wind_u * WIND_SCALE, -32767, 32767).astype(np.int16)
        wind_v_i16 = np.clip(wind_v * WIND_SCALE, -32767, 32767).astype(np.int16)

        # Write binary
        bin_data = wind_u_i16.tobytes() + wind_v_i16.tobytes()
        with open(output_bin, 'wb') as f:
            f.write(bin_data)

        bin_hash = hashlib.sha256(bin_data).hexdigest()[:16]

        # Write metadata
        meta = {
            "timestamp": timestamp,
            "model_run": cycle_time.isoformat(),
            "forecast_hour": forecast_hour,
            "source": "NOAA GFS 0.25° (real-time)",
            "bin_hash": bin_hash,
            "altitude_grid": {
                **result,
                "data_file": "wind.bin",
                "data_length": len(wind_u),
                "data_type": "int16",
                "wind_scale": WIND_SCALE
            }
        }

        with open(output_json, 'w') as f:
            json.dump(meta, f)

        print()
        print("=" * 60)
        print(f"Output written:")
        print(f"  {output_json}: {output_json.stat().st_size / 1024:.1f} KB")
        print(f"  {output_bin}: {output_bin.stat().st_size / 1024:.1f} KB")
        print(f"  Model run: {cycle_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Forecast hour: f{forecast_hour:03d}")
        print(f"  Valid time: {forecast_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Levels: {len(result['levels_m'])}")
        print("=" * 60)

    except Exception as e:
        print(f"\nFatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
