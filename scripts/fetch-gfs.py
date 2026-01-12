#!/usr/bin/env python3
# Copyright (c) 2026 Mete Balci. All Rights Reserved.
"""
Fetch stratospheric wind data from NOAA GFS model (22-48km).

Data source: NOAA Global Forecast System (GFS) 0.25° resolution
Available on AWS Open Data: https://registry.opendata.aws/noaa-gfs-bdp-pds/

Pressure levels: 10, 7, 5, 3, 2, 1 hPa (approximately 31-48 km altitude)

Output: gfs-wind.json + gfs-wind.bin with stratospheric wind grid
"""

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
import requests
import numpy as np

# GFS data on AWS S3
GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# Switzerland bounds (WGS84) - same as ICON
SWISS_BOUNDS = {
    "west": 5.9,
    "east": 10.5,
    "south": 45.8,
    "north": 47.8
}

# Grid resolution - match ICON output
GRID_RESOLUTION_KM = 5

# Scale factor for Int16 encoding
WIND_SCALE = 100  # 0.01 m/s precision

# GFS pressure levels for stratosphere (hPa) with approximate altitudes (m)
# Standard atmosphere: h = -8500 * ln(p/1013.25)
GFS_PRESSURE_LEVELS = [
    (10, 31000),   # 10 hPa ≈ 31 km
    (7, 34000),    # 7 hPa ≈ 34 km
    (5, 37000),    # 5 hPa ≈ 37 km
    (3, 40000),    # 3 hPa ≈ 40 km
    (2, 43000),    # 2 hPa ≈ 43 km
    (1, 48000),    # 1 hPa ≈ 48 km
]


def compute_grid_dimensions() -> Tuple[int, int]:
    """Compute grid dimensions from bounds and resolution."""
    width_km = (SWISS_BOUNDS["east"] - SWISS_BOUNDS["west"]) * 111 * math.cos(math.radians(46.8))
    height_km = (SWISS_BOUNDS["north"] - SWISS_BOUNDS["south"]) * 111

    cols = int(width_km / GRID_RESOLUTION_KM)
    rows = int(height_km / GRID_RESOLUTION_KM)

    return cols, rows


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


def get_gfs_file_url(date_str: str, cycle: str, forecast_hour: int = 0) -> str:
    """Construct GFS file URL on AWS S3.

    Args:
        date_str: Date in YYYYMMDD format
        cycle: Cycle hour (00, 06, 12, 18)
        forecast_hour: Forecast hour (0 for analysis)

    Returns:
        URL to the GFS GRIB2 file
    """
    # GFS 0.25 degree files
    # Pattern: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
    return f"{GFS_BASE_URL}/gfs.{date_str}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"


def get_gfs_index_url(gfs_url: str) -> str:
    """Get the .idx index file URL for a GFS file."""
    return gfs_url + ".idx"


def parse_gfs_index(idx_content: str) -> List[dict]:
    """Parse GFS .idx file to get byte ranges for specific variables.

    Index format:
    1:0:d=2024010100:UGRD:1 mb:anl:
    2:123456:d=2024010100:VGRD:1 mb:anl:

    Returns:
        List of dicts with 'num', 'start', 'var', 'level', 'end' keys
    """
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
                'end': None  # Will be filled in
            }
            entries.append(entry)

    # Fill in end byte positions
    for i in range(len(entries) - 1):
        entries[i]['end'] = entries[i + 1]['start'] - 1

    # Last entry - estimate based on typical message size
    if entries:
        entries[-1]['end'] = entries[-1]['start'] + 500000

    return entries


def find_variable_range(entries: List[dict], var_name: str, level_mb: int) -> Optional[Tuple[int, int]]:
    """Find byte range for a specific variable and level.

    Args:
        entries: Parsed index entries
        var_name: Variable name (UGRD, VGRD)
        level_mb: Pressure level in mb (same as hPa)

    Returns:
        Tuple of (start_byte, end_byte) or None if not found
    """
    level_str = f"{level_mb} mb"

    for entry in entries:
        if entry['var'] == var_name and entry['level'] == level_str:
            return entry['start'], entry['end']

    return None


def download_byte_range(url: str, start: int, end: int) -> Optional[bytes]:
    """Download a specific byte range from a URL.

    Args:
        url: URL to download from
        start: Start byte
        end: End byte

    Returns:
        Bytes data or None if failed
    """
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
    """Extract data from GRIB2 bytes for the Switzerland region.

    Args:
        grib_bytes: Raw GRIB2 message bytes
        bounds: Geographic bounds dict with west, east, south, north

    Returns:
        Tuple of (data, lats, lons) arrays for the bounded region, or None if failed
    """
    import eccodes
    import tempfile

    # Write bytes to temp file (eccodes needs file handle)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as f:
        f.write(grib_bytes)
        temp_path = f.name

    try:
        with open(temp_path, 'rb') as f:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                return None

            try:
                # Get grid dimensions
                ni = eccodes.codes_get(gid, 'Ni')
                nj = eccodes.codes_get(gid, 'Nj')

                # Get grid parameters
                lat_first = eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
                lat_last = eccodes.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
                lon_first = eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
                lon_last = eccodes.codes_get(gid, 'longitudeOfLastGridPointInDegrees')

                # Get values
                values = eccodes.codes_get_values(gid)
                data = values.reshape((nj, ni))

                # Create coordinate arrays
                lats = np.linspace(lat_first, lat_last, nj)
                lons = np.linspace(lon_first, lon_last, ni)

                # Handle longitude wrapping (GFS uses 0-360)
                if lon_first > 180:
                    lon_first -= 360
                if lon_last > 180:
                    lon_last -= 360
                lons = np.linspace(lon_first, lon_last, ni)

                # For 0-360 format, we need to handle Switzerland (~6-10°E)
                # GFS 0.25° typically spans 0-360°
                if lons[0] >= 0 and lons[-1] > 180:
                    # Convert to -180 to 180
                    lons = np.where(lons > 180, lons - 360, lons)
                    # Sort and reorder data accordingly
                    sort_idx = np.argsort(lons)
                    lons = lons[sort_idx]
                    data = data[:, sort_idx]

                # Extract Switzerland region
                lat_mask = (lats >= bounds['south'] - 0.5) & (lats <= bounds['north'] + 0.5)
                lon_mask = (lons >= bounds['west'] - 0.5) & (lons <= bounds['east'] + 0.5)

                lat_indices = np.where(lat_mask)[0]
                lon_indices = np.where(lon_mask)[0]

                if len(lat_indices) == 0 or len(lon_indices) == 0:
                    print(f"  Warning: No data in Switzerland bounds", flush=True)
                    print(f"  Lat range: {lats.min():.1f} to {lats.max():.1f}", flush=True)
                    print(f"  Lon range: {lons.min():.1f} to {lons.max():.1f}", flush=True)
                    return None

                # Extract bounded region
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


def interpolate_to_grid(
    data: np.ndarray,
    data_lats: np.ndarray,
    data_lons: np.ndarray,
    target_cols: int,
    target_rows: int,
    bounds: dict
) -> np.ndarray:
    """Interpolate GFS data to our target grid.

    Args:
        data: 2D array of values (lat x lon)
        data_lats: 1D array of latitudes
        data_lons: 1D array of longitudes
        target_cols: Number of columns in target grid
        target_rows: Number of rows in target grid
        bounds: Geographic bounds

    Returns:
        Interpolated data array (target_rows x target_cols)
    """
    from scipy.interpolate import RegularGridInterpolator

    # GFS data might have lat in descending order
    if data_lats[0] > data_lats[-1]:
        data_lats = data_lats[::-1]
        data = data[::-1, :]

    # Create interpolator
    interp = RegularGridInterpolator(
        (data_lats, data_lons),
        data,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Create target grid
    target_lats = np.linspace(bounds['south'], bounds['north'], target_rows)
    target_lons = np.linspace(bounds['west'], bounds['east'], target_cols)
    target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)

    # Interpolate
    points = np.column_stack([target_lat_grid.ravel(), target_lon_grid.ravel()])
    result = interp(points).reshape(target_rows, target_cols)

    return result


def fetch_gfs_stratospheric_wind() -> Optional[dict]:
    """Fetch stratospheric wind data from GFS.

    Returns:
        Dict with wind data and metadata, or None if failed
    """
    print("Fetching NOAA GFS stratospheric wind data...", flush=True)

    cols, rows = compute_grid_dimensions()
    print(f"  Target grid: {cols}x{rows} ({GRID_RESOLUTION_KM}km resolution)", flush=True)
    print(f"  Pressure levels: {[p for p, _ in GFS_PRESSURE_LEVELS]} hPa", flush=True)

    # Get latest GFS cycle
    date_str, cycle, cycle_time = get_latest_gfs_cycle()
    print(f"  GFS cycle: {date_str} {cycle}Z", flush=True)

    # Get file URL and index
    gfs_url = get_gfs_file_url(date_str, cycle, forecast_hour=0)
    idx_url = get_gfs_index_url(gfs_url)

    print(f"  Fetching index: {idx_url}", flush=True)

    try:
        idx_response = requests.get(idx_url, timeout=30)
        idx_response.raise_for_status()
        idx_content = idx_response.text
    except Exception as e:
        print(f"  Failed to fetch index: {e}", flush=True)
        return None

    entries = parse_gfs_index(idx_content)
    print(f"  Index has {len(entries)} entries", flush=True)

    # Build 3D wind arrays
    num_levels = len(GFS_PRESSURE_LEVELS)
    total_cells = rows * cols * num_levels

    wind_u = np.zeros(total_cells, dtype=np.float32)
    wind_v = np.zeros(total_cells, dtype=np.float32)
    altitude_levels = []

    for level_idx, (pressure_mb, altitude_m) in enumerate(GFS_PRESSURE_LEVELS):
        altitude_levels.append(altitude_m)

        print(f"  Fetching {pressure_mb} hPa (~{altitude_m/1000:.0f} km)...", flush=True)

        # Find byte ranges for U and V
        u_range = find_variable_range(entries, 'UGRD', pressure_mb)
        v_range = find_variable_range(entries, 'VGRD', pressure_mb)

        if u_range is None or v_range is None:
            print(f"    Warning: Could not find UGRD/VGRD at {pressure_mb} mb", flush=True)
            continue

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

        # Extract and interpolate U
        u_result = extract_grib_data(u_bytes, SWISS_BOUNDS)
        if u_result is None:
            continue
        u_data, u_lats, u_lons = u_result
        u_grid = interpolate_to_grid(u_data, u_lats, u_lons, cols, rows, SWISS_BOUNDS)

        # Extract and interpolate V
        v_result = extract_grib_data(v_bytes, SWISS_BOUNDS)
        if v_result is None:
            continue
        v_data, v_lats, v_lons = v_result
        v_grid = interpolate_to_grid(v_data, v_lats, v_lons, cols, rows, SWISS_BOUNDS)

        # Store in output arrays
        for row in range(rows):
            for col in range(cols):
                idx = row * cols * num_levels + col * num_levels + level_idx
                wind_u[idx] = u_grid[row, col]
                wind_v[idx] = v_grid[row, col]

        # Log wind stats for this level
        speed = np.sqrt(u_grid**2 + v_grid**2)
        print(f"    Wind speed: {speed.min():.1f} - {speed.max():.1f} m/s", flush=True)

    # Overall stats
    speeds = np.sqrt(wind_u**2 + wind_v**2)
    print(f"  Total wind speed range: {speeds.min():.1f} - {speeds.max():.1f} m/s", flush=True)

    return {
        "cycle_time": cycle_time,
        "bounds": SWISS_BOUNDS,
        "cols": cols,
        "rows": rows,
        "levels_m": altitude_levels,
        "wind_u": wind_u.tolist(),
        "wind_v": wind_v.tolist()
    }


def main():
    # Output files
    output_json = Path("gfs-wind.json")
    output_bin = Path("gfs-wind.bin")

    print("=" * 60)
    print("NOAA GFS Stratospheric Wind Fetcher")
    print("=" * 60)
    print()

    cols, rows = compute_grid_dimensions()
    print(f"Configuration:")
    print(f"  Grid resolution: {GRID_RESOLUTION_KM} km")
    print(f"  Grid size: {cols} x {rows}")
    print(f"  Pressure levels: {len(GFS_PRESSURE_LEVELS)}")
    print(f"  Altitude range: {GFS_PRESSURE_LEVELS[0][1]/1000:.0f} - {GFS_PRESSURE_LEVELS[-1][1]/1000:.0f} km")
    print()

    try:
        result = fetch_gfs_stratospheric_wind()

        if not result:
            print("\nERROR: Failed to fetch GFS data", flush=True)
            sys.exit(1)

        # Extract data
        cycle_time = result.pop("cycle_time")
        timestamp = cycle_time.isoformat()

        wind_u = np.array(result.pop("wind_u"), dtype=np.float32)
        wind_v = np.array(result.pop("wind_v"), dtype=np.float32)

        # Convert to Int16 with scale factor
        wind_u_i16 = np.clip(wind_u * WIND_SCALE, -32767, 32767).astype(np.int16)
        wind_v_i16 = np.clip(wind_v * WIND_SCALE, -32767, 32767).astype(np.int16)

        # Write binary file
        bin_data = wind_u_i16.tobytes() + wind_v_i16.tobytes()
        with open(output_bin, 'wb') as f:
            f.write(bin_data)

        # Compute hash
        bin_hash = hashlib.sha256(bin_data).hexdigest()[:16]

        # Write metadata JSON
        meta = {
            "timestamp": timestamp,
            "source": "NOAA GFS 0.25°",
            "grid_resolution_km": GRID_RESOLUTION_KM,
            "bin_hash": bin_hash,
            "altitude_grid": {
                **result,
                "data_file": "gfs-wind.bin",
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
        print(f"  Timestamp: {timestamp}")
        print("=" * 60)

    except Exception as e:
        print(f"\nFatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
