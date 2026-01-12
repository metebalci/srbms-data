#!/usr/bin/env python3
# Copyright (c) 2026 Mete Balci. All Rights Reserved.
"""
Fetch real-time 3D wind data from MeteoSwiss ICON-CH1-EPS model.

Data source: ICON-CH1-EPS numerical weather prediction model (~1km resolution)
Output: icon-wind.json + icon-wind.bin with 3D wind grid for altitudes 0-22km

No longer uses SMN weather stations - ICON provides complete 3D wind field.
"""

import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple
import requests
import numpy as np

# ICON-CH1-EPS API
STAC_API_URL = "https://data.geo.admin.ch/api/stac/v1/search"
STAC_ASSETS_URL = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-forecasting-icon-ch1/assets"
ICON_COLLECTION = "ch.meteoschweiz.ogd-forecasting-icon-ch1"

# Cache directory for grid coordinates (they don't change)
CACHE_DIR = Path(tempfile.gettempdir()) / "icon_cache"

# Switzerland bounds (WGS84)
SWISS_BOUNDS = {
    "west": 5.9,
    "east": 10.5,
    "south": 45.8,
    "north": 47.8
}

# Grid resolution in km (aligned to ICON 1km grid)
GRID_RESOLUTION_KM = 5  # 5km gives good detail while keeping files small

# Scale factors for Int16 encoding (divide by these to get actual values)
WIND_SCALE = 100      # 0.01 m/s precision, ±327 m/s range
SPEED_STD_SCALE = 100 # 0.01 m/s precision
DIR_SPREAD_SCALE = 10 # 0.1 degree precision, ±3276° range

# Altitude levels (meters) - surface 10m + 1km spacing above 2000m
ALTITUDE_LEVELS = [
    10,                                  # Surface (10m above ground)
    500, 1000, 1500, 2000,              # Lower troposphere
    3000, 4000, 5000, 6000, 7000,       # Mid troposphere
    8000, 9000, 10000, 11000, 12000,    # Upper troposphere / jet stream
    13000, 14000, 15000, 16000, 17000,  # Tropopause region
    18000, 19000, 20000, 21000, 22000   # Lower stratosphere
]

# ICON model level to approximate altitude mapping (meters)
# ICON has 80 full levels, level 1 is top of atmosphere, level 80 is surface
# Heights from vertical_constants file (mean across Switzerland)
ICON_LEVEL_ALTITUDES = {
    80: 484,     # Surface (varies 5-4460m with terrain)
    79: 514,
    78: 553,
    77: 596,
    76: 645,
    75: 698,
    74: 754,
    73: 815,
    72: 879,
    71: 946,
    70: 1017,
    69: 1091,
    68: 1168,
    67: 1248,
    66: 1331,
    65: 1418,
    64: 1507,
    63: 1600,
    62: 1696,
    61: 1795,
    60: 1897,
    59: 2002,
    58: 2111,
    57: 2223,
    56: 2338,
    55: 2456,
    54: 2579,
    53: 2704,
    52: 2833,
    51: 2966,
    50: 3103,
    49: 3244,
    48: 3388,
    47: 3537,
    46: 3689,
    45: 3846,
    44: 4007,
    43: 4173,
    42: 4343,
    41: 4518,
    40: 4698,
    39: 4883,
    38: 5073,
    37: 5268,
    36: 5469,
    35: 5675,
    34: 5887,
    33: 6106,
    32: 6330,
    31: 6562,
    30: 6800,
    29: 7045,
    28: 7297,
    27: 7558,
    26: 7826,
    25: 8103,
    24: 8388,
    23: 8683,
    22: 8988,
    21: 9304,
    20: 9630,
    19: 9969,
    18: 10320,
    17: 10684,
    16: 11064,
    15: 11459,
    14: 11872,
    13: 12304,
    12: 12757,
    11: 13234,
    10: 13738,
    9: 14273,
    8: 14843,
    7: 15456,
    6: 16119,
    5: 16851,
    4: 17671,
    3: 18624,
    2: 19808,
    1: 22000,
}

# Map our target altitudes to closest ICON model levels
def get_icon_levels_for_altitudes(target_altitudes: list) -> dict:
    """Map target altitudes to closest ICON model levels."""
    result = {}
    level_alts = list(ICON_LEVEL_ALTITUDES.items())

    for target in target_altitudes:
        # Find closest ICON level
        best_level = min(level_alts, key=lambda x: abs(x[1] - target))
        result[target] = best_level[0]

    return result


def compute_grid_dimensions() -> Tuple[int, int]:
    """Compute grid dimensions from bounds and resolution."""
    width_km = (SWISS_BOUNDS["east"] - SWISS_BOUNDS["west"]) * 111 * math.cos(math.radians(46.8))  # ~78km per degree at Swiss latitude
    height_km = (SWISS_BOUNDS["north"] - SWISS_BOUNDS["south"]) * 111  # ~111km per degree latitude

    cols = int(width_km / GRID_RESOLUTION_KM)
    rows = int(height_km / GRID_RESOLUTION_KM)

    return cols, rows


def get_horizontal_constants_url() -> Optional[str]:
    """Query STAC API to get the horizontal_constants file URL."""
    try:
        response = requests.get(STAC_ASSETS_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Assets is a list of asset objects
        assets = data.get("assets", [])
        if isinstance(assets, list):
            for asset in assets:
                asset_id = asset.get("id", "")
                if "horizontal_constants" in asset_id.lower():
                    return asset.get("href")
        else:
            # Fallback for dict format
            for asset_id, asset_info in assets.items():
                if "horizontal_constants" in asset_id.lower():
                    return asset_info.get("href")

        return None
    except Exception as e:
        print(f"  Error fetching horizontal_constants URL: {e}", flush=True)
        return None


def load_icon_grid_coordinates() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load ICON grid coordinates (TLON, TLAT) from horizontal_constants file.

    Caches the coordinates as numpy arrays to avoid re-downloading.
    Returns (lons, lats) arrays or None if unavailable.
    """
    import eccodes

    CACHE_DIR.mkdir(exist_ok=True)
    lons_cache = CACHE_DIR / "icon_lons.npy"
    lats_cache = CACHE_DIR / "icon_lats.npy"

    # Check cache first
    if lons_cache.exists() and lats_cache.exists():
        try:
            lons = np.load(lons_cache)
            lats = np.load(lats_cache)
            print(f"  Loaded cached grid coordinates: {len(lons)} points", flush=True)
            return lons, lats
        except Exception as e:
            print(f"  Cache load failed: {e}, re-downloading...", flush=True)

    # Get download URL from STAC API
    print("  Fetching horizontal_constants URL from STAC API...", flush=True)
    url = get_horizontal_constants_url()
    if not url:
        print("  Could not find horizontal_constants file URL", flush=True)
        return None

    # Download the file
    grib_path = CACHE_DIR / "horizontal_constants.grib2"
    print(f"  Downloading horizontal_constants.grib2...", flush=True)

    try:
        response = requests.get(url, timeout=300, stream=True)  # Large file, longer timeout
        response.raise_for_status()

        with open(grib_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Downloaded {grib_path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)
    except Exception as e:
        print(f"  Download failed: {e}", flush=True)
        return None

    # Extract TLON and TLAT from GRIB file
    lons = None
    lats = None

    try:
        with open(grib_path, 'rb') as f:
            while True:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break

                short_name = eccodes.codes_get(gid, 'shortName')

                if short_name == 'tlon':
                    lons = eccodes.codes_get_array(gid, 'values')
                    print(f"  Extracted TLON: {len(lons)} points, range {lons.min():.3f} to {lons.max():.3f}°", flush=True)
                elif short_name == 'tlat':
                    lats = eccodes.codes_get_array(gid, 'values')
                    print(f"  Extracted TLAT: {len(lats)} points, range {lats.min():.3f} to {lats.max():.3f}°", flush=True)

                eccodes.codes_release(gid)

                if lons is not None and lats is not None:
                    break

        if lons is None or lats is None:
            print("  Could not find TLON/TLAT in horizontal_constants file", flush=True)
            return None

        # Cache the coordinates
        np.save(lons_cache, lons)
        np.save(lats_cache, lats)
        print(f"  Cached coordinates to {CACHE_DIR}", flush=True)

        return lons, lats

    except Exception as e:
        print(f"  Error extracting coordinates: {e}", flush=True)
        return None


def circular_std(angles_deg: np.ndarray) -> float:
    """Compute circular standard deviation of angles in degrees."""
    if len(angles_deg) == 0:
        return 0.0

    angles_rad = np.radians(angles_deg)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))

    # R is the mean resultant length (0 to 1)
    R = np.sqrt(mean_sin**2 + mean_cos**2)

    # Circular standard deviation
    if R >= 1.0:
        return 0.0
    return np.degrees(np.sqrt(-2 * np.log(R)))


def find_closest_icon_levels(target_altitudes: list) -> dict:
    """Find ICON model levels closest to our target altitudes."""
    level_to_altitude = ICON_LEVEL_ALTITUDES
    altitude_to_level = {}

    for target in target_altitudes:
        closest_level = min(level_to_altitude.keys(),
                          key=lambda l: abs(level_to_altitude[l] - target))
        altitude_to_level[target] = closest_level

    return altitude_to_level


def get_candidate_run_times(max_runs: int = 4) -> list:
    """Get list of candidate ICON model run times to try.

    Returns most recent runs in order (newest first).
    ICON-CH1 runs every 3 hours: 00, 03, 06, 09, 12, 15, 18, 21 UTC
    If a run's data isn't available yet, the query will fail and we try older runs.
    """
    now = datetime.now(timezone.utc)
    hour = (now.hour // 3) * 3
    current_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)

    candidates = []
    for i in range(max_runs):
        run_time = current_run - timedelta(hours=3 * i)
        candidates.append(run_time)

    return candidates


def compute_forecast_horizon(model_run_time: datetime, target_time: datetime) -> str:
    """Compute ISO 8601 duration string for forecast horizon.

    Args:
        model_run_time: When the model run started
        target_time: The time we want the forecast for

    Returns:
        ISO 8601 duration string (e.g., "P0DT3H0M0S" for 3 hours)
    """
    delta = target_time - model_run_time
    total_seconds = int(delta.total_seconds())

    if total_seconds < 0:
        return "P0DT0H0M0S"  # Target is before model run, use analysis

    hours = total_seconds // 3600
    return f"P0DT{hours}H0M0S"


def query_icon_stac(
    variable: str,
    reference_datetime: Optional[datetime] = None,
    target_time: Optional[datetime] = None
) -> Optional[Tuple[str, datetime, datetime]]:
    """Query STAC API for ICON-CH1-EPS data and return download URL.

    Args:
        variable: Wind variable name (e.g., 'U_10M', 'V_10M', 'U', 'V')
        reference_datetime: Forecast reference time (model run), or None for latest
        target_time: Time we want the forecast for (default: now)

    Returns:
        Tuple of (download URL, model run datetime, forecast valid time) or None if not found
    """
    # Default target time is now
    if target_time is None:
        target_time = datetime.now(timezone.utc)

    # If no reference datetime, try multiple recent runs
    if reference_datetime is None:
        candidates = get_candidate_run_times()
        for candidate in candidates:
            result = query_icon_stac(variable, candidate, target_time)
            if result:
                return result
        return None

    model_run_time = reference_datetime
    reference_datetime_str = model_run_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Compute horizon from model run to target time
    horizon = compute_forecast_horizon(model_run_time, target_time)

    query = {
        "collections": [ICON_COLLECTION],
        "forecast:reference_datetime": reference_datetime_str,
        "forecast:variable": variable,
        "forecast:perturbed": False,  # Deterministic forecast, not ensemble
        "forecast:horizon": horizon,
        "limit": 1
    }

    try:
        response = requests.post(STAC_API_URL, json=query, timeout=30)
        response.raise_for_status()

        data = response.json()
        features = data.get("features", [])

        if not features:
            print(f"  No data found for {variable} at {reference_datetime_str} + {horizon}", flush=True)
            return None

        # Compute the actual forecast valid time (model run + horizon hours)
        horizon_hours = int(horizon.split("T")[1].split("H")[0])
        forecast_valid_time = model_run_time + timedelta(hours=horizon_hours)

        # Get the asset URL
        assets = features[0].get("assets", {})
        if "data" in assets:
            return (assets["data"].get("href"), model_run_time, forecast_valid_time)

        # Try first available asset
        for asset in assets.values():
            if "href" in asset:
                return (asset["href"], model_run_time, forecast_valid_time)

        return None

    except Exception as e:
        print(f"  STAC query error for {variable}: {e}", flush=True)
        return None


def download_grib(url: str, output_path: Path) -> bool:
    """Download a GRIB2 file from URL."""
    try:
        print(f"  Downloading: {url}", flush=True)
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"  Download error: {e}", flush=True)
        return False


def parse_grib_with_cfgrib(grib_path: Path) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Parse GRIB2 file using cfgrib and return data array with coordinates.

    Returns:
        Tuple of (data, lats, lons) or (data, None, None) if coordinates unavailable
    """
    try:
        import xarray as xr

        ds = xr.open_dataset(grib_path, engine='cfgrib')

        # Get the data variable (first non-coordinate variable)
        data_vars = [v for v in ds.data_vars]
        if not data_vars:
            print(f"  No data variables in GRIB file", flush=True)
            return None

        data = ds[data_vars[0]].values

        # Get coordinates - ICON uses unstructured grid so these may be 1D
        lats = None
        lons = None
        if 'latitude' in ds.coords:
            lats = ds.coords['latitude'].values
        if 'longitude' in ds.coords:
            lons = ds.coords['longitude'].values

        if lats is not None and lons is not None:
            print(f"  GRIB data shape: {data.shape}, lat range: {lats.min():.2f}-{lats.max():.2f}, lon range: {lons.min():.2f}-{lons.max():.2f}", flush=True)
        else:
            print(f"  GRIB data shape: {data.shape} (unstructured grid, need to load coords separately)", flush=True)

        ds.close()
        return data, lats, lons

    except ImportError:
        print("  cfgrib not installed, trying eccodes directly", flush=True)
        result = parse_grib_with_eccodes(grib_path)
        if result is not None:
            return result, None, None
        return None
    except Exception as e:
        print(f"  cfgrib parse error: {e}", flush=True)
        return None


def parse_grib_with_eccodes(grib_path: Path) -> Optional[np.ndarray]:
    """Parse GRIB2 file using eccodes directly."""
    try:
        import eccodes

        with open(grib_path, 'rb') as f:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                return None

            try:
                ni = eccodes.codes_get(gid, 'Ni')
                nj = eccodes.codes_get(gid, 'Nj')
                values = eccodes.codes_get_values(gid)
                data = values.reshape((nj, ni))

                print(f"  GRIB data shape: {data.shape}", flush=True)
                return data

            finally:
                eccodes.codes_release(gid)

    except ImportError:
        print("  eccodes not installed", flush=True)
        return None
    except Exception as e:
        print(f"  eccodes parse error: {e}", flush=True)
        return None


def downsample_to_grid(
    data: np.ndarray,
    data_lats: np.ndarray,
    data_lons: np.ndarray,
    target_cols: int,
    target_rows: int,
    bounds: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample high-resolution data to our target grid.

    Returns:
        Tuple of (mean_values, std_values, direction_spread) arrays
    """
    # Create target grid coordinates
    target_lons = np.linspace(bounds["west"], bounds["east"], target_cols)
    target_lats = np.linspace(bounds["south"], bounds["north"], target_rows)

    mean_values = np.zeros((target_rows, target_cols))
    std_values = np.zeros((target_rows, target_cols))

    # Cell size in degrees
    lon_step = (bounds["east"] - bounds["west"]) / target_cols
    lat_step = (bounds["north"] - bounds["south"]) / target_rows

    for i, target_lat in enumerate(target_lats):
        for j, target_lon in enumerate(target_lons):
            # Find all source cells within this target cell
            lat_mask = (data_lats >= target_lat) & (data_lats < target_lat + lat_step)
            lon_mask = (data_lons >= target_lon) & (data_lons < target_lon + lon_step)

            # For 2D lat/lon arrays
            if data_lats.ndim == 2:
                cell_mask = lat_mask & lon_mask
                cell_values = data[cell_mask]
            else:
                # For 1D coordinate arrays
                lat_indices = np.where(lat_mask)[0]
                lon_indices = np.where(lon_mask)[0]
                if len(lat_indices) > 0 and len(lon_indices) > 0:
                    cell_values = data[np.ix_(lat_indices, lon_indices)].flatten()
                else:
                    cell_values = np.array([])

            if len(cell_values) > 0:
                mean_values[i, j] = np.mean(cell_values)
                std_values[i, j] = np.std(cell_values)
            else:
                # Nearest neighbor fallback
                if data_lats.ndim == 1:
                    lat_idx = np.argmin(np.abs(data_lats - target_lat))
                    lon_idx = np.argmin(np.abs(data_lons - target_lon))
                    mean_values[i, j] = data[lat_idx, lon_idx]
                else:
                    # Find nearest point
                    dist = (data_lats - target_lat)**2 + (data_lons - target_lon)**2
                    nearest_idx = np.unravel_index(np.argmin(dist), dist.shape)
                    mean_values[i, j] = data[nearest_idx]
                std_values[i, j] = 0

    return mean_values, std_values


def interpolate_unstructured_to_grid(
    values: np.ndarray,
    source_lats: np.ndarray,
    source_lons: np.ndarray,
    target_cols: int,
    target_rows: int,
    bounds: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate from unstructured ICON grid to regular lat/lon grid.

    Uses nearest neighbor interpolation for simplicity and speed.
    Returns (interpolated_values, std_values).
    """
    from scipy.spatial import cKDTree

    # Create target grid
    target_lons = np.linspace(bounds["west"], bounds["east"], target_cols)
    target_lats = np.linspace(bounds["south"], bounds["north"], target_rows)
    target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)

    # Build KD-tree from source points
    source_points = np.column_stack([source_lons, source_lats])
    tree = cKDTree(source_points)

    # For each target point, find nearest source points
    target_points = np.column_stack([target_lon_grid.ravel(), target_lat_grid.ravel()])

    # Use k nearest neighbors for interpolation and std calculation
    k = 4  # Use 4 nearest neighbors
    distances, indices = tree.query(target_points, k=k)

    # Compute weighted average and std
    interpolated = np.zeros(len(target_points))
    std_values = np.zeros(len(target_points))

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        neighbor_values = values[idxs]

        # Inverse distance weighting
        if dists[0] < 0.001:  # Very close to a source point
            interpolated[i] = neighbor_values[0]
            std_values[i] = np.std(neighbor_values)
        else:
            weights = 1.0 / np.maximum(dists, 0.001)
            weights /= weights.sum()
            interpolated[i] = np.sum(neighbor_values * weights)
            std_values[i] = np.std(neighbor_values)

    return interpolated.reshape(target_rows, target_cols), std_values.reshape(target_rows, target_cols)


def parse_multilevel_grib(grib_path: Path, target_levels: list) -> Optional[dict]:
    """Parse multi-level GRIB2 file and extract data at specific levels.

    Args:
        grib_path: Path to GRIB2 file
        target_levels: List of ICON model levels to extract (1-80)

    Returns:
        Dict mapping level -> data array, or None if failed
    """
    import eccodes

    result = {}
    target_set = set(target_levels)

    try:
        with open(grib_path, 'rb') as f:
            while True:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break

                level = eccodes.codes_get(gid, 'level')
                if level in target_set:
                    values = eccodes.codes_get_array(gid, 'values')
                    result[level] = values

                eccodes.codes_release(gid)

        return result if result else None

    except Exception as e:
        print(f"  Error parsing multi-level GRIB: {e}", flush=True)
        return None


def fetch_icon_wind(target_time: Optional[datetime] = None) -> Optional[dict]:
    """Fetch 3D wind data from ICON-CH1-EPS model.

    Fetches forecast wind data for a specific target time.

    Args:
        target_time: The time we want the forecast for (default: now)

    Returns:
        altitude_grid dict with wind_u, wind_v, and quality metrics,
        or None if fetching fails
    """
    if target_time is None:
        target_time = datetime.now(timezone.utc)

    print(f"Fetching ICON-CH1-EPS wind data for {target_time.strftime('%Y-%m-%d %H:%M UTC')}...", flush=True)

    cols, rows = compute_grid_dimensions()
    print(f"  Target grid: {cols}x{rows} ({GRID_RESOLUTION_KM}km resolution)", flush=True)
    print(f"  Altitude levels: {len(ALTITUDE_LEVELS)}", flush=True)

    # First, load grid coordinates from horizontal_constants file
    print("Loading ICON grid coordinates...", flush=True)
    coords = load_icon_grid_coordinates()
    if coords is None:
        print("  Could not load ICON grid coordinates", flush=True)
        return None

    grid_lons, grid_lats = coords

    # Create Switzerland mask for filtering
    buffer = 0.5
    swiss_mask = (
        (grid_lats >= SWISS_BOUNDS["south"] - buffer) &
        (grid_lats <= SWISS_BOUNDS["north"] + buffer) &
        (grid_lons >= SWISS_BOUNDS["west"] - buffer) &
        (grid_lons <= SWISS_BOUNDS["east"] + buffer)
    )
    lats_filtered = grid_lats[swiss_mask]
    lons_filtered = grid_lons[swiss_mask]
    print(f"  Swiss region: {len(lats_filtered)} grid points", flush=True)

    # Map target altitudes to ICON levels
    altitude_to_level = get_icon_levels_for_altitudes(ALTITUDE_LEVELS)
    needed_levels = list(set(altitude_to_level.values()))
    print(f"  Need ICON levels: {sorted(needed_levels)}", flush=True)

    # Query for surface wind (U_10M, V_10M) and multi-level wind (U, V)
    u_10m_result = query_icon_stac("U_10M", target_time=target_time)
    v_10m_result = query_icon_stac("V_10M", target_time=target_time)
    u_ml_result = query_icon_stac("U", target_time=target_time)  # Multi-level
    v_ml_result = query_icon_stac("V", target_time=target_time)  # Multi-level

    if not u_10m_result or not v_10m_result:
        print("  Could not find ICON surface wind data", flush=True)
        return None

    if not u_ml_result or not v_ml_result:
        print("  Could not find ICON multi-level wind data", flush=True)
        return None

    # Extract URLs, model run time, and forecast valid time
    u_10m_url, model_run_time, forecast_valid_time = u_10m_result
    v_10m_url, _, _ = v_10m_result
    u_ml_url, _, _ = u_ml_result
    v_ml_url, _, _ = v_ml_result

    print(f"  Model run: {model_run_time.strftime('%Y-%m-%d %H:%M UTC')}", flush=True)
    print(f"  Forecast valid: {forecast_valid_time.strftime('%Y-%m-%d %H:%M UTC')}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Download all files
        u_10m_path = tmpdir / "u_10m.grib2"
        v_10m_path = tmpdir / "v_10m.grib2"
        u_ml_path = tmpdir / "u_ml.grib2"
        v_ml_path = tmpdir / "v_ml.grib2"

        print("  Downloading wind data...", flush=True)
        if not download_grib(u_10m_url, u_10m_path):
            return None
        if not download_grib(v_10m_url, v_10m_path):
            return None
        if not download_grib(u_ml_url, u_ml_path):
            return None
        if not download_grib(v_ml_url, v_ml_path):
            return None

        # Parse surface wind
        print("  Parsing surface wind...", flush=True)
        u_10m_result = parse_grib_with_cfgrib(u_10m_path)
        v_10m_result = parse_grib_with_cfgrib(v_10m_path)

        if u_10m_result is None or v_10m_result is None:
            print("  Failed to parse surface wind files", flush=True)
            return None

        u_10m_data = u_10m_result[0].ravel()
        v_10m_data = v_10m_result[0].ravel()

        # Parse multi-level wind
        print("  Parsing multi-level wind...", flush=True)
        u_ml_data = parse_multilevel_grib(u_ml_path, needed_levels)
        v_ml_data = parse_multilevel_grib(v_ml_path, needed_levels)

        if u_ml_data is None or v_ml_data is None:
            print("  Failed to parse multi-level wind files", flush=True)
            return None

        print(f"  Got data for {len(u_ml_data)} levels", flush=True)

        # Build 3D grid
        num_levels = len(ALTITUDE_LEVELS)
        total_cells = rows * cols * num_levels

        wind_u = np.zeros(total_cells, dtype=np.float32)
        wind_v = np.zeros(total_cells, dtype=np.float32)
        wind_speed_std = np.zeros(total_cells, dtype=np.float32)
        wind_dir_spread = np.zeros(total_cells, dtype=np.float32)

        print("  Interpolating each altitude level...", flush=True)

        for level_idx, altitude in enumerate(ALTITUDE_LEVELS):
            # Get data for this altitude
            if altitude == 10:
                # Use surface wind for 10m level
                u_raw = u_10m_data
                v_raw = v_10m_data
            else:
                # Use multi-level data
                icon_level = altitude_to_level[altitude]
                if icon_level not in u_ml_data or icon_level not in v_ml_data:
                    print(f"  Warning: Level {icon_level} not found, using nearest", flush=True)
                    # Find nearest available level
                    available = list(u_ml_data.keys())
                    icon_level = min(available, key=lambda l: abs(ICON_LEVEL_ALTITUDES.get(l, 0) - altitude))
                u_raw = u_ml_data[icon_level]
                v_raw = v_ml_data[icon_level]

            # Filter to Switzerland
            u_swiss = u_raw[swiss_mask]
            v_swiss = v_raw[swiss_mask]

            # Interpolate to regular grid
            try:
                u_grid, u_std = interpolate_unstructured_to_grid(
                    u_swiss, lats_filtered, lons_filtered, cols, rows, SWISS_BOUNDS
                )
                v_grid, v_std = interpolate_unstructured_to_grid(
                    v_swiss, lats_filtered, lons_filtered, cols, rows, SWISS_BOUNDS
                )
            except Exception as e:
                print(f"  Interpolation error at altitude {altitude}m: {e}", flush=True)
                return None

            # Compute quality metrics
            speed_std = np.sqrt(u_std**2 + v_std**2)

            # Compute direction spread from U/V std
            # Approximation: use average of component stds
            dir_spread = np.degrees(np.arctan2(np.sqrt(u_std**2 + v_std**2),
                                               np.sqrt(u_grid**2 + v_grid**2) + 0.01))

            # Store in output arrays
            for row in range(rows):
                for col in range(cols):
                    idx = row * cols * num_levels + col * num_levels + level_idx
                    wind_u[idx] = u_grid[row, col]
                    wind_v[idx] = v_grid[row, col]
                    wind_speed_std[idx] = speed_std[row, col]
                    wind_dir_spread[idx] = dir_spread[row, col]

            # Progress indicator
            if (level_idx + 1) % 5 == 0:
                print(f"    Processed {level_idx + 1}/{num_levels} levels", flush=True)

        # Log some stats
        speeds = np.sqrt(wind_u**2 + wind_v**2)
        print(f"  Wind speed range: {speeds.min():.1f} - {speeds.max():.1f} m/s", flush=True)

        return {
            "model_run_time": model_run_time,
            "forecast_valid_time": forecast_valid_time,
            "bounds": SWISS_BOUNDS,
            "cols": cols,
            "rows": rows,
            "levels_m": ALTITUDE_LEVELS,
            "wind_u": wind_u.tolist(),
            "wind_v": wind_v.tolist(),
            "wind_speed_std": wind_speed_std.tolist(),
            "wind_dir_spread": wind_dir_spread.tolist()
        }


def main():
    # Output files (binary format)
    weather_json = Path("icon-wind.json")
    weather_bin = Path("icon-wind.bin")
    quality_json = Path("icon-wind-quality.json")
    quality_bin = Path("icon-wind-quality.bin")

    print("=" * 60)
    print("ICON-CH1-EPS Wind Data Fetcher")
    print("=" * 60)
    print()

    cols, rows = compute_grid_dimensions()
    print(f"Configuration:")
    print(f"  Grid resolution: {GRID_RESOLUTION_KM} km")
    print(f"  Grid size: {cols} x {rows}")
    print(f"  Altitude levels: {len(ALTITUDE_LEVELS)}")
    print(f"  Total cells: {cols * rows * len(ALTITUDE_LEVELS):,}")
    print()

    try:
        # Calculate target time as the next full hour
        # e.g., if running at 07:05, fetch forecast for 08:00
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        print(f"Target forecast time: {next_hour.strftime('%Y-%m-%d %H:%M UTC')}")
        print()

        # Fetch ICON data for the next hour
        altitude_grid = fetch_icon_wind(target_time=next_hour)

        if not altitude_grid:
            print("\nERROR: Failed to fetch ICON data", flush=True)
            print("No output written - keeping previous data in R2")
            sys.exit(1)

        # Use the forecast valid time as the timestamp (the time the forecast is for)
        model_run_time = altitude_grid.pop("model_run_time")
        forecast_valid_time = altitude_grid.pop("forecast_valid_time")
        timestamp = forecast_valid_time.isoformat()
        model_run_timestamp = model_run_time.isoformat()
        print(f"  Forecast valid time (timestamp): {timestamp}", flush=True)
        print(f"  Model run time: {model_run_timestamp}", flush=True)

        # Extract arrays for binary output
        wind_u = np.array(altitude_grid.pop("wind_u"), dtype=np.float32)
        wind_v = np.array(altitude_grid.pop("wind_v"), dtype=np.float32)
        wind_speed_std = altitude_grid.pop("wind_speed_std", None)
        wind_dir_spread = altitude_grid.pop("wind_dir_spread", None)

        # Convert to Int16 with scale factors (50% size reduction)
        wind_u_i16 = np.clip(wind_u * WIND_SCALE, -32767, 32767).astype(np.int16)
        wind_v_i16 = np.clip(wind_v * WIND_SCALE, -32767, 32767).astype(np.int16)

        # Write icon-wind.bin (wind_u followed by wind_v as Int16)
        weather_bin_data = wind_u_i16.tobytes() + wind_v_i16.tobytes()
        with open(weather_bin, 'wb') as f:
            f.write(weather_bin_data)

        # Compute hash for verification (SHA-256, first 16 chars)
        weather_bin_hash = hashlib.sha256(weather_bin_data).hexdigest()[:16]

        # Write icon-wind.json (metadata only)
        weather_meta = {
            "timestamp": timestamp,  # Forecast valid time
            "model_run": model_run_timestamp,  # When the model was run
            "grid_resolution_km": GRID_RESOLUTION_KM,
            "bin_hash": weather_bin_hash,  # Hash of icon-wind.bin for verification
            "altitude_grid": {
                **altitude_grid,
                "data_file": "icon-wind.bin",
                "data_length": len(wind_u),
                "data_type": "int16",
                "wind_scale": WIND_SCALE
            }
        }
        with open(weather_json, 'w') as f:
            json.dump(weather_meta, f)

        # Write quality files if available
        if wind_speed_std is not None and wind_dir_spread is not None:
            # Convert to Int16 with scale factors
            std_i16 = np.clip(np.array(wind_speed_std) * SPEED_STD_SCALE, 0, 32767).astype(np.int16)
            spread_i16 = np.clip(np.array(wind_dir_spread) * DIR_SPREAD_SCALE, 0, 32767).astype(np.int16)

            # Write icon-wind-quality.bin
            with open(quality_bin, 'wb') as f:
                f.write(std_i16.tobytes())
                f.write(spread_i16.tobytes())

            # Write icon-wind-quality.json (metadata only)
            quality_meta = {
                "timestamp": timestamp,
                "data_file": "icon-wind-quality.bin",
                "data_length": len(std_i16),
                "data_type": "int16",
                "speed_std_scale": SPEED_STD_SCALE,
                "dir_spread_scale": DIR_SPREAD_SCALE
            }
            with open(quality_json, 'w') as f:
                json.dump(quality_meta, f)

        # Report sizes
        print()
        print("=" * 60)
        print(f"Output written:")
        print(f"  {weather_json}: {weather_json.stat().st_size / 1024:.1f} KB")
        print(f"  {weather_bin}: {weather_bin.stat().st_size / 1024:.1f} KB")
        if quality_bin.exists():
            print(f"  {quality_json}: {quality_json.stat().st_size / 1024:.1f} KB")
            print(f"  {quality_bin}: {quality_bin.stat().st_size / 1024:.1f} KB")
        print(f"Grid: {cols}x{rows}x{len(ALTITUDE_LEVELS)}")
        print("=" * 60)

    except Exception as e:
        print(f"\nFatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print("No output written - keeping previous data in R2")
        sys.exit(1)


if __name__ == "__main__":
    main()
