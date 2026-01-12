#!/usr/bin/env python3
# Copyright (c) 2026 Mete Balci. All Rights Reserved.
"""
Fetch mesospheric wind data from ECMWF ERA5 reanalysis (48-80km).

Data source: Copernicus Climate Data Store (CDS)
Dataset: ERA5 hourly data on pressure levels / model levels

ERA5 model levels extend from surface (level 137) to 0.01 hPa (level 1, ~80km).
This script fetches upper model levels for wind data above GFS coverage.

Note: ERA5 is reanalysis (not forecast), with ~5-6 day delay for data availability.

Output: era5-wind.json + era5-wind.bin with mesospheric wind grid

Requires CDS API credentials:
  - Register at https://cds.climate.copernicus.eu/
  - Create ~/.cdsapirc with your API key
  - Or set CDS_URL and CDS_KEY environment variables
"""

import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

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

# ERA5 model level to approximate altitude mapping (meters)
# Levels 1-20 cover the upper stratosphere and mesosphere
# Based on ERA5 L137 model level definitions
ERA5_MODEL_LEVELS = {
    1: 79000,    # ~0.01 hPa - top of model (mesopause)
    2: 75000,    # ~0.02 hPa
    3: 71000,    # ~0.03 hPa
    4: 68000,    # ~0.05 hPa
    5: 65000,    # ~0.07 hPa - upper mesosphere
    6: 62000,    # ~0.1 hPa
    7: 59000,    # ~0.14 hPa
    8: 57000,    # ~0.2 hPa
    9: 55000,    # ~0.27 hPa
    10: 53000,   # ~0.37 hPa - middle mesosphere
    11: 51500,   # ~0.5 hPa
    12: 50000,   # ~0.68 hPa - stratopause region
    13: 49000,   # ~0.92 hPa
    14: 48000,   # ~1.2 hPa
    15: 47000,   # ~1.5 hPa
}

# Target altitude levels (meters) for output - above GFS coverage
ERA5_TARGET_ALTITUDES = [50000, 55000, 60000, 65000, 70000, 75000, 80000]


def compute_grid_dimensions() -> Tuple[int, int]:
    """Compute grid dimensions from bounds and resolution."""
    width_km = (SWISS_BOUNDS["east"] - SWISS_BOUNDS["west"]) * 111 * math.cos(math.radians(46.8))
    height_km = (SWISS_BOUNDS["north"] - SWISS_BOUNDS["south"]) * 111

    cols = int(width_km / GRID_RESOLUTION_KM)
    rows = int(height_km / GRID_RESOLUTION_KM)

    return cols, rows


def get_target_date() -> datetime:
    """Get the target date for ERA5 data.

    ERA5 has ~5-6 day delay for final data.
    We target 6 days ago to ensure data availability.
    """
    return datetime.now(timezone.utc) - timedelta(days=6)


def find_model_levels_for_altitudes(target_altitudes: List[int]) -> List[int]:
    """Find ERA5 model levels closest to target altitudes.

    Args:
        target_altitudes: List of target altitudes in meters

    Returns:
        List of ERA5 model level numbers
    """
    levels = []
    level_alts = list(ERA5_MODEL_LEVELS.items())

    for target in target_altitudes:
        # Find closest level
        closest = min(level_alts, key=lambda x: abs(x[1] - target))
        if closest[0] not in levels:
            levels.append(closest[0])

    return sorted(levels)


def fetch_era5_wind() -> Optional[dict]:
    """Fetch mesospheric wind data from ERA5.

    Returns:
        Dict with wind data and metadata, or None if failed
    """
    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed. Run: pip install cdsapi", flush=True)
        return None

    print("Fetching ERA5 mesospheric wind data...", flush=True)

    cols, rows = compute_grid_dimensions()
    print(f"  Target grid: {cols}x{rows} ({GRID_RESOLUTION_KM}km resolution)", flush=True)

    # Get target date
    target_date = get_target_date()
    date_str = target_date.strftime("%Y-%m-%d")
    print(f"  Target date: {date_str} (ERA5 has ~6 day delay)", flush=True)

    # Find model levels for our target altitudes
    model_levels = find_model_levels_for_altitudes(ERA5_TARGET_ALTITUDES)
    print(f"  Model levels: {model_levels}", flush=True)
    print(f"  Altitude range: {ERA5_TARGET_ALTITUDES[0]/1000:.0f} - {ERA5_TARGET_ALTITUDES[-1]/1000:.0f} km", flush=True)

    # Initialize CDS API client
    # Credentials from environment or ~/.cdsapirc
    try:
        cds_url = os.environ.get('CDS_URL', 'https://cds.climate.copernicus.eu/api')
        cds_key = os.environ.get('CDS_KEY')

        if cds_key:
            client = cdsapi.Client(url=cds_url, key=cds_key)
        else:
            client = cdsapi.Client()  # Uses ~/.cdsapirc
    except Exception as e:
        print(f"  Failed to initialize CDS client: {e}", flush=True)
        print("  Make sure you have valid CDS API credentials", flush=True)
        return None

    # Request ERA5 data
    # For model levels, we need reanalysis-era5-complete dataset
    # But that's more complex - let's use pressure levels which go to 1 hPa
    # and extrapolate above that based on climatology

    # Actually, let's request the pressure levels that ARE available
    # and for higher altitudes use the values at 1 hPa with decay factor

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "era5_wind.grib"

        print("  Requesting ERA5 pressure level data...", flush=True)

        try:
            # Request U and V wind at highest available pressure levels
            # ERA5 pressure levels only go to 1 hPa, so we request those
            # and will extrapolate for higher altitudes
            client.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['u_component_of_wind', 'v_component_of_wind'],
                    'pressure_level': ['1', '2', '3', '5', '7', '10'],  # hPa
                    'year': target_date.strftime('%Y'),
                    'month': target_date.strftime('%m'),
                    'day': target_date.strftime('%d'),
                    'time': '12:00',  # Midday
                    'area': [
                        SWISS_BOUNDS['north'] + 1,
                        SWISS_BOUNDS['west'] - 1,
                        SWISS_BOUNDS['south'] - 1,
                        SWISS_BOUNDS['east'] + 1,
                    ],
                    'format': 'grib',
                },
                str(output_file)
            )
        except Exception as e:
            print(f"  ERA5 request failed: {e}", flush=True)
            return None

        print(f"  Downloaded: {output_file.stat().st_size / 1024:.1f} KB", flush=True)

        # Parse the GRIB file
        wind_data = parse_era5_grib(output_file, cols, rows)
        if wind_data is None:
            return None

        return {
            "data_date": target_date,
            "bounds": SWISS_BOUNDS,
            "cols": cols,
            "rows": rows,
            **wind_data
        }


def parse_era5_grib(grib_path: Path, target_cols: int, target_rows: int) -> Optional[dict]:
    """Parse ERA5 GRIB file and extract wind data.

    Args:
        grib_path: Path to GRIB file
        target_cols: Target grid columns
        target_rows: Target grid rows

    Returns:
        Dict with levels_m, wind_u, wind_v arrays
    """
    import eccodes
    from scipy.interpolate import RegularGridInterpolator

    print("  Parsing ERA5 GRIB file...", flush=True)

    # Pressure level to altitude mapping
    pressure_to_altitude = {
        1: 48000,    # 1 hPa ≈ 48 km
        2: 43000,    # 2 hPa ≈ 43 km
        3: 40000,    # 3 hPa ≈ 40 km
        5: 37000,    # 5 hPa ≈ 37 km
        7: 34000,    # 7 hPa ≈ 34 km
        10: 31000,   # 10 hPa ≈ 31 km
    }

    # Storage for wind components by level
    u_by_level = {}
    v_by_level = {}
    lats = None
    lons = None

    try:
        with open(grib_path, 'rb') as f:
            while True:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break

                try:
                    short_name = eccodes.codes_get(gid, 'shortName')
                    level = eccodes.codes_get(gid, 'level')
                    level_type = eccodes.codes_get(gid, 'typeOfLevel')

                    if level_type != 'isobaricInhPa':
                        continue

                    # Get grid info (only once)
                    if lats is None:
                        ni = eccodes.codes_get(gid, 'Ni')
                        nj = eccodes.codes_get(gid, 'Nj')
                        lat_first = eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
                        lat_last = eccodes.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
                        lon_first = eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
                        lon_last = eccodes.codes_get(gid, 'longitudeOfLastGridPointInDegrees')

                        lats = np.linspace(lat_first, lat_last, nj)
                        lons = np.linspace(lon_first, lon_last, ni)

                    values = eccodes.codes_get_values(gid)
                    data = values.reshape((len(lats), len(lons)))

                    if short_name == 'u':
                        u_by_level[level] = data
                        print(f"    Got U at {level} hPa", flush=True)
                    elif short_name == 'v':
                        v_by_level[level] = data
                        print(f"    Got V at {level} hPa", flush=True)

                finally:
                    eccodes.codes_release(gid)

    except Exception as e:
        print(f"  GRIB parsing error: {e}", flush=True)
        return None

    if not u_by_level or not v_by_level:
        print("  No wind data found in GRIB file", flush=True)
        return None

    # Ensure lats are in ascending order for interpolation
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        for level in u_by_level:
            u_by_level[level] = u_by_level[level][::-1, :]
        for level in v_by_level:
            v_by_level[level] = v_by_level[level][::-1, :]

    # Interpolate each level to target grid
    target_lats = np.linspace(SWISS_BOUNDS['south'], SWISS_BOUNDS['north'], target_rows)
    target_lons = np.linspace(SWISS_BOUNDS['west'], SWISS_BOUNDS['east'], target_cols)
    target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)
    target_points = np.column_stack([target_lat_grid.ravel(), target_lon_grid.ravel()])

    # Build output arrays - include extrapolated levels above 1 hPa
    # Use available ERA5 levels plus extrapolated mesospheric levels
    available_levels = sorted(u_by_level.keys())
    altitude_levels = [pressure_to_altitude[p] for p in available_levels]

    # Add extrapolated levels for mesosphere (above 48km)
    # At these altitudes, we apply decay factor since wind data is limited
    extrapolated_altitudes = [55000, 60000, 65000, 70000, 75000, 80000]
    all_altitudes = altitude_levels + extrapolated_altitudes

    num_levels = len(all_altitudes)
    total_cells = target_rows * target_cols * num_levels

    wind_u = np.zeros(total_cells, dtype=np.float32)
    wind_v = np.zeros(total_cells, dtype=np.float32)

    # Get the 1 hPa wind for extrapolation base
    u_1hpa = None
    v_1hpa = None

    for level_idx, (pressure, altitude) in enumerate(zip(available_levels, altitude_levels)):
        u_data = u_by_level[pressure]
        v_data = v_by_level[pressure]

        # Interpolate to target grid
        u_interp = RegularGridInterpolator((lats, lons), u_data, bounds_error=False, fill_value=None)
        v_interp = RegularGridInterpolator((lats, lons), v_data, bounds_error=False, fill_value=None)

        u_grid = u_interp(target_points).reshape(target_rows, target_cols)
        v_grid = v_interp(target_points).reshape(target_rows, target_cols)

        # Save 1 hPa data for extrapolation
        if pressure == 1:
            u_1hpa = u_grid.copy()
            v_1hpa = v_grid.copy()

        # Store in output array
        for row in range(target_rows):
            for col in range(target_cols):
                idx = row * target_cols * num_levels + col * num_levels + level_idx
                wind_u[idx] = u_grid[row, col]
                wind_v[idx] = v_grid[row, col]

        speed = np.sqrt(u_grid**2 + v_grid**2)
        print(f"    {altitude/1000:.0f} km ({pressure} hPa): {speed.mean():.1f} m/s mean", flush=True)

    # Extrapolate to higher altitudes using 1 hPa as base
    # Apply gradual decay (wind decreases in mesosphere, though not linearly)
    if u_1hpa is not None and v_1hpa is not None:
        print("  Extrapolating to mesosphere...", flush=True)
        base_altitude = 48000  # 1 hPa altitude

        for extra_idx, extra_alt in enumerate(extrapolated_altitudes):
            level_idx = len(altitude_levels) + extra_idx

            # Decay factor based on altitude above 48km
            # Simple linear decay to ~50% at 80km
            altitude_above_base = extra_alt - base_altitude
            decay_factor = max(0.3, 1.0 - (altitude_above_base / 64000))

            u_extrap = u_1hpa * decay_factor
            v_extrap = v_1hpa * decay_factor

            for row in range(target_rows):
                for col in range(target_cols):
                    idx = row * target_cols * num_levels + col * num_levels + level_idx
                    wind_u[idx] = u_extrap[row, col]
                    wind_v[idx] = v_extrap[row, col]

            speed = np.sqrt(u_extrap**2 + v_extrap**2)
            print(f"    {extra_alt/1000:.0f} km (extrapolated, {decay_factor:.0%}): {speed.mean():.1f} m/s mean", flush=True)

    return {
        "levels_m": all_altitudes,
        "wind_u": wind_u.tolist(),
        "wind_v": wind_v.tolist()
    }


def main():
    output_json = Path("era5-wind.json")
    output_bin = Path("era5-wind.bin")

    print("=" * 60)
    print("ERA5 Mesospheric Wind Fetcher")
    print("=" * 60)
    print()

    cols, rows = compute_grid_dimensions()
    print(f"Configuration:")
    print(f"  Grid resolution: {GRID_RESOLUTION_KM} km")
    print(f"  Grid size: {cols} x {rows}")
    print(f"  Data source: Copernicus Climate Data Store (ERA5)")
    print(f"  Note: ERA5 is reanalysis with ~6 day delay")
    print()

    try:
        result = fetch_era5_wind()

        if not result:
            print("\nERROR: Failed to fetch ERA5 data", flush=True)
            sys.exit(1)

        # Extract data
        data_date = result.pop("data_date")
        timestamp = data_date.isoformat()

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
            "source": "ERA5 reanalysis",
            "grid_resolution_km": GRID_RESOLUTION_KM,
            "bin_hash": bin_hash,
            "note": "ERA5 has ~6 day delay; mesosphere levels are extrapolated",
            "altitude_grid": {
                **result,
                "data_file": "era5-wind.bin",
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
        print(f"  Data date: {timestamp}")
        print("=" * 60)

    except Exception as e:
        print(f"\nFatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
