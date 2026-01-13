#!/usr/bin/env python3
# Copyright (c) 2026 Mete Balci. All Rights Reserved.
"""
Fetch full-atmosphere wind data from ECMWF ERA5 reanalysis (0-80km).

Data source: Copernicus Climate Data Store (CDS)
Dataset: ERA5 hourly data on pressure levels

ERA5 provides wind data from surface to 1 hPa (~48km) via pressure levels.
Above 48km, wind is extrapolated with decay factors.

Note: ERA5 is reanalysis (not forecast), with ~5-6 day delay for data availability.

Output: era5-wind.json + era5-wind.bin with full atmosphere wind grid

Requires CDS API credentials:
  - Register at https://cds.climate.copernicus.eu/
  - Create ~/.cdsapirc with your API key
  - Or set CDS_URL and CDS_KEY environment variables
"""

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import numpy as np

# Switzerland bounds (WGS84)
SWISS_BOUNDS = {
    "west": 5.9,
    "east": 10.5,
    "south": 45.8,
    "north": 47.8
}

# Grid resolution - use native ERA5 0.25° resolution (~31km)
GRID_RESOLUTION_DEG = 0.25

# Scale factor for Int16 encoding
WIND_SCALE = 100  # 0.01 m/s precision

# ERA5 pressure levels covering full atmosphere (hPa) with approximate altitudes (m)
# Standard atmosphere: h = -8500 * ln(p/1013.25)
ERA5_PRESSURE_LEVELS = [
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
    # Stratosphere (12-48km)
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
    (1, 48000),     # Top of ERA5 pressure level data
]

# Mesosphere levels (extrapolated from 1 hPa with decay)
MESOSPHERE_ALTITUDES = [55000, 60000, 65000, 70000, 75000, 80000]


def get_target_date() -> datetime:
    """Get the target date for ERA5 data.

    ERA5 has ~5-6 day delay for final data.
    We target 6 days ago to ensure data availability.
    """
    return datetime.now(timezone.utc) - timedelta(days=6)


def fetch_era5_wind() -> Optional[dict]:
    """Fetch full-atmosphere wind data from ERA5.

    Returns:
        Dict with wind data and metadata, or None if failed
    """
    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed. Run: pip install cdsapi", flush=True)
        return None

    print("Fetching ERA5 full-atmosphere wind data...", flush=True)
    print(f"  Grid resolution: {GRID_RESOLUTION_DEG}° (native ERA5)", flush=True)

    # Get target date
    target_date = get_target_date()
    date_str = target_date.strftime("%Y-%m-%d")
    print(f"  Target date: {date_str} (ERA5 has ~6 day delay)", flush=True)

    # Pressure levels to request
    pressure_levels = [str(p) for p, _ in ERA5_PRESSURE_LEVELS]
    print(f"  Pressure levels: {len(pressure_levels)} ({pressure_levels[0]} to {pressure_levels[-1]} hPa)", flush=True)

    # Initialize CDS API client
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "era5_wind.grib"

        print("  Requesting ERA5 pressure level data...", flush=True)

        try:
            client.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['u_component_of_wind', 'v_component_of_wind'],
                    'pressure_level': pressure_levels,
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

        # Parse the GRIB file - returns native ERA5 grid info
        wind_data = parse_era5_grib(output_file)
        if wind_data is None:
            return None

        return {
            "data_date": target_date,
            **wind_data
        }


def parse_era5_grib(grib_path: Path) -> Optional[dict]:
    """Parse ERA5 GRIB file and extract wind data as-is (no resampling).

    Returns:
        Dict with grid info, levels_m, wind_u, wind_v arrays
    """
    import eccodes

    print("  Parsing ERA5 GRIB file...", flush=True)

    # Storage for wind components by pressure level
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
                    elif short_name == 'v':
                        v_by_level[level] = data

                finally:
                    eccodes.codes_release(gid)

    except Exception as e:
        print(f"  GRIB parsing error: {e}", flush=True)
        return None

    if not u_by_level or not v_by_level:
        print("  No wind data found in GRIB file", flush=True)
        return None

    print(f"  Found {len(u_by_level)} pressure levels", flush=True)
    print(f"  ERA5 grid: {len(lons)}x{len(lats)} (native)", flush=True)

    # ERA5 typically has latitudes north to south (descending order)
    # Physics code expects row 0 = north, which matches this order
    # So we keep the original order - NO flipping needed
    # Just ensure bounds are reported correctly (south < north)
    lat_south = min(lats[0], lats[-1])
    lat_north = max(lats[0], lats[-1])

    rows = len(lats)
    cols = len(lons)

    # Build output altitude levels: pressure levels + mesosphere extrapolation
    output_altitudes = [alt for _, alt in ERA5_PRESSURE_LEVELS] + MESOSPHERE_ALTITUDES
    num_levels = len(output_altitudes)
    total_cells = rows * cols * num_levels

    wind_u = np.zeros(total_cells, dtype=np.float32)
    wind_v = np.zeros(total_cells, dtype=np.float32)

    print("  Storing pressure levels (no resampling)...", flush=True)

    # Process each pressure level - store as-is
    for level_idx, (pressure_hpa, altitude_m) in enumerate(ERA5_PRESSURE_LEVELS):
        if pressure_hpa not in u_by_level or pressure_hpa not in v_by_level:
            print(f"    Warning: {pressure_hpa} hPa not found, skipping", flush=True)
            continue

        u_data = u_by_level[pressure_hpa]
        v_data = v_by_level[pressure_hpa]

        # Store directly (no interpolation)
        for row in range(rows):
            for col in range(cols):
                idx = row * cols * num_levels + col * num_levels + level_idx
                wind_u[idx] = u_data[row, col]
                wind_v[idx] = v_data[row, col]

        speed = np.sqrt(u_data**2 + v_data**2)
        print(f"    {pressure_hpa} hPa (~{altitude_m/1000:.0f} km): {speed.min():.1f} - {speed.max():.1f} m/s", flush=True)

    # Extrapolate mesosphere levels from 1 hPa base
    print("  Extrapolating mesosphere levels...", flush=True)

    if 1 not in u_by_level or 1 not in v_by_level:
        print("  Warning: 1 hPa level not found, cannot extrapolate mesosphere", flush=True)
    else:
        u_base = u_by_level[1]
        v_base = v_by_level[1]
        BASE_ALTITUDE = 48000  # 1 hPa

        for meso_idx, altitude in enumerate(MESOSPHERE_ALTITUDES):
            level_idx = len(ERA5_PRESSURE_LEVELS) + meso_idx

            # Decay factor: wind decreases with altitude above 48km
            altitude_above_base = altitude - BASE_ALTITUDE
            decay_factor = max(0.3, 1.0 - (altitude_above_base / 64000))

            # Apply decay to base level wind
            u_level = u_base * decay_factor
            v_level = v_base * decay_factor

            for row in range(rows):
                for col in range(cols):
                    idx = row * cols * num_levels + col * num_levels + level_idx
                    wind_u[idx] = u_level[row, col]
                    wind_v[idx] = v_level[row, col]

            speed = np.sqrt(u_level**2 + v_level**2)
            print(f"    {altitude/1000:.0f} km (decay {decay_factor:.0%}): {speed.min():.1f} - {speed.max():.1f} m/s", flush=True)

    # Return data with actual ERA5 grid info
    return {
        "bounds": {
            "west": float(min(lons[0], lons[-1])),
            "east": float(max(lons[0], lons[-1])),
            "south": float(lat_south),
            "north": float(lat_north)
        },
        "cols": cols,
        "rows": rows,
        "levels_m": output_altitudes,
        "wind_u": wind_u.tolist(),
        "wind_v": wind_v.tolist()
    }


def main():
    output_json = Path("era5-wind.json")
    output_bin = Path("era5-wind.bin")

    print("=" * 60)
    print("ERA5 Full-Atmosphere Wind Fetcher")
    print("=" * 60)
    print()

    num_levels = len(ERA5_PRESSURE_LEVELS) + len(MESOSPHERE_ALTITUDES)

    print(f"Configuration:")
    print(f"  Data source: ECMWF ERA5 reanalysis")
    print(f"  Grid resolution: {GRID_RESOLUTION_DEG}° (~31km native)")
    print(f"  Altitude levels: {num_levels} (surface to 80km)")
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
            "model_run": timestamp,
            "source": "ERA5 reanalysis (~31km)",
            "grid_resolution_deg": GRID_RESOLUTION_DEG,
            "bin_hash": bin_hash,
            "note": "ERA5 reanalysis has ~6 day delay; mesosphere (55-80km) extrapolated from 1 hPa",
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
