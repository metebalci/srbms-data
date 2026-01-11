#!/usr/bin/env python3
"""
Fetch real-time snow data from SLF IMIS and MeteoSwiss SwissMetNet stations.

Data sources:
- SLF Measurement API (measurement-api.slf.ch) - Alpine snow stations
- MeteoSwiss ogd-smn (data.geo.admin.ch) - Lower-altitude weather stations

Output:
- snow.json: Station locations, elevations, and snow depths
- snow-voronoi.bin: Uint8 array mapping each terrain cell to nearest station index
"""

import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.spatial import cKDTree

# SLF API endpoints
SLF_STATIONS_URL = "https://measurement-api.slf.ch/public/api/imis/stations"
SLF_MEASUREMENTS_URL = "https://measurement-api.slf.ch/public/api/imis/measurements"

# MeteoSwiss ogd-smn endpoints
MCH_STATIONS_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_stations.csv"
MCH_DATAINV_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_datainventory.csv"
MCH_DATA_BASE = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"

# Snow depth parameter code (current value, 10-min data)
MCH_SNOW_PARAM = "htoauts0"

# Switzerland terrain bounds (LV95) - from base-index.json
TERRAIN_BOUNDS_LV95 = {
    "minX": 2479900,
    "maxX": 2865100,
    "minY": 1061900,
    "maxY": 1302100
}

# Terrain grid dimensions - from base-index.json
TERRAIN_GRID = {
    "cols": 1926,
    "rows": 1201,
    "cellSize": 200  # meters
}


def wgs84_to_lv95(lon: float, lat: float) -> tuple:
    """Convert WGS84 coordinates to Swiss LV95.

    Based on approximate formulas from swisstopo.
    """
    # Convert to sexagesimal seconds
    phi = lat * 3600
    lam = lon * 3600

    # Auxiliary values
    phi_prime = (phi - 169028.66) / 10000
    lam_prime = (lam - 26782.5) / 10000

    # LV95 Easting (E)
    x = 2600072.37 \
        + 211455.93 * lam_prime \
        - 10938.51 * lam_prime * phi_prime \
        - 0.36 * lam_prime * phi_prime * phi_prime \
        - 44.54 * lam_prime * lam_prime * lam_prime

    # LV95 Northing (N)
    y = 1200147.07 \
        + 308807.95 * phi_prime \
        + 3745.25 * lam_prime * lam_prime \
        + 76.63 * phi_prime * phi_prime \
        - 194.56 * lam_prime * lam_prime * phi_prime \
        + 119.79 * phi_prime * phi_prime * phi_prime

    return x, y


def fetch_slf_data() -> list:
    """Fetch snow data from SLF IMIS API (alpine stations)."""
    print("Fetching SLF IMIS station data...", flush=True)

    # Fetch stations
    print("  Fetching stations...", flush=True)
    stations_response = requests.get(SLF_STATIONS_URL, timeout=30)
    stations_response.raise_for_status()
    raw_stations = stations_response.json()
    print(f"  Got {len(raw_stations)} stations", flush=True)

    # Fetch measurements
    print("  Fetching measurements...", flush=True)
    measurements_response = requests.get(SLF_MEASUREMENTS_URL, timeout=30)
    measurements_response.raise_for_status()
    raw_measurements = measurements_response.json()
    print(f"  Got {len(raw_measurements)} measurements", flush=True)

    # Build measurement map (latest measurement per station)
    measurement_map = {}
    for m in raw_measurements:
        code = m.get("station_code")
        if code:
            existing = measurement_map.get(code)
            if not existing or m.get("measure_date", "") > existing.get("measure_date", ""):
                measurement_map[code] = m

    # Filter to SNOW_FLAT stations in Switzerland and convert coordinates
    stations = []
    for s in raw_stations:
        # Only snow stations in Switzerland
        if s.get("type") != "SNOW_FLAT" or s.get("country_code") != "CH":
            continue

        lon = s.get("lon")
        lat = s.get("lat")
        if lon is None or lat is None:
            continue

        # Convert to LV95
        lv95_x, lv95_y = wgs84_to_lv95(lon, lat)

        # Check if within terrain bounds
        if not (TERRAIN_BOUNDS_LV95["minX"] <= lv95_x <= TERRAIN_BOUNDS_LV95["maxX"] and
                TERRAIN_BOUNDS_LV95["minY"] <= lv95_y <= TERRAIN_BOUNDS_LV95["maxY"]):
            continue

        # Get measurement
        measurement = measurement_map.get(s.get("code"))
        snow_depth = measurement.get("HS") if measurement else None
        measure_date = measurement.get("measure_date") if measurement else None

        stations.append({
            "code": s.get("code"),
            "label": s.get("label"),
            "canton": s.get("canton_code"),
            "lv95X": round(lv95_x, 1),
            "lv95Y": round(lv95_y, 1),
            "elevation": s.get("elevation"),
            "snowDepth": snow_depth,  # cm, null if no data
            "measureDate": measure_date
        })

    print(f"  Filtered to {len(stations)} SNOW_FLAT stations within terrain bounds", flush=True)
    with_data = sum(1 for s in stations if s.get("snowDepth") is not None)
    print(f"  {with_data} stations reporting snow depth", flush=True)

    return stations


def fetch_meteoswiss_data() -> list:
    """Fetch snow data from MeteoSwiss SwissMetNet (lower-altitude stations)."""
    print("\nFetching MeteoSwiss SwissMetNet data...", flush=True)

    # Fetch station metadata
    print("  Fetching station metadata...", flush=True)
    stations_response = requests.get(MCH_STATIONS_URL, timeout=30)
    stations_response.raise_for_status()
    stations_csv = csv.DictReader(io.StringIO(stations_response.text), delimiter=';')
    raw_stations = list(stations_csv)
    print(f"  Got {len(raw_stations)} stations", flush=True)

    # Fetch data inventory to find stations with snow sensors
    print("  Fetching data inventory...", flush=True)
    inv_response = requests.get(MCH_DATAINV_URL, timeout=30)
    inv_response.raise_for_status()
    inv_csv = csv.DictReader(io.StringIO(inv_response.text), delimiter=';')

    # Find stations that have snow depth data (correct column names)
    snow_stations = set()
    for row in inv_csv:
        if row.get("parameter_shortname") == MCH_SNOW_PARAM:
            snow_stations.add(row.get("station_abbr"))

    print(f"  {len(snow_stations)} stations have snow depth sensors", flush=True)

    # Build station info map (using correct column names from CSV)
    station_info = {}
    for s in raw_stations:
        code = s.get("station_abbr")
        if code and code in snow_stations:
            try:
                # Coordinates are in LV95 already
                lv95_x = float(s.get("station_coordinates_lv95_east", 0))
                lv95_y = float(s.get("station_coordinates_lv95_north", 0))
                elevation = int(float(s.get("station_height_masl", 0)))

                # Check if within terrain bounds
                if (TERRAIN_BOUNDS_LV95["minX"] <= lv95_x <= TERRAIN_BOUNDS_LV95["maxX"] and
                    TERRAIN_BOUNDS_LV95["minY"] <= lv95_y <= TERRAIN_BOUNDS_LV95["maxY"]):
                    station_info[code] = {
                        "code": code,
                        "label": s.get("station_name", code),
                        "canton": s.get("station_canton", ""),
                        "lv95X": round(lv95_x, 1),
                        "lv95Y": round(lv95_y, 1),
                        "elevation": elevation,
                        "snowDepth": None,
                        "measureDate": None
                    }
            except (ValueError, TypeError):
                continue

    print(f"  {len(station_info)} snow stations within terrain bounds", flush=True)

    # Fetch current snow depth from each station's data file
    print("  Fetching current snow depths...", flush=True)
    fetched = 0
    with_data = 0

    for code in station_info:
        try:
            # Fetch 10-minute "now" data file (most recent values)
            # URL format: ogd-smn_{station_lowercase}_t_now.csv
            code_lower = code.lower()
            data_url = f"{MCH_DATA_BASE}/{code_lower}/ogd-smn_{code_lower}_t_now.csv"
            data_response = requests.get(data_url, timeout=10)
            if data_response.status_code != 200:
                continue

            # Parse CSV and get latest snow depth
            data_csv = csv.DictReader(io.StringIO(data_response.text), delimiter=';')

            # Find the most recent row with snow depth data
            latest_time = None
            latest_depth = None
            for row in data_csv:
                snow_val = row.get(MCH_SNOW_PARAM)
                if snow_val and snow_val.strip() and snow_val != '-':
                    try:
                        depth = float(snow_val)
                        time_str = row.get("reference_timestamp")
                        if time_str and (latest_time is None or time_str > latest_time):
                            latest_time = time_str
                            latest_depth = depth
                    except ValueError:
                        continue

            if latest_depth is not None:
                station_info[code]["snowDepth"] = round(latest_depth)
                station_info[code]["measureDate"] = latest_time
                with_data += 1

            fetched += 1
            if fetched % 20 == 0:
                print(f"    Fetched {fetched}/{len(station_info)} stations...", flush=True)

        except Exception as e:
            # Skip failed stations silently
            continue

    print(f"  Fetched data from {fetched} stations, {with_data} with snow depth", flush=True)

    return list(station_info.values())


def fetch_snow_data() -> dict:
    """Fetch snow data from both SLF IMIS and MeteoSwiss."""
    # Fetch from both sources
    slf_stations = fetch_slf_data()
    mch_stations = fetch_meteoswiss_data()

    # Count stats
    slf_with_data = sum(1 for s in slf_stations if s.get("snowDepth") is not None)
    mch_with_data = sum(1 for s in mch_stations if s.get("snowDepth") is not None)

    # Get snow depth stats
    all_stations = slf_stations + mch_stations
    depths = [s["snowDepth"] for s in all_stations if s.get("snowDepth") is not None and s["snowDepth"] >= 0]
    if depths:
        print(f"\nCombined snow depth range: {min(depths):.0f} - {max(depths):.0f} cm", flush=True)
        print(f"Mean snow depth: {sum(depths)/len(depths):.0f} cm", flush=True)

    # Elevation stats
    elevations = [s["elevation"] for s in all_stations if s.get("elevation")]
    if elevations:
        print(f"Elevation range: {min(elevations)} - {max(elevations)} m", flush=True)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "terrain_bounds_lv95": TERRAIN_BOUNDS_LV95,
        "slf": {
            "source": "SLF IMIS",
            "api_url": SLF_STATIONS_URL,
            "station_count": len(slf_stations),
            "stations_with_data": slf_with_data,
            "stations": slf_stations
        },
        "meteoswiss": {
            "source": "MeteoSwiss SwissMetNet",
            "api_url": MCH_DATA_BASE,
            "station_count": len(mch_stations),
            "stations_with_data": mch_with_data,
            "stations": mch_stations
        }
    }


def generate_voronoi_index(all_stations: list, voronoi_path: Path) -> str:
    """Generate Voronoi index mapping each terrain cell to nearest station.

    Returns MD5 hash of the generated file for verification.
    """
    import time

    print(f"\nGenerating Voronoi index for {len(all_stations)} stations...", flush=True)

    cols = TERRAIN_GRID["cols"]
    rows = TERRAIN_GRID["rows"]
    cell_size = TERRAIN_GRID["cellSize"]
    min_x = TERRAIN_BOUNDS_LV95["minX"]
    min_y = TERRAIN_BOUNDS_LV95["minY"]

    # Build KD-tree from station positions
    t0 = time.perf_counter()
    station_coords = np.array([[s["lv95X"], s["lv95Y"]] for s in all_stations])
    tree = cKDTree(station_coords)
    t1 = time.perf_counter()
    print(f"  KD-tree build: {(t1-t0)*1000:.1f}ms", flush=True)

    # Generate grid of cell center coordinates
    # Cell (col, row) has center at (minX + col*cellSize + cellSize/2, minY + row*cellSize + cellSize/2)
    col_coords = min_x + (np.arange(cols) + 0.5) * cell_size
    row_coords = min_y + (np.arange(rows) + 0.5) * cell_size

    # Create meshgrid of all cell centers
    xx, yy = np.meshgrid(col_coords, row_coords)
    cell_centers = np.column_stack([xx.ravel(), yy.ravel()])

    print(f"  Querying {len(cell_centers):,} cells...", flush=True)

    # Find nearest station for each cell
    t2 = time.perf_counter()
    distances, indices = tree.query(cell_centers)
    t3 = time.perf_counter()
    print(f"  KD-tree query: {(t3-t2)*1000:.1f}ms", flush=True)

    # Convert to uint8 (assumes < 256 stations)
    if len(all_stations) > 255:
        raise ValueError(f"Too many stations ({len(all_stations)}) for uint8 index")

    voronoi_bytes = indices.astype(np.uint8).tobytes()

    # Compute hash for verification (SHA-256, truncated to 16 chars for brevity)
    voronoi_hash = hashlib.sha256(voronoi_bytes).hexdigest()[:16]

    # Save binary file (row-major order, same as terrain)
    with open(voronoi_path, 'wb') as f:
        f.write(voronoi_bytes)

    print(f"  Voronoi index: {voronoi_path} ({voronoi_path.stat().st_size / 1024:.1f} KB)", flush=True)
    print(f"  Hash: {voronoi_hash}", flush=True)

    return voronoi_hash


def main():
    output_path = Path("snow.json")
    voronoi_path = Path("snow-voronoi.bin")

    print("=" * 60)
    print("Snow Data Fetcher (SLF IMIS + MeteoSwiss SwissMetNet)")
    print("=" * 60)
    print()

    try:
        data = fetch_snow_data()

        slf = data['slf']
        mch = data['meteoswiss']

        # Generate Voronoi index first (to get hash for snow.json)
        all_stations = slf['stations'] + mch['stations']
        voronoi_hash = generate_voronoi_index(all_stations, voronoi_path)

        # Add voronoi hash to data for client verification
        data['voronoi_hash'] = voronoi_hash

        # Write snow.json (after voronoi so hash is included)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print()
        print("=" * 60)
        print(f"Output written: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        print(f"SLF IMIS: {slf['station_count']} stations ({slf['stations_with_data']} with data)")
        print(f"MeteoSwiss: {mch['station_count']} stations ({mch['stations_with_data']} with data)")
        total = slf['station_count'] + mch['station_count']
        with_data = slf['stations_with_data'] + mch['stations_with_data']
        print(f"Total: {total} stations ({with_data} with data)")
        print("=" * 60)

    except Exception as e:
        print(f"\nFatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print("No output written")
        sys.exit(1)


if __name__ == "__main__":
    main()
