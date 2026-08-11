from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import urlopen

import cfgrib
import numpy as np
import rasterio
import zarr
from ecmwf.opendata import Client as ECMWFClient
from obstore import put
from obstore.store import GCSStore, LocalStore, ObjectStore
from rasterio.features import shapes as raster_shapes
from skyfield.api import load, load_file, wgs84
from skyfield.constants import AU_KM
from tilebox.workflows import Client, ExecutionContext, Runner, Task
from tilebox.workflows.cache import LocalFileSystemCache
from zarr.codecs import BloscCodec
from zarr.storage import ObjectStore as ZarrObjectStore

RESULTS_URI = os.environ.get("SOLAR_ECLIPSE_RESULTS_URI", "outputs")
PUBLIC_RESULTS_BASE = os.environ.get("SOLAR_ECLIPSE_PUBLIC_RESULTS_BASE")
OUTPUT_HEIGHT = 360
OUTPUT_WIDTH = 1440
RESOLUTION_DEGREES = 0.25
SPATIAL_CHUNK_SIZE = 64
NODATA_UINT16 = np.uint16(65535)
ECLIPSE_START_SECOND = 15 * 3600 + 30 * 60
ECLIPSE_END_SECOND = 20 * 3600
FORECAST_VALID_HOURS = (15, 18, 21)
DE440S_URL = "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440s.bsp"
DE440S_SHA256 = "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2"
SUN_RADIUS_KM = 695_700.0
MOON_RADIUS_KM = 1_737.4
WEB_MERCATOR_MAX_LATITUDE = 85.05112878
NASA_2026_PATH_URL = (
    "https://eclipse.gsfc.nasa.gov/SEpath/SEpath2001/SE2026Aug12Tpath.html"
)
# WGS84 coordinates from NASA/GSFC's two-minute central eclipse path table.
# Each coordinate is stored as (longitude, latitude).
NASA_2026_NORTHERN_LIMIT = (
    (108.69, 75.1733),
    (108.758, 75.9367),
    (103.217, 82.1633),
    (90.395, 84.85),
    (65.8233, 86.3433),
    (32.7283, 86.545),
    (8.375, 85.72),
    (-4.81, 84.4817),
    (-12.0017, 83.1317),
    (-16.2167, 81.775),
    (-18.8417, 80.4417),
    (-20.5383, 79.1417),
    (-21.6567, 77.875),
    (-22.3933, 76.6417),
    (-22.865, 75.44),
    (-23.145, 74.2667),
    (-23.285, 73.1167),
    (-23.3133, 71.9917),
    (-23.2583, 70.885),
    (-23.1317, 69.7983),
    (-22.9483, 68.7267),
    (-22.7133, 67.67),
    (-22.4367, 66.6267),
    (-22.12, 65.5933),
    (-21.7683, 64.5717),
    (-21.3817, 63.5567),
    (-20.9617, 62.5483),
    (-20.5083, 61.5467),
    (-20.0217, 60.5483),
    (-19.5, 59.5533),
    (-18.9433, 58.56),
    (-18.3483, 57.565),
    (-17.7117, 56.5683),
    (-17.0283, 55.5683),
    (-16.295, 54.5617),
    (-15.5033, 53.5467),
    (-14.6467, 52.52),
    (-13.7117, 51.4783),
    (-12.685, 50.4167),
    (-11.5467, 49.33),
    (-10.2667, 48.2083),
    (-8.80167, 47.0383),
    (-7.07667, 45.8017),
    (-4.94833, 44.4567),
    (-2.085, 42.9083),
    (3.295, 40.665),
    (6.34, 39.7083),
)
NASA_2026_SOUTHERN_LIMIT = (
    (117.96, 74.9133),
    (119.423, 85.3217),
    (108.432, 87.7533),
    (38.1483, 89.0667),
    (-19.5067, 87.7883),
    (-29.2167, 86.1417),
    (-32.2467, 84.565),
    (-33.4167, 83.0717),
    (-33.8383, 81.65),
    (-33.8967, 80.2917),
    (-33.76, 78.9867),
    (-33.505, 77.7267),
    (-33.1783, 76.5067),
    (-32.805, 75.3233),
    (-32.4, 74.1683),
    (-31.9717, 73.0433),
    (-31.5267, 71.94),
    (-31.0683, 70.86),
    (-30.6, 69.7967),
    (-30.12, 68.7533),
    (-29.6317, 67.7233),
    (-29.1333, 66.7067),
    (-28.625, 65.7033),
    (-28.1067, 64.71),
    (-27.5767, 63.7267),
    (-27.0333, 62.75),
    (-26.4767, 61.78),
    (-25.905, 60.8167),
    (-25.3167, 59.8567),
    (-24.7067, 58.9),
    (-24.0767, 57.945),
    (-23.4217, 56.99),
    (-22.7383, 56.0367),
    (-22.025, 55.0783),
    (-21.275, 54.1183),
    (-20.485, 53.1517),
    (-19.6467, 52.1767),
    (-18.755, 51.1933),
    (-17.7983, 50.195),
    (-16.765, 49.1817),
    (-15.6383, 48.1467),
    (-14.3967, 47.0833),
    (-13.0083, 45.9833),
    (-11.42, 44.8317),
    (-9.55167, 43.6067),
    (-7.23667, 42.2633),
    (-4.04, 40.6833),
    (4.54, 37.69),
)
NASA_2026_CENTERLINE = (
    (113.452, 75.0783),
    (112.487, 82.275),
    (104.215, 85.295),
    (81.525, 87.2783),
    (33.0, 87.8233),
    (-1.63833, 86.835),
    (-15.1817, 85.4033),
    (-21.1867, 83.9317),
    (-24.2717, 82.495),
    (-25.9917, 81.11),
    (-26.9817, 79.7733),
    (-27.54, 78.4833),
    (-27.825, 77.2333),
    (-27.9283, 76.0183),
    (-27.905, 74.8367),
    (-27.7883, 73.6833),
    (-27.6033, 72.5567),
    (-27.3617, 71.45),
    (-27.0783, 70.365),
    (-26.76, 69.2983),
    (-26.41, 68.2467),
    (-26.0317, 67.21),
    (-25.63, 66.185),
    (-25.205, 65.1717),
    (-24.7567, 64.1683),
    (-24.2867, 63.1717),
    (-23.7933, 62.1833),
    (-23.2767, 61.2),
    (-22.7367, 60.2217),
    (-22.17, 59.245),
    (-21.5733, 58.2717),
    (-20.9467, 57.2967),
    (-20.2867, 56.3217),
    (-19.5883, 55.3433),
    (-18.8467, 54.3617),
    (-18.0567, 53.3717),
    (-17.2117, 52.3717),
    (-16.3033, 51.36),
    (-15.3167, 50.3333),
    (-14.2383, 49.285),
    (-13.0483, 48.2117),
    (-11.715, 47.1017),
    (-10.19, 45.9433),
    (-8.39833, 44.7133),
    (-6.18833, 43.3717),
    (-3.185, 41.8167),
    (2.95, 39.4083),
    (5.415, 38.68),
)


def _preferred_cache_dir() -> Path:
    for value in (
        os.environ.get("SOLAR_ECLIPSE_CACHE_DIR"),
        os.environ.get("CACHE_DIR"),
        "/cache",
    ):
        if not value:
            continue
        root = Path(value)
        if value == "/cache" and not root.exists():
            continue
        try:
            path = root / "solar-eclipse-weather"
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    path = Path(".cache") / "solar-eclipse-weather"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalized_prefix(prefix: str) -> str:
    normalized = prefix.strip("/")
    if not normalized or ".." in normalized.split("/"):
        raise ValueError("output_prefix must be a non-empty object prefix without '..'")
    return normalized


def _cube_uri(prefix: str) -> str:
    path = f"{_normalized_prefix(prefix)}/eclipse_weather.zarr"
    if urlparse(RESULTS_URI).scheme:
        return f"{RESULTS_URI.rstrip('/')}/{path}"
    return str(Path(RESULTS_URI) / path)


def _public_url(prefix: str, filename: str) -> str:
    key = f"{_normalized_prefix(prefix)}/{filename}"
    if PUBLIC_RESULTS_BASE:
        return f"{PUBLIC_RESULTS_BASE.rstrip('/')}/{quote(key, safe='/')}"

    parsed = urlparse(RESULTS_URI)
    if parsed.scheme == "gs":
        path = "/".join(part for part in (parsed.path.strip("/"), key) if part)
        return f"https://storage.googleapis.com/{parsed.netloc}/{quote(path, safe='/')}"

    root = Path(parsed.path if parsed.scheme == "file" else RESULTS_URI)
    return (root / key).resolve().as_uri()


@lru_cache
def _object_store(uri: str) -> ObjectStore:
    parsed = urlparse(uri)
    prefix = parsed.path.lstrip("/")
    if parsed.scheme == "gs":
        return GCSStore(bucket=parsed.netloc, prefix=prefix)
    if parsed.scheme == "file":
        return LocalStore(parsed.path, mkdir=True)
    if parsed.scheme == "":
        return LocalStore(uri, mkdir=True)
    raise ValueError("Only gs://, file://, and local stores are supported")


def _zarr_store(uri: str, *, read_only: bool = False) -> ZarrObjectStore:
    return ZarrObjectStore(_object_store(uri), read_only=read_only)


@lru_cache
def _results_store() -> ObjectStore:
    return _object_store(RESULTS_URI)


def _upload_result(
    prefix: str, filename: str, content: Path | bytes, content_type: str
) -> str:
    key = f"{_normalized_prefix(prefix)}/{filename}"
    options = {}
    if urlparse(RESULTS_URI).scheme == "gs":
        options["attributes"] = {
            "Content-Type": content_type,
            "Cache-Control": "public, max-age=300",
        }
    put(_results_store(), key, content, **options)
    return _public_url(prefix, filename)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_ephemeris() -> Path:
    destination = _preferred_cache_dir() / "de440s.bsp"
    if destination.is_file() and _sha256(destination) == DE440S_SHA256:
        return destination

    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    with urlopen(DE440S_URL, timeout=120) as response, temporary.open("wb") as file:
        while block := response.read(1024 * 1024):
            file.write(block)
    if _sha256(temporary) != DE440S_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("Downloaded JPL DE440s ephemeris failed its SHA-256 check")
    os.replace(temporary, destination)
    return destination


@lru_cache(maxsize=4)
def _astronomy(ephemeris_path: str):
    return load_file(ephemeris_path), load.timescale()


def _grid_coordinates(
    y_start: int = 0,
    y_stop: int = OUTPUT_HEIGHT,
    x_start: int = 0,
    x_stop: int = OUTPUT_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    latitudes = (
        90.0 - (np.arange(y_start, y_stop, dtype=np.float64) + 0.5) * RESOLUTION_DEGREES
    )
    longitudes = (
        -180.0
        + (np.arange(x_start, x_stop, dtype=np.float64) + 0.5) * RESOLUTION_DEGREES
    )
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    return latitude_grid, longitude_grid


def _circle_overlap_fraction(
    separation: np.ndarray,
    sun_radius: np.ndarray,
    moon_radius: np.ndarray,
) -> np.ndarray:
    separation, sun_radius, moon_radius = np.broadcast_arrays(
        separation, sun_radius, moon_radius
    )
    overlap = np.zeros(separation.shape, dtype=np.float64)

    contained = separation <= np.abs(moon_radius - sun_radius)
    overlap[contained] = np.minimum(
        1.0,
        np.square(moon_radius[contained] / sun_radius[contained]),
    )

    partial = (separation < sun_radius + moon_radius) & ~contained
    if np.any(partial):
        distance = separation[partial]
        sun = sun_radius[partial]
        moon = moon_radius[partial]
        sun_angle = np.arccos(
            np.clip(
                (distance**2 + sun**2 - moon**2) / (2.0 * distance * sun), -1.0, 1.0
            )
        )
        moon_angle = np.arccos(
            np.clip(
                (distance**2 + moon**2 - sun**2) / (2.0 * distance * moon), -1.0, 1.0
            )
        )
        lens = 0.5 * np.sqrt(
            np.clip(
                (-distance + sun + moon)
                * (distance + sun - moon)
                * (distance - sun + moon)
                * (distance + sun + moon),
                0.0,
                None,
            )
        )
        overlap[partial] = (sun**2 * sun_angle + moon**2 * moon_angle - lens) / (
            math.pi * sun**2
        )
    return np.clip(overlap, 0.0, 1.0)


def _skyfield_time(timescale, event_date: date, utc_seconds: int | np.ndarray):
    seconds = np.asarray(utc_seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return timescale.utc(
        event_date.year,
        event_date.month,
        event_date.day,
        hours,
        minutes,
        remaining_seconds,
    )


def _eclipse_geometry(
    ephemeris,
    timescale,
    event_date: date,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    utc_seconds: int | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_latitudes = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    flat_longitudes = np.asarray(longitudes, dtype=np.float64).reshape(-1)
    seconds = np.asarray(utc_seconds)
    time = _skyfield_time(timescale, event_date, seconds)

    earth = ephemeris["earth"]
    sun = ephemeris["sun"]
    moon = ephemeris["moon"]
    sun_geocentric = earth.at(time).observe(sun).apparent().position.au
    moon_geocentric = earth.at(time).observe(moon).apparent().position.au
    site_position = wgs84.latlon(flat_latitudes, flat_longitudes).at(time).position.au

    if seconds.ndim == 0:
        sun_geocentric = sun_geocentric[:, np.newaxis]
        moon_geocentric = moon_geocentric[:, np.newaxis]

    sun_vector = sun_geocentric - site_position
    moon_vector = moon_geocentric - site_position
    sun_distance = np.linalg.norm(sun_vector, axis=0)
    moon_distance = np.linalg.norm(moon_vector, axis=0)
    separation = np.arccos(
        np.clip(
            np.sum(sun_vector * moon_vector, axis=0) / (sun_distance * moon_distance),
            -1.0,
            1.0,
        )
    )
    sun_radius = np.arcsin(SUN_RADIUS_KM / (sun_distance * AU_KM))
    moon_radius = np.arcsin(MOON_RADIUS_KM / (moon_distance * AU_KM))
    altitude = np.degrees(
        np.arcsin(
            np.clip(
                np.sum(sun_vector * site_position, axis=0)
                / (sun_distance * np.linalg.norm(site_position, axis=0)),
                -1.0,
                1.0,
            )
        )
    )
    obscuration = _circle_overlap_fraction(separation, sun_radius, moon_radius)
    obscuration[altitude <= 0.0] = 0.0
    total = (
        (altitude > 0.0)
        & (moon_radius >= sun_radius)
        & (separation <= moon_radius - sun_radius)
    )
    return obscuration, altitude, total


def calculate_eclipse_chunk(
    event_date: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    ephemeris_path: Path,
) -> dict[str, np.ndarray]:
    parsed_date = date.fromisoformat(event_date)
    shape = np.asarray(latitudes).shape
    flat_latitudes = np.asarray(latitudes).reshape(-1)
    flat_longitudes = np.asarray(longitudes).reshape(-1)
    ephemeris, timescale = _astronomy(str(ephemeris_path))

    best_obscuration = np.zeros(flat_latitudes.size, dtype=np.float64)
    best_seconds = np.full(flat_latitudes.size, ECLIPSE_START_SECOND, dtype=np.int32)
    best_altitude = np.zeros(flat_latitudes.size, dtype=np.float64)
    ever_total = np.zeros(flat_latitudes.size, dtype=bool)

    for utc_second in range(ECLIPSE_START_SECOND, ECLIPSE_END_SECOND + 1, 300):
        obscuration, altitude, total = _eclipse_geometry(
            ephemeris,
            timescale,
            parsed_date,
            flat_latitudes,
            flat_longitudes,
            utc_second,
        )
        better = obscuration > best_obscuration
        best_obscuration[better] = obscuration[better]
        best_seconds[better] = utc_second
        best_altitude[better] = altitude[better]
        ever_total |= total

    active_indices = np.flatnonzero(best_obscuration > 0.0)
    if active_indices.size:
        active_latitudes = flat_latitudes[active_indices]
        active_longitudes = flat_longitudes[active_indices]
        coarse_best = best_seconds[active_indices].copy()
        for offset in range(-300, 301, 30):
            candidate_seconds = np.clip(
                coarse_best + offset,
                ECLIPSE_START_SECOND,
                ECLIPSE_END_SECOND,
            )
            obscuration, altitude, total = _eclipse_geometry(
                ephemeris,
                timescale,
                parsed_date,
                active_latitudes,
                active_longitudes,
                candidate_seconds,
            )
            current = best_obscuration[active_indices]
            better = obscuration > current
            selected = active_indices[better]
            best_obscuration[selected] = obscuration[better]
            best_seconds[selected] = candidate_seconds[better]
            best_altitude[selected] = altitude[better]
            ever_total[active_indices] |= total

    eclipse_class = np.zeros(flat_latitudes.size, dtype=np.uint8)
    eclipse_class[best_obscuration > 0.0] = 1
    eclipse_class[ever_total] = 2
    return {
        "obscuration_fraction": best_obscuration.reshape(shape).astype(np.float32),
        "max_utc_second": best_seconds.reshape(shape).astype(np.uint32),
        "sun_altitude_degree": best_altitude.reshape(shape).astype(np.float32),
        "eclipse_class": eclipse_class.reshape(shape),
    }


def _forecast_bracket(
    valid_times: np.ndarray, target_time: int
) -> tuple[int, int, float]:
    valid = np.asarray(valid_times, dtype=np.int64)
    if valid.ndim != 1 or valid.size == 0 or np.any(np.diff(valid) <= 0):
        raise ValueError("Forecast valid times must be a non-empty increasing array")
    upper = int(np.searchsorted(valid, target_time, side="left"))
    if upper == 0:
        return 0, 0, 0.0
    if upper >= valid.size:
        last = valid.size - 1
        return last, last, 0.0
    lower = upper - 1
    weight = (target_time - int(valid[lower])) / (int(valid[upper]) - int(valid[lower]))
    return lower, upper, float(weight)


def _ecmwf_valid_times(
    run_date: date, run_hour: int, event_date: date
) -> tuple[list[int], list[datetime]]:
    run_time = datetime.combine(run_date, datetime.min.time(), tzinfo=UTC) + timedelta(
        hours=run_hour
    )
    valid_times = [
        datetime.combine(event_date, datetime.min.time(), tzinfo=UTC)
        + timedelta(hours=hour)
        for hour in FORECAST_VALID_HOURS
    ]
    steps = [
        int((valid_time - run_time).total_seconds() // 3600)
        for valid_time in valid_times
    ]
    if any(step < 0 or step % 3 != 0 for step in steps):
        raise ValueError(
            "ECMWF IFS open-data run must provide the 15, 18, and 21 UTC 3-hour steps"
        )
    return steps, valid_times


def _read_forecast_grib(path: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = cfgrib.open_dataset(path, backend_kwargs={"indexpath": ""})
    cloud = np.asarray(dataset["tcc"].values, dtype=np.float32)
    if cloud.ndim == 2:
        cloud = cloud[np.newaxis, :, :]
    if cloud.shape[1:] != (721, 1440):
        raise ValueError(f"Unexpected ECMWF 0.25° grid shape: {cloud.shape}")

    valid_times = (
        np.atleast_1d(dataset["valid_time"].values)
        .astype("datetime64[s]")
        .astype(np.int64)
    )
    if valid_times.size != cloud.shape[0]:
        raise ValueError("Forecast valid-time count does not match cloud-cover layers")

    north = cloud[:, :OUTPUT_HEIGHT, :]
    south = cloud[:, 1 : OUTPUT_HEIGHT + 1, :]
    centered = 0.25 * (
        north + south + np.roll(north, -1, axis=2) + np.roll(south, -1, axis=2)
    )
    cloud_percent = np.rint(np.clip(centered, 0.0, 1.0) * 100.0).astype(np.uint8)
    return cloud_percent, valid_times


def _initialize_cube(
    cube_uri: str,
    event_date: str,
    forecast_run: str,
    valid_times: np.ndarray,
    cloud_percent: np.ndarray,
) -> None:
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    spatial_chunks = (SPATIAL_CHUNK_SIZE, SPATIAL_CHUNK_SIZE)
    group = zarr.group(
        store=_zarr_store(cube_uri),
        overwrite=True,
        attributes={
            "title": "Cloud forecast and solar eclipse visibility cube",
            "event_date": event_date,
            "event": "Total solar eclipse of 12 August 2026",
            "forecast_model": "ECMWF IFS 0.25 degree open data",
            "forecast_run": forecast_run,
            "forecast_parameter": "tcc (total cloud cover)",
            "forecast_license": "CC BY 4.0",
            "forecast_valid_times_utc": [
                datetime.fromtimestamp(int(value), tz=UTC).isoformat()
                for value in valid_times
            ],
            "ephemeris": "JPL DE440s",
            "crs": "EPSG:4326",
            "transform": [
                RESOLUTION_DEGREES,
                0.0,
                -180.0,
                0.0,
                -RESOLUTION_DEGREES,
                90.0,
            ],
            "shape": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "spatial_chunk_shape": list(spatial_chunks),
            "method": (
                "ECMWF total cloud cover is bilinearly centered onto 0.25 degree cells. "
                "Each spatial chunk is linearly interpolated between the bracketing 3-hour "
                "forecast layers at that chunk's median local maximum-eclipse time."
            ),
        },
    )
    group.create_array(
        "time",
        data=np.asarray(valid_times, dtype=np.int64),
        chunks=(len(valid_times),),
        dimension_names=("time",),
        attributes={
            "standard_name": "time",
            "units": "seconds since 1970-01-01T00:00:00Z",
        },
    )
    latitudes, longitudes = _grid_coordinates()
    group.create_array(
        "latitude",
        data=latitudes[:, 0].astype(np.float32),
        chunks=(SPATIAL_CHUNK_SIZE,),
        dimension_names=("latitude",),
        attributes={"standard_name": "latitude", "units": "degrees_north"},
    )
    group.create_array(
        "longitude",
        data=longitudes[0, :].astype(np.float32),
        chunks=(SPATIAL_CHUNK_SIZE,),
        dimension_names=("longitude",),
        attributes={"standard_name": "longitude", "units": "degrees_east"},
    )
    group.create_array(
        "forecast_cloud_cover_percent",
        data=cloud_percent,
        chunks=(1, *spatial_chunks),
        compressors=(compressor,),
        dimension_names=("time", "latitude", "longitude"),
        attributes={
            "long_name": "ECMWF IFS total cloud cover",
            "units": "percent",
            "valid_min": 0,
            "valid_max": 100,
        },
    )

    shape = (OUTPUT_HEIGHT, OUTPUT_WIDTH)
    common = {
        "shape": shape,
        "chunks": spatial_chunks,
        "compressors": (compressor,),
        "dimension_names": ("latitude", "longitude"),
    }
    group.create_array(
        "eclipse_obscuration_hundredths_percent",
        **common,
        dtype=np.uint16,
        fill_value=0,
        attributes={"units": "0.01 percent", "valid_min": 0, "valid_max": 10000},
    )
    group.create_array(
        "matched_cloud_cover_hundredths_percent",
        **common,
        dtype=np.uint16,
        fill_value=0,
        attributes={"units": "0.01 percent", "valid_min": 0, "valid_max": 10000},
    )
    group.create_array(
        "clear_sky_hundredths_percent",
        **common,
        dtype=np.uint16,
        fill_value=0,
        attributes={"units": "0.01 percent", "valid_min": 0, "valid_max": 10000},
    )
    group.create_array(
        "max_eclipse_utc_second",
        **common,
        dtype=np.uint32,
        fill_value=0,
        attributes={"units": "seconds since event-date midnight UTC"},
    )
    group.create_array(
        "forecast_effective_utc_second",
        **common,
        dtype=np.uint32,
        fill_value=0,
        attributes={"units": "seconds since event-date midnight UTC"},
    )
    group.create_array(
        "sun_altitude_hundredths_degree",
        **common,
        dtype=np.int16,
        fill_value=0,
        attributes={"units": "0.01 degree"},
    )
    group.create_array(
        "eclipse_class",
        **common,
        dtype=np.uint8,
        fill_value=0,
        attributes={"flag_values": [0, 1, 2], "flag_meanings": "none partial total"},
    )


def _write_chunk(
    cube_uri: str,
    event_date: str,
    y_start: int,
    y_stop: int,
    x_start: int,
    x_stop: int,
    ephemeris_path: Path,
) -> dict[str, int | float]:
    group = zarr.open_group(store=_zarr_store(cube_uri), mode="a")
    latitudes, longitudes = _grid_coordinates(y_start, y_stop, x_start, x_stop)
    eclipse = calculate_eclipse_chunk(event_date, latitudes, longitudes, ephemeris_path)
    obscuration = eclipse["obscuration_fraction"]
    active = obscuration > 0.0

    event_midnight = int(
        datetime.combine(
            date.fromisoformat(event_date), datetime.min.time(), tzinfo=UTC
        ).timestamp()
    )
    if np.any(active):
        effective_second = int(np.median(eclipse["max_utc_second"][active]))
    else:
        effective_second = 18 * 3600
    effective_timestamp = event_midnight + effective_second
    valid_times = np.asarray(group["time"][:], dtype=np.int64)
    lower, upper, weight = _forecast_bracket(valid_times, effective_timestamp)
    cloud_lower = np.asarray(
        group["forecast_cloud_cover_percent"][lower, y_start:y_stop, x_start:x_stop],
        dtype=np.float32,
    )
    if upper == lower:
        cloud = cloud_lower
    else:
        cloud_upper = np.asarray(
            group["forecast_cloud_cover_percent"][
                upper, y_start:y_stop, x_start:x_stop
            ],
            dtype=np.float32,
        )
        cloud = cloud_lower * (1.0 - weight) + cloud_upper * weight

    output_slice = np.s_[y_start:y_stop, x_start:x_stop]
    group["eclipse_obscuration_hundredths_percent"][output_slice] = np.rint(
        obscuration * 10_000.0
    ).astype(np.uint16)
    group["matched_cloud_cover_hundredths_percent"][output_slice] = np.rint(
        np.clip(cloud, 0.0, 100.0) * 100.0
    ).astype(np.uint16)
    group["clear_sky_hundredths_percent"][output_slice] = np.rint(
        np.clip(100.0 - cloud, 0.0, 100.0) * 100.0
    ).astype(np.uint16)
    group["max_eclipse_utc_second"][output_slice] = eclipse["max_utc_second"]
    group["forecast_effective_utc_second"][output_slice] = np.full(
        obscuration.shape,
        effective_second,
        dtype=np.uint32,
    )
    group["sun_altitude_hundredths_degree"][output_slice] = np.rint(
        eclipse["sun_altitude_degree"] * 100.0
    ).astype(np.int16)
    group["eclipse_class"][output_slice] = eclipse["eclipse_class"]
    return {
        "active_pixels": int(np.count_nonzero(active)),
        "total_pixels": int(np.count_nonzero(eclipse["eclipse_class"] == 2)),
        "effective_utc_second": effective_second,
        "forecast_lower_index": lower,
        "forecast_upper_index": upper,
        "forecast_weight": weight,
    }


def _cog_band_arrays(group) -> list[np.ndarray]:
    obscuration = np.asarray(
        group["eclipse_obscuration_hundredths_percent"][:], dtype=np.uint16
    )
    cloud = np.asarray(
        group["matched_cloud_cover_hundredths_percent"][:], dtype=np.uint16
    )
    clear = np.asarray(group["clear_sky_hundredths_percent"][:], dtype=np.uint16)
    max_minute = np.rint(np.asarray(group["max_eclipse_utc_second"][:]) / 60.0).astype(
        np.uint16
    )
    eclipse_class = np.asarray(group["eclipse_class"][:], dtype=np.uint16)
    forecast_minute = np.rint(
        np.asarray(group["forecast_effective_utc_second"][:]) / 60.0
    ).astype(np.uint16)
    altitude = np.clip(
        np.asarray(group["sun_altitude_hundredths_degree"][:], dtype=np.int32) + 9000,
        0,
        18000,
    ).astype(np.uint16)

    valid = obscuration >= 10
    arrays = [
        obscuration,
        cloud,
        clear,
        max_minute,
        eclipse_class,
        forecast_minute,
        altitude,
    ]
    for array in arrays:
        array[~valid] = NODATA_UINT16
    return arrays


def _write_cog(group, path: Path) -> dict[str, int | float]:
    arrays = _cog_band_arrays(group)
    height, width = arrays[0].shape
    transform_values = group.attrs.get(
        "transform",
        [RESOLUTION_DEGREES, 0.0, -180.0, 0.0, -RESOLUTION_DEGREES, 90.0],
    )
    transform = rasterio.Affine(*transform_values)
    descriptions = (
        "Eclipse obscuration (0.01 percent)",
        "Matched total cloud cover (0.01 percent)",
        "Clear-sky fraction (0.01 percent)",
        "Maximum eclipse UTC minute of day",
        "Eclipse class (1 partial, 2 total)",
        "Effective forecast UTC minute of day",
        "Sun altitude plus 90 degrees (0.01 degree)",
    )
    with rasterio.open(
        path,
        "w",
        driver="COG",
        width=width,
        height=height,
        count=len(arrays),
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
        nodata=int(NODATA_UINT16),
        compress="DEFLATE",
        predictor="YES",
        blocksize=256,
        overview_resampling="nearest",
        num_threads="ALL_CPUS",
    ) as dataset:
        for band_index, (array, description) in enumerate(
            zip(arrays, descriptions, strict=True), 1
        ):
            dataset.write(array, band_index)
            dataset.set_band_description(band_index, description)
        dataset.update_tags(
            EVENT_DATE=str(group.attrs["event_date"]),
            FORECAST_MODEL=str(group.attrs["forecast_model"]),
            FORECAST_RUN=str(group.attrs["forecast_run"]),
            FORECAST_LICENSE=str(group.attrs["forecast_license"]),
            EPHEMERIS=str(group.attrs["ephemeris"]),
        )
    return {
        "partial_pixels": int(np.count_nonzero(arrays[4] == 1)),
        "total_pixels": int(np.count_nonzero(arrays[4] == 2)),
        "clear_pixels": int(
            np.count_nonzero((arrays[2] >= 7500) & (arrays[2] != NODATA_UINT16))
        ),
        "cloudy_pixels": int(
            np.count_nonzero((arrays[1] >= 7500) & (arrays[1] != NODATA_UINT16))
        ),
    }


def _web_mercator_line_geometry(
    coordinates: tuple[tuple[float, float], ...],
) -> dict[str, object]:
    segments: list[list[list[float]]] = []
    segment: list[list[float]] = []
    previous: tuple[float, float] | None = None
    for point in coordinates:
        visible = abs(point[1]) <= WEB_MERCATOR_MAX_LATITUDE
        if previous is not None:
            previous_visible = abs(previous[1]) <= WEB_MERCATOR_MAX_LATITUDE
            if visible != previous_visible:
                boundary_latitude = math.copysign(
                    WEB_MERCATOR_MAX_LATITUDE,
                    point[1] if not visible else previous[1],
                )
                fraction = (boundary_latitude - previous[1]) / (point[1] - previous[1])
                boundary = [
                    previous[0] + fraction * (point[0] - previous[0]),
                    boundary_latitude,
                ]
                if previous_visible:
                    segment.append(boundary)
                    segments.append(segment)
                    segment = []
                else:
                    segment = [boundary]
        if visible:
            segment.append(list(point))
        previous = point
    if segment:
        segments.append(segment)
    segments = [value for value in segments if len(value) >= 2]
    if len(segments) == 1:
        return {"type": "LineString", "coordinates": segments[0]}
    return {"type": "MultiLineString", "coordinates": segments}


def _nasa_2026_eclipse_path_geojson() -> dict[str, object]:
    source = {
        "source": "Eclipse Predictions by Fred Espenak, NASA/GSFC",
        "source_url": NASA_2026_PATH_URL,
    }
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "role": "totality-corridor",
                    "boundary": "northern",
                    "name": "Northern limit of totality (NASA/GSFC)",
                    **source,
                },
                "geometry": _web_mercator_line_geometry(NASA_2026_NORTHERN_LIMIT),
            },
            {
                "type": "Feature",
                "properties": {
                    "role": "totality-corridor",
                    "boundary": "southern",
                    "name": "Southern limit of totality (NASA/GSFC)",
                    **source,
                },
                "geometry": _web_mercator_line_geometry(NASA_2026_SOUTHERN_LIMIT),
            },
            {
                "type": "Feature",
                "properties": {
                    "role": "centerline",
                    "name": "Eclipse centerline (NASA/GSFC)",
                    "start_utc": "16:58",
                    "end_utc": "18:34",
                    **source,
                },
                "geometry": _web_mercator_line_geometry(NASA_2026_CENTERLINE),
            },
        ],
    }


def _eclipse_path_geojson(group) -> dict[str, object]:
    if str(group.attrs.get("event_date", "")) == "2026-08-12":
        return _nasa_2026_eclipse_path_geojson()

    eclipse_class = np.asarray(group["eclipse_class"][:], dtype=np.uint8)
    totality = eclipse_class == 2
    transform_values = group.attrs.get(
        "transform",
        [RESOLUTION_DEGREES, 0.0, -180.0, 0.0, -RESOLUTION_DEGREES, 90.0],
    )
    transform = rasterio.Affine(*transform_values)
    features: list[dict[str, object]] = []
    for geometry, _ in raster_shapes(
        totality.astype(np.uint8),
        mask=totality,
        connectivity=8,
        transform=transform,
    ):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "role": "totality-corridor",
                    "name": "Path of totality",
                },
                "geometry": geometry,
            }
        )

    rows, columns = np.where(totality)
    if rows.size:
        maximum_seconds = np.asarray(
            group["max_eclipse_utc_second"][:], dtype=np.uint32
        )[totality].astype(np.int64)
        latitudes = 90.0 - (rows + 0.5) * RESOLUTION_DEGREES
        longitudes = -180.0 + (columns + 0.5) * RESOLUTION_DEGREES
        points: list[list[float]] = []
        first_bin = int(maximum_seconds.min() // 120 * 120)
        last_bin = int(maximum_seconds.max() // 120 * 120)
        for bin_start in range(first_bin, last_bin + 1, 120):
            selected = (maximum_seconds >= bin_start) & (
                maximum_seconds < bin_start + 120
            )
            if not np.any(selected):
                continue
            longitude_radians = np.deg2rad(longitudes[selected])
            longitude = math.degrees(
                math.atan2(
                    float(np.sin(longitude_radians).mean()),
                    float(np.cos(longitude_radians).mean()),
                )
            )
            latitude = float(np.median(latitudes[selected]))
            points.append([round(longitude, 4), round(latitude, 4)])

        segments: list[list[list[float]]] = []
        for point in points:
            if not segments or abs(point[0] - segments[-1][-1][0]) > 180.0:
                segments.append([point])
            else:
                segments[-1].append(point)
        segments = [segment for segment in segments if len(segment) >= 2]
        if segments:
            geometry: dict[str, object]
            if len(segments) == 1:
                geometry = {"type": "LineString", "coordinates": segments[0]}
            else:
                geometry = {"type": "MultiLineString", "coordinates": segments}
            start_second = int(maximum_seconds.min())
            end_second = int(maximum_seconds.max())
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "role": "centerline",
                        "name": "Approximate eclipse centerline",
                        "start_utc": (
                            f"{start_second // 3600:02d}:"
                            f"{start_second % 3600 // 60:02d}"
                        ),
                        "end_utc": (
                            f"{end_second // 3600:02d}:{end_second % 3600 // 60:02d}"
                        ),
                    },
                    "geometry": geometry,
                }
            )

    return {"type": "FeatureCollection", "features": features}


def _viewer_html(metadata: dict[str, object]) -> str:
    config = json.dumps(metadata, separators=(",", ":"))
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>12 August 2026 eclipse cloud outlook</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    html, body, #map { width: 100%; height: 100%; margin: 0; }
    body { background: #09111d; }
    .panel { position: absolute; z-index: 1000; top: 16px; left: 16px; width: min(390px, calc(100vw - 56px));
      padding: 16px 18px; border: 1px solid #ffffff2b; border-radius: 14px; background: #08111de8;
      box-shadow: 0 12px 40px #0008; backdrop-filter: blur(10px); }
    h1 { margin: 0 0 8px; font-size: 20px; line-height: 1.2; }
    p { margin: 6px 0; color: #c8d4e3; font-size: 13px; line-height: 1.4; }
    select { width: 100%; margin: 10px 0; padding: 8px 10px; border-radius: 8px; color: white;
      border: 1px solid #ffffff38; background: #172336; }
    .legend { display: flex; align-items: center; gap: 7px; margin-top: 8px; font-size: 12px; color: #dbe6f2; }
    .swatch { width: 20px; height: 10px; border-radius: 3px; }
    .meta { color: #8194ab; font-size: 11px; }
    .leaflet-popup-content-wrapper, .leaflet-popup-tip { background: #101b2a; color: #f5f8fb; }
    .eclipse-corridor { filter: drop-shadow(0 0 2px #00131a); }
    .eclipse-centerline { filter: drop-shadow(0 0 2px #000) drop-shadow(0 0 2px #000); }
    .status { color: #9fc7ff; }
    a { color: #77b7ff; }
    @media (max-width: 620px) { .panel { top: 8px; left: 8px; padding: 12px; } h1 { font-size: 17px; } }
  </style>
</head>
<body>
  <div id="map"></div>
  <section class="panel">
    <h1>Where will the 2026 eclipse be clear?</h1>
    <p>ECMWF IFS total-cloud forecast matched to the local maximum eclipse time in each 16° × 16° processing chunk.</p>
    <select id="mode">
      <option value="cloud">Cloud outlook — green clear, red cloudy</option>
      <option value="eclipse">Maximum solar obscuration</option>
      <option value="clear">Clear-sky fraction</option>
    </select>
    <div id="legend"></div>
    <p class="status" id="status">Loading the Cloud Optimized GeoTIFF…</p>
    <p class="meta" id="meta"></p>
    <p class="meta">Path: <a href="https://eclipse.gsfc.nasa.gov/SEpath/SEpath2001/SE2026Aug12Tpath.html" target="_blank" rel="noopener">NASA/GSFC (Fred Espenak)</a>.</p>
    <p class="meta">Click the eclipse footprint for values. Times are UTC. Eye protection is required outside totality.</p>
  </section>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/georaster"></script>
  <script src="https://unpkg.com/georaster-layer-for-leaflet"></script>
  <script>
    const config = __CONFIG__;
    const map = L.map('map', {worldCopyJump: true, minZoom: 1}).setView([58, -25], 2);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19
    }).addTo(map);
    let georaster, rasterLayer, pathLayer;
    const nodata = 65535;
    const rgba = (r, g, b, a) => `rgba(${r},${g},${b},${a})`;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    function color(values, mode) {
      const [obsc, cloud, clear, , klass] = values;
      if (obsc === nodata || obsc == null) return null;
      const alpha = clamp(0.28 + 0.68 * Math.sqrt(obsc / 10000), 0.28, 0.96);
      if (mode === 'eclipse') {
        const t = clamp(obsc / 10000, 0, 1);
        if (klass === 2) return rgba(100, 224, 255, 0.98);
        return rgba(120 + 135 * t, 45 + 160 * t, 175 - 95 * t, alpha);
      }
      if (mode === 'clear') {
        const t = clamp(clear / 10000, 0, 1);
        return rgba(235 - 190 * t, 80 + 155 * t, 72 + 80 * t, alpha);
      }
      const c = cloud / 100;
      if (c <= 25) return rgba(44, 211, 126, alpha);
      if (c <= 60) return rgba(245, 187, 66, alpha);
      return rgba(239, 78, 82, alpha);
    }
    const legends = {
      cloud: [['#2cd37e','≤25% cloud'],['#f5bb42','25–60% cloud'],['#ef4e52','>60% cloud']],
      eclipse: [['#782daf','Partial'],['#ffe080','Deep partial'],['#64e0ff','Totality grid cells']],
      clear: [['#eb5048','Cloudy'],['#9dac70','Mixed'],['#2deba0','Clear']]
    };
    function drawLegend(mode) {
      document.getElementById('legend').innerHTML = legends[mode].map(([c, t]) =>
        `<div class="legend"><span class="swatch" style="background:${c}"></span>${t}</div>`).join('') +
        '<div class="legend"><span class="swatch" style="height:3px;background:#41ddff"></span>Path of totality</div>' +
        '<div class="legend"><span class="swatch" style="height:0;border-top:2px dashed white"></span>NASA/GSFC centerline</div>';
    }
    function draw(mode) {
      drawLegend(mode);
      if (!georaster) return;
      const colorFunction = values => color(values, mode);
      if (!rasterLayer) {
        rasterLayer = new GeoRasterLayer({
          georaster, opacity: 0.65, resolution: 256,
          pixelValuesToColorFn: colorFunction
        });
        rasterLayer.addTo(map);
      } else {
        rasterLayer.updateColors(colorFunction);
      }
      if (pathLayer) pathLayer.bringToFront();
    }
    function drawPath(path) {
      pathLayer = L.geoJSON(path, {
        style: feature => feature.properties.role === 'centerline'
          ? {color: '#ffffff', weight: 3.5, opacity: 1, dashArray: '8 6', className: 'eclipse-centerline'}
          : {color: '#41ddff', weight: 5, opacity: 1, fillColor: '#41ddff', fillOpacity: 0.10, className: 'eclipse-corridor'},
        onEachFeature: (feature, layer) => layer.bindTooltip(feature.properties.name, {sticky: true})
      }).addTo(map);
      pathLayer.bringToFront();
    }
    function valueAt(band, row, col) {
      const values = georaster.values && georaster.values[band];
      if (!values) return null;
      return Array.isArray(values[row]) || ArrayBuffer.isView(values[row]) ? values[row][col] : values[row * georaster.width + col];
    }
    function hhmm(minutes) {
      if (minutes == null || minutes === nodata) return 'n/a';
      const h = String(Math.floor(minutes / 60)).padStart(2, '0');
      const m = String(minutes % 60).padStart(2, '0');
      return `${h}:${m}`;
    }
    map.on('click', event => {
      if (!georaster || !georaster.values) return;
      const col = Math.floor((event.latlng.lng - georaster.xmin) / georaster.pixelWidth);
      const row = Math.floor((georaster.ymax - event.latlng.lat) / Math.abs(georaster.pixelHeight));
      if (row < 0 || col < 0 || row >= georaster.height || col >= georaster.width) return;
      const obsc = valueAt(0, row, col);
      if (obsc == null || obsc === nodata) return;
      const cloud = valueAt(1, row, col), clear = valueAt(2, row, col);
      const maxTime = valueAt(3, row, col), klass = valueAt(4, row, col);
      const forecastTime = valueAt(5, row, col), altitude = valueAt(6, row, col);
      L.popup().setLatLng(event.latlng).setContent(
        `<b>${klass === 2 ? 'Total' : 'Partial'} eclipse</b><br>` +
        `Maximum obscuration: ${(obsc / 100).toFixed(1)}%<br>` +
        `Cloud cover: ${(cloud / 100).toFixed(0)}%<br>` +
        `Clear-sky fraction: ${(clear / 100).toFixed(0)}%<br>` +
        `Maximum: ${hhmm(maxTime)} UTC<br>` +
        `Forecast matched at: ${hhmm(forecastTime)} UTC<br>` +
        `Sun altitude: ${(altitude / 100 - 90).toFixed(1)}°`
      ).openOn(map);
    });
    document.getElementById('mode').addEventListener('change', event => draw(event.target.value));
    drawLegend('cloud');
    document.getElementById('meta').textContent = `${config.forecast_model} · run ${config.forecast_run} · ${config.resolution} · JPL DE440s`;
    Promise.all([
      parseGeoraster(new URL(config.cog_filename, window.location.href).href),
      fetch(new URL(config.path_filename, window.location.href).href, {cache: 'no-store'}).then(response => {
        if (!response.ok) throw new Error(`Could not load eclipse path: HTTP ${response.status}`);
        return response.json();
      })
    ]).then(([raster, path]) => {
      georaster = raster;
      draw(document.getElementById('mode').value);
      drawPath(path);
      document.getElementById('status').textContent = `${config.partial_pixels.toLocaleString()} partial and ${config.total_pixels.toLocaleString()} totality grid cells mapped.`;
    }).catch(error => {
      console.error(error);
      document.getElementById('status').innerHTML = `Could not render the COG in this browser. <a href="${config.cog_filename}">Download it directly</a>.`;
    });
  </script>
</body>
</html>
"""
    return template.replace("__CONFIG__", config)


class RunEclipseWeather(Task):
    event_date: str
    forecast_run_date: str
    forecast_run_hour: int
    output_prefix: str
    forecast_source: str = "google"
    verify_tls: bool = True

    @staticmethod
    def identifier() -> tuple[str, str]:
        return "tilebox.com/solar-eclipse-weather/RunEclipseWeather", "v1.3"

    def execute(self, context: ExecutionContext) -> None:
        date.fromisoformat(self.event_date)
        date.fromisoformat(self.forecast_run_date)
        prefix = _normalized_prefix(self.output_prefix)
        cube_uri = _cube_uri(prefix)
        context.current_task.display = f"Plan eclipse weather {self.event_date}"
        prepare = context.submit_subtask(
            PrepareForecastCube(
                event_date=self.event_date,
                forecast_run_date=self.forecast_run_date,
                forecast_run_hour=self.forecast_run_hour,
                forecast_source=self.forecast_source,
                verify_tls=self.verify_tls,
                cube_uri=cube_uri,
            ),
            max_retries=2,
        )

        chunk_tasks = []
        for y_start in range(0, OUTPUT_HEIGHT, SPATIAL_CHUNK_SIZE):
            for x_start in range(0, OUTPUT_WIDTH, SPATIAL_CHUNK_SIZE):
                chunk_tasks.append(
                    ComputeVisibilityChunk(
                        event_date=self.event_date,
                        cube_uri=cube_uri,
                        y_start=y_start,
                        y_stop=min(y_start + SPATIAL_CHUNK_SIZE, OUTPUT_HEIGHT),
                        x_start=x_start,
                        x_stop=min(x_start + SPATIAL_CHUNK_SIZE, OUTPUT_WIDTH),
                    )
                )
        context.progress("visibility-chunks").add(len(chunk_tasks))
        futures = [
            context.submit_subtask(task, depends_on=[prepare], max_retries=2)
            for task in chunk_tasks
        ]
        context.submit_subtask(
            PublishEclipseMap(
                event_date=self.event_date,
                cube_uri=cube_uri,
                output_prefix=prefix,
            ),
            depends_on=futures,
            max_retries=2,
        )
        context.logger.info(
            "Submitted eclipse workflow",
            event_date=self.event_date,
            forecast_run=f"{self.forecast_run_date}T{self.forecast_run_hour:02d}:00:00Z",
            chunks=len(chunk_tasks),
            cube_uri=cube_uri,
            output_prefix=prefix,
        )


class PrepareForecastCube(Task):
    event_date: str
    forecast_run_date: str
    forecast_run_hour: int
    forecast_source: str
    verify_tls: bool
    cube_uri: str

    @staticmethod
    def identifier() -> tuple[str, str]:
        return "tilebox.com/solar-eclipse-weather/PrepareForecastCube", "v1.3"

    def execute(self, context: ExecutionContext) -> None:
        event = date.fromisoformat(self.event_date)
        run_date = date.fromisoformat(self.forecast_run_date)
        steps, expected_valid_times = _ecmwf_valid_times(
            run_date, self.forecast_run_hour, event
        )
        forecast_run = f"{run_date.isoformat()}T{self.forecast_run_hour:02d}:00:00Z"
        context.current_task.display = f"Download ECMWF IFS {forecast_run}"
        cache_path = _preferred_cache_dir() / (
            f"ecmwf-ifs-tcc-{run_date:%Y%m%d}-{self.forecast_run_hour:02d}z-"
            f"{'-'.join(map(str, steps))}.grib2"
        )
        if not cache_path.is_file() or cache_path.stat().st_size == 0:
            with context.tracer.span("download-ecmwf-open-data"):
                ECMWFClient(
                    source=self.forecast_source,
                    model="ifs",
                    resol="0p25",
                    preserve_request_order=True,
                    verify=self.verify_tls,
                    maximum_retries=3,
                    retry_after=30,
                ).retrieve(
                    date=run_date.strftime("%Y%m%d"),
                    time=self.forecast_run_hour,
                    type="fc",
                    stream="oper",
                    step=steps,
                    param="tcc",
                    target=str(cache_path),
                )
        with context.tracer.span("decode-ecmwf-grib"):
            cloud_percent, valid_times = _read_forecast_grib(cache_path)
        expected_timestamps = np.asarray(
            [int(value.timestamp()) for value in expected_valid_times],
            dtype=np.int64,
        )
        if not np.array_equal(valid_times, expected_timestamps):
            raise ValueError(
                f"ECMWF valid times {valid_times.tolist()} do not match expected "
                f"{expected_timestamps.tolist()}"
            )
        context.current_task.display = f"Initialize Zarr {cloud_percent.shape}"
        with context.tracer.span("write-forecast-zarr"):
            _initialize_cube(
                cube_uri=self.cube_uri,
                event_date=self.event_date,
                forecast_run=forecast_run,
                valid_times=valid_times,
                cloud_percent=cloud_percent,
            )
        context.logger.info(
            "Prepared ECMWF forecast cube",
            cube_uri=self.cube_uri,
            forecast_run=forecast_run,
            valid_times=[value.isoformat() for value in expected_valid_times],
            shape=list(cloud_percent.shape),
            chunks=[1, SPATIAL_CHUNK_SIZE, SPATIAL_CHUNK_SIZE],
            grib_bytes=cache_path.stat().st_size,
        )


class ComputeVisibilityChunk(Task):
    event_date: str
    cube_uri: str
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int

    @staticmethod
    def identifier() -> tuple[str, str]:
        return "tilebox.com/solar-eclipse-weather/ComputeVisibilityChunk", "v1.3"

    def execute(self, context: ExecutionContext) -> None:
        context.current_task.display = (
            f"Eclipse y{self.y_start}:{self.y_stop} x{self.x_start}:{self.x_stop}"
        )
        ephemeris_path = _ensure_ephemeris()
        with context.tracer.span("calculate-eclipse-and-cloud-match"):
            summary = _write_chunk(
                cube_uri=self.cube_uri,
                event_date=self.event_date,
                y_start=self.y_start,
                y_stop=self.y_stop,
                x_start=self.x_start,
                x_stop=self.x_stop,
                ephemeris_path=ephemeris_path,
            )
        context.progress("visibility-chunks").done(1)
        context.logger.info(
            "Wrote visibility chunk",
            y_start=self.y_start,
            y_stop=self.y_stop,
            x_start=self.x_start,
            x_stop=self.x_stop,
            **summary,
        )


class PublishEclipseMap(Task):
    event_date: str
    cube_uri: str
    output_prefix: str

    @staticmethod
    def identifier() -> tuple[str, str]:
        return "tilebox.com/solar-eclipse-weather/PublishEclipseMap", "v1.3"

    def execute(self, context: ExecutionContext) -> None:
        context.current_task.display = "Publish eclipse COG and web map"
        group = zarr.open_group(
            store=_zarr_store(self.cube_uri, read_only=True), mode="r"
        )
        with tempfile.TemporaryDirectory(
            prefix="eclipse-weather-"
        ) as temporary_directory:
            cog_path = Path(temporary_directory) / "eclipse_visibility.tif"
            with context.tracer.span("write-eclipse-cog"):
                statistics = _write_cog(group, cog_path)
            cog_url = _upload_result(
                self.output_prefix,
                "eclipse_visibility.tif",
                cog_path,
                "image/tiff",
            )

        path_filename = "eclipse_path.geojson"
        path_url = _upload_result(
            self.output_prefix,
            path_filename,
            json.dumps(_eclipse_path_geojson(group), separators=(",", ":")).encode(
                "utf-8"
            ),
            "application/geo+json; charset=utf-8",
        )
        metadata = {
            "event_date": self.event_date,
            "forecast_model": group.attrs["forecast_model"],
            "forecast_run": group.attrs["forecast_run"],
            "forecast_valid_times_utc": group.attrs["forecast_valid_times_utc"],
            "resolution": f"{RESOLUTION_DEGREES}°",
            "cog_filename": "eclipse_visibility.tif",
            "cog_url": cog_url,
            "path_filename": path_filename,
            "path_url": path_url,
            "zarr_uri": self.cube_uri,
            **statistics,
        }
        html = _viewer_html(metadata).encode("utf-8")
        viewer_url = _upload_result(
            self.output_prefix,
            "index.html",
            html,
            "text/html; charset=utf-8",
        )
        manifest_url = _upload_result(
            self.output_prefix,
            "manifest.json",
            json.dumps({**metadata, "viewer_url": viewer_url}, indent=2).encode(
                "utf-8"
            ),
            "application/json; charset=utf-8",
        )
        context.logger.info(
            "Published eclipse weather map",
            viewer_url=viewer_url,
            cog_url=cog_url,
            manifest_url=manifest_url,
            cube_uri=self.cube_uri,
            **statistics,
        )


runner = Runner(
    tasks=[
        RunEclipseWeather,
        PrepareForecastCube,
        ComputeVisibilityChunk,
        PublishEclipseMap,
    ],
    cache=LocalFileSystemCache(_preferred_cache_dir()),
)


if __name__ == "__main__":
    runner.connect_to(Client()).run_forever()
