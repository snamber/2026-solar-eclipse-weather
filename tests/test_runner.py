from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import zarr

import runner as runner_module
from runner import (
    NODATA_UINT16,
    _circle_overlap_fraction,
    _eclipse_path_geojson,
    _forecast_bracket,
    _grid_coordinates,
    _viewer_html,
    _write_cog,
)


def test_circle_overlap_handles_none_partial_and_total() -> None:
    separation = np.asarray([3.0, 1.0, 0.0])
    sun_radius = np.ones(3)
    moon_radius = np.ones(3) * 1.1

    overlap = _circle_overlap_fraction(separation, sun_radius, moon_radius)

    assert overlap[0] == 0.0
    assert 0.0 < overlap[1] < 1.0
    assert overlap[2] == 1.0


def test_forecast_bracket_interpolates_and_clamps() -> None:
    valid = np.asarray([100, 200, 300])

    assert _forecast_bracket(valid, 50) == (0, 0, 0.0)
    assert _forecast_bracket(valid, 150) == (0, 1, 0.5)
    assert _forecast_bracket(valid, 300) == (1, 2, 1.0)
    assert _forecast_bracket(valid, 350) == (2, 2, 0.0)


def test_output_grid_has_quarter_degree_cell_centers() -> None:
    latitudes, longitudes = _grid_coordinates(0, 2, 0, 2)

    np.testing.assert_allclose(latitudes[:, 0], [89.875, 89.625])
    np.testing.assert_allclose(longitudes[0, :], [-179.875, -179.625])


def test_cog_writer_preserves_bands_nodata_and_layout(tmp_path: Path) -> None:
    group = zarr.group(store=str(tmp_path / "cube.zarr"), overwrite=True)
    group.attrs.update(
        {
            "event_date": "2026-08-12",
            "forecast_model": "ECMWF IFS 0.25 degree open data",
            "forecast_run": "2026-08-11T00:00:00Z",
            "forecast_license": "CC BY 4.0",
            "ephemeris": "JPL DE440s",
            "transform": [0.25, 0.0, -1.0, 0.0, -0.25, 1.0],
        }
    )
    arrays = {
        "eclipse_obscuration_hundredths_percent": np.asarray(
            [[10000, 5000], [0, 0]], dtype=np.uint16
        ),
        "matched_cloud_cover_hundredths_percent": np.asarray(
            [[1000, 8000], [0, 0]], dtype=np.uint16
        ),
        "clear_sky_hundredths_percent": np.asarray(
            [[9000, 2000], [0, 0]], dtype=np.uint16
        ),
        "max_eclipse_utc_second": np.asarray([[64800, 64800], [0, 0]], dtype=np.uint32),
        "forecast_effective_utc_second": np.asarray(
            [[64800, 64800], [0, 0]], dtype=np.uint32
        ),
        "sun_altitude_hundredths_degree": np.asarray(
            [[1000, 500], [0, 0]], dtype=np.int16
        ),
        "eclipse_class": np.asarray([[2, 1], [0, 0]], dtype=np.uint8),
    }
    for name, values in arrays.items():
        group.create_array(name, data=values)

    path = tmp_path / "eclipse.tif"
    statistics = _write_cog(group, path)

    assert statistics["total_pixels"] == 1
    assert statistics["partial_pixels"] == 1
    with rasterio.open(path) as dataset:
        assert dataset.count == 7
        assert dataset.nodata == int(NODATA_UINT16)
        image_structure = dataset.tags(ns="IMAGE_STRUCTURE")
        assert image_structure["LAYOUT"] == "COG"
        assert image_structure["COMPRESSION"] == "DEFLATE"
        assert dataset.read(1)[1, 0] == NODATA_UINT16
        assert dataset.descriptions[4] == "Eclipse class (1 partial, 2 total)"


def test_eclipse_path_contains_corridor_and_centerline(tmp_path: Path) -> None:
    group = zarr.group(store=str(tmp_path / "path.zarr"), overwrite=True)
    group.attrs["transform"] = [0.25, 0.0, -1.0, 0.0, -0.25, 1.0]
    group.create_array(
        "eclipse_class",
        data=np.asarray(
            [
                [0, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 0, 2, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    group.create_array(
        "max_eclipse_utc_second",
        data=np.asarray(
            [
                [0, 0, 0, 0],
                [0, 61200, 61200, 0],
                [0, 0, 61320, 61320],
                [0, 0, 0, 0],
            ],
            dtype=np.uint32,
        ),
    )

    geojson = _eclipse_path_geojson(group)

    roles = {feature["properties"]["role"] for feature in geojson["features"]}
    assert roles == {"totality-corridor", "centerline"}
    centerline = next(
        feature
        for feature in geojson["features"]
        if feature["properties"]["role"] == "centerline"
    )
    assert centerline["geometry"]["type"] == "LineString"
    assert len(centerline["geometry"]["coordinates"]) == 2


def test_2026_eclipse_path_uses_nasa_sunset_limit(tmp_path: Path) -> None:
    group = zarr.group(store=str(tmp_path / "nasa-path.zarr"), overwrite=True)
    group.attrs["event_date"] = "2026-08-12"

    geojson = _eclipse_path_geojson(group)

    boundaries = [
        feature
        for feature in geojson["features"]
        if feature["properties"]["role"] == "totality-corridor"
    ]
    centerline = next(
        feature
        for feature in geojson["features"]
        if feature["properties"]["role"] == "centerline"
    )
    assert {feature["properties"]["boundary"] for feature in boundaries} == {
        "northern",
        "southern",
    }
    assert all(
        feature["properties"]["source"].endswith("NASA/GSFC") for feature in boundaries
    )
    northern = next(
        feature
        for feature in boundaries
        if feature["properties"]["boundary"] == "northern"
    )
    southern = next(
        feature
        for feature in boundaries
        if feature["properties"]["boundary"] == "southern"
    )
    assert northern["geometry"]["coordinates"][-1][-1] == [6.34, 39.7083]
    assert southern["geometry"]["coordinates"][-1][-1] == [4.54, 37.69]
    assert centerline["geometry"]["type"] == "MultiLineString"
    assert centerline["geometry"]["coordinates"][-1][-3:] == [
        [-3.185, 41.8167],
        [2.95, 39.4083],
        [5.415, 38.68],
    ]


def test_viewer_references_relative_cog() -> None:
    html = _viewer_html(
        {
            "forecast_model": "ECMWF IFS",
            "forecast_run": "2026-08-11T00:00:00Z",
            "resolution": "0.25°",
            "cog_filename": "eclipse_visibility.tif",
            "path_filename": "eclipse_path.geojson",
            "partial_pixels": 10,
            "total_pixels": 2,
        }
    )

    assert "parseGeoraster" in html
    assert '"cog_filename":"eclipse_visibility.tif"' in html
    assert '"path_filename":"eclipse_path.geojson"' in html
    assert "{cache: 'no-store'}" in html
    assert "rasterLayer.updateColors" in html
    assert "georaster, opacity: 0.65" in html
    assert "Path of totality" in html


def test_results_default_to_local_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "RESULTS_URI", str(tmp_path))
    runner_module._results_store.cache_clear()

    result_url = runner_module._upload_result(
        "eclipse/run", "manifest.json", b"{}", "application/json"
    )

    result_path = tmp_path / "eclipse/run/manifest.json"
    assert result_path.read_bytes() == b"{}"
    assert result_url == result_path.as_uri()
    assert runner_module._cube_uri("eclipse/run") == str(
        tmp_path / "eclipse/run/eclipse_weather.zarr"
    )
