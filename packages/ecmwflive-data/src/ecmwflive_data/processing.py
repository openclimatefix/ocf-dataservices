from pathlib import Path

import cfgrib
import xarray as xr
from schemas.ecmwf_hres import transform_ecmwf_hres
from schemas.validation import validates

from .schema import EcmwfLiveSchema


def _read_raw_grib(grib_path: Path | str, bbox_nwse: list[float]) -> xr.Dataset:
    """Read raw ECMWF Live GRIB data into an Xarray Dataset with its original variable/coord names."""
    # Open using cfgrib with ignore_keys to prevent DatasetBuildError
    # from conflicting vertical levels and stepTypes.
    # The GRIB file contains different resolutions (numberOfPoints), so we use cfgrib.open_datasets
    # to open all hypercubes, then filter for the one we want.
    try:
        datasets = cfgrib.open_datasets(
            grib_path,
            backend_kwargs={
                "indexpath": "",
                "read_keys": ["generatingProcessIdentifier", "expver", "tablesVersion"],
                "ignore_keys": [
                    "valid_time",
                    "stepType",
                    "typeOfLevel",
                    "surface",
                    "heightAboveGround",
                    "meanSea",
                ],
            },
        )
    except Exception as e:
        raise ValueError(f"Failed to open datasets from {grib_path}: {e}") from e

    if not datasets:
        raise ValueError(f"No datasets found in {grib_path}")

    # The GRIB files contain fields on different grids (global vs subsets).
    # We want to merge fields that share the same lat/lon grid.
    # A safe heuristic is to merge everything on the largest grid or keep them separate.
    # Since ECMWF Live contains subsets with different `numberOfPoints`, we filter the datasets
    # to find the ones with both `latitude` and `longitude` coords, and merge them.
    valid_datasets = [
        d
        for d in datasets
        if "latitude" in d.coords
        and "longitude" in d.coords
        and d.latitude.ndim == 1
        and d.longitude.ndim == 1
    ]

    if not valid_datasets:
        raise ValueError(f"No regular lat/lon grid datasets found in {grib_path}")

    # Wrap longitudes into [-180, 180), orient latitudes north-to-south, crop to bbox_nwse, and
    # group by spatial grid so mixed global/regional or multi-resolution GRIB cubes align cleanly.
    n, w, s, e = bbox_nwse
    by_grid: dict[tuple[int, int, float, float], list[xr.Dataset]] = {}
    for d in valid_datasets:
        if float(d.longitude.max()) > 180:
            d = d.assign_coords(longitude=(((d.longitude + 180) % 360) - 180)).sortby("longitude")
        if d.latitude.size > 1 and float(d.latitude[0]) < float(d.latitude[-1]):
            d = d.sortby("latitude", ascending=False)

        d_sub = d.sel(latitude=slice(n, s), longitude=slice(w, e))
        if d_sub.sizes.get("latitude", 0) == 0 or d_sub.sizes.get("longitude", 0) == 0:
            continue

        grid_key = (
            int(d_sub.sizes["latitude"]),
            int(d_sub.sizes["longitude"]),
            round(float(d_sub.latitude[0]), 4),
            round(float(d_sub.longitude[0]), 4),
        )
        by_grid.setdefault(grid_key, []).append(d_sub)

    if not by_grid:
        raise ValueError(f"No grid datasets overlap the target bbox {bbox_nwse} in {grib_path}")

    best_group = max(
        by_grid.values(),
        key=lambda grp: (
            grp[0].sizes["latitude"] * grp[0].sizes["longitude"],
            sum(len(x.data_vars) for x in grp),
        ),
    )
    ds = xr.merge(best_group, join="exact", compat="override", combine_attrs="override")

    # The schemas do not expect extra scalar coordinates that cfgrib automatically adds
    # (like 'number', 'level', etc.), but we strip them globally in _transform rather than here.
    return ds


def process_ecmwf_live(
    grib_path: Path | str,
    bbox_nwse: list[float],
    max_step_hours: int,
) -> xr.Dataset:
    """Process raw ECMWF live GRIB data into an unvalidated `xarray.Dataset`.

    Use :func:`process_ecmwf_live_validated` to validate the output against :class:`EcmwfLiveSchema`.
    """
    ds = _read_raw_grib(grib_path, bbox_nwse=bbox_nwse)
    return transform_ecmwf_hres(ds, bbox_nwse=bbox_nwse, max_step_hours=max_step_hours)


@validates(EcmwfLiveSchema)
def process_ecmwf_live_validated(
    grib_path: Path | str,
    bbox_nwse: list[float],
    max_step_hours: int,
) -> xr.Dataset:
    """Process raw ECMWF Live GRIB data for a region, validated against EcmwfLiveSchema."""
    return process_ecmwf_live(
        grib_path=grib_path, bbox_nwse=bbox_nwse, max_step_hours=max_step_hours
    )
