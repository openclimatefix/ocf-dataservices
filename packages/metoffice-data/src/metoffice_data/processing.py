from collections.abc import Sequence
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr
from schemas.dim_order import enforce_dim_order
from schemas.validation import validates

from .schema import (
    MetOfficeGlobalIndiaSchema,
    MetOfficeGlobalWesteuropeSchema,
    MetOfficeUkvSchema,
)

# Surface-adjusted wind parameters arrive as 'unknown' and must be disambiguated by the GRIB2
# parameterNumber. See https://datahub.metoffice.gov.uk/docs/glossary?sortOrder=GRIB2_CODE.
_PARAMETER_NUMBER_NAMES: dict[int, str] = {
    1: "tcc",
    192: "u10",
    193: "v10",
    194: "wdir",
    195: "si10",
}

_VAR_MAPPING: dict[str, str] = {
    "tcc": "total_cloud_cover_atmosphere",
    "hcc": "high_cloud_cover",
    "mcc": "medium_cloud_cover",
    "lcc": "low_cloud_cover",
    "vis": "visibility",
    "r": "relative_humidity_2m",
    "r2": "relative_humidity_2m",
    "prmsl": "pressure_reduced_to_mean_sea_level",
    "msl": "pressure_reduced_to_mean_sea_level",
    "sd": "snow_depth",
    "dswrf": "downward_short_wave_radiation_flux_surface",
    "sdswrf": "downward_short_wave_radiation_flux_surface",
    "ssrd": "downward_short_wave_radiation_flux_surface",
    "dlwrf": "downward_long_wave_radiation_flux_surface",
    "sdlwrf": "downward_long_wave_radiation_flux_surface",
    "strd": "downward_long_wave_radiation_flux_surface",
    "t2m": "temperature_2m",
    "t": "temperature_2m",
    "u10": "wind_u_10m",
    "10u": "wind_u_10m",
    "v10": "wind_v_10m",
    "10v": "wind_v_10m",
    "si10": "wind_speed_10m",
    "10si": "wind_speed_10m",
    "wdir": "wind_direction_10m",
    "10wdir": "wind_direction_10m",
    "prate": "total_precipitation_rate",
    "lsrr": "total_precipitation_rate",
}

_CLOUD_VARS = (
    "total_cloud_cover_atmosphere",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
)

# The UKV Lambert Azimuthal Equal Area grid (metres). cfgrib does not read the projected
# coordinate values reliably, so they are reassigned from these known arrays (as read via
# iris-grib), matched by size. y runs north-to-south (descending), x west-to-east (ascending).
_UKV_Y_LAEA = np.arange(700000, -576000 - 2000, -2000)
_UKV_X_LAEA = np.arange(-576000, 332000 + 2000, 2000)


def _assign_ukv_coords(ds: xr.Dataset) -> xr.Dataset:
    """Reassign the UKV horizontal coordinates to their known LAEA values (in metres)."""
    if ds.sizes.get("x") != len(_UKV_X_LAEA) or ds.sizes.get("y") != len(_UKV_Y_LAEA):
        raise ValueError(
            f"UKV grid shape (y={ds.sizes.get('y')}, x={ds.sizes.get('x')}) does not match the "
            f"expected LAEA grid (y={len(_UKV_Y_LAEA)}, x={len(_UKV_X_LAEA)})"
        )
    return (
        ds.assign_coords(x=list(range(ds.sizes["x"])), y=list(range(ds.sizes["y"])))
        .sortby("y", ascending=False)
        .sortby("x")
        .assign_coords(x=_UKV_X_LAEA, y=_UKV_Y_LAEA)
        .rename({"x": "x_laea", "y": "y_laea"})
        .drop_vars(["latitude", "longitude"], errors="ignore")
    )


def _read_raw_grib(grib_path: Path | str) -> xr.Dataset:
    """Read a concatenated MetOffice DataHub GRIB file into one Xarray Dataset.

    For some reason, wind and cloud parameters all are read with shortName='unknown' by cfgrib,
    which causes them to overwrite each other if read together. This function extracts them
    separately using filter_by_keys on their parameterNumber, before reading the remaining
    variables and merging everything.
    """
    cleaned: list[xr.Dataset] = []

    # Handle the "unknown" variables.
    for param_num, target_name in _PARAMETER_NUMBER_NAMES.items():
        try:
            ds_param = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {"parameterNumber": param_num},
                },
            )
            if "unknown" in ds_param.data_vars:
                ds_param = ds_param.rename({"unknown": target_name})
                cleaned.append(
                    ds_param.drop_vars(
                        ["number", "surface", "heightAboveGround", "atmosphere", "meanSea"],
                        errors="ignore",
                    )
                )
        except Exception:  # noqa: S110, BLE001
            pass  # Variable not present in this file

    # Then read the rest of the variables.
    try:
        datasets = cfgrib.open_datasets(
            grib_path,
            backend_kwargs={"indexpath": ""},
        )
    except Exception as e:
        raise ValueError(f"Failed to open datasets from {grib_path}: {e}") from e

    for ds in datasets:
        # Already handled 'unknown', so any left over are unwanted.
        if "unknown" in ds.data_vars:
            ds = ds.drop_vars("unknown")

        if len(ds.data_vars) > 0:
            cleaned.append(
                ds.drop_vars(
                    ["number", "surface", "heightAboveGround", "atmosphere", "meanSea"],
                    errors="ignore",
                )
            )

    if not cleaned:
        raise ValueError(f"No valid datasets found in {grib_path}")

    return xr.merge(cleaned, compat="override", combine_attrs="override")


def _transform(ds: xr.Dataset, dims: Sequence[str]) -> xr.Dataset:
    """Rename, convert units, and reorder dimensions to match the target schema `dims`."""
    # Global grids may arrive on a 0-360 longitude; the schemas use -180 to 180. The UKV grid
    # stays on its native LAEA projection, so this only applies where longitude is a dimension.
    if "longitude" in ds.dims and ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")

    if "time" in ds.coords:
        ds = ds.rename({"time": "init_time"})
    if "init_time" in ds.coords and "init_time" not in ds.dims:
        ds = ds.expand_dims("init_time")

    rename_vars = {k: v for k, v in _VAR_MAPPING.items() if k in ds.data_vars}
    ds = ds.rename(rename_vars)

    if "temperature_2m" in ds.data_vars:
        ds["temperature_2m"] = ds["temperature_2m"] - 273.15

    if "x_laea" in dims:
        ds = _assign_ukv_coords(ds)

    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)

    extra_coords = [c for c in ds.coords if c not in dims]
    if extra_coords:
        ds = ds.drop_vars(extra_coords, errors="ignore")

    print(list(ds.data_vars))

    schema_vars = list(set(_VAR_MAPPING.values()))
    return enforce_dim_order(ds, dims, keep_vars=schema_vars)


def process_metoffice(grib_path: Path | str, dims: Sequence[str]) -> xr.Dataset:
    """Read and transform a MetOffice DataHub GRIB file, without validation.

    Callers should use one of the validated entrypoints (`process_metoffice_westeurope`,
    `process_metoffice_india`, `process_metoffice_ukv`), which wrap this and validate the result.
    """
    return _transform(_read_raw_grib(grib_path), dims=dims)


@validates(MetOfficeGlobalWesteuropeSchema)
def process_metoffice_westeurope(grib_path: Path | str) -> xr.Dataset:
    """Process the westeurope global order, validated against MetOfficeGlobalWesteuropeSchema."""
    return process_metoffice(grib_path, dims=MetOfficeGlobalWesteuropeSchema.dims())


@validates(MetOfficeGlobalIndiaSchema)
def process_metoffice_india(grib_path: Path | str) -> xr.Dataset:
    """Process the india global order, validated against MetOfficeGlobalIndiaSchema."""
    return process_metoffice(grib_path, dims=MetOfficeGlobalIndiaSchema.dims())


@validates(MetOfficeUkvSchema)
def process_metoffice_ukv(grib_path: Path | str) -> xr.Dataset:
    """Process the UKV order, validated against MetOfficeUkvSchema."""
    return process_metoffice(grib_path, dims=MetOfficeUkvSchema.dims())
