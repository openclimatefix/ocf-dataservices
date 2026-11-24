from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr
from schemas.dim_order import enforce_dim_order
from schemas.nwp_coordinates import extract_grib_attr, utc_now_coord
from schemas.um_grib import normalize_um_variables, read_um_grib_datasets
from schemas.validation import validates

from .schema import (
    MetOfficeGlobalIndiaSchema,
    MetOfficeGlobalWesteuropeSchema,
    MetOfficeUkvSchema,
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
    """Read a concatenated MetOffice DataHub GRIB file into an `xarray.Dataset`."""
    raw_datasets = read_um_grib_datasets(
        grib_path,
        read_keys=[
            "generatingProcessIdentifier",
            "localTablesVersion",
            "latitudeOfFirstGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
        ],
    )
    cleaned: list[xr.Dataset] = []
    for ds in raw_datasets:
        ds_clean = ds.drop_vars(
            ["number", "surface", "heightAboveGround", "atmosphere", "meanSea"],
            errors="ignore",
        )
        if "step" in ds_clean.coords and "step" not in ds_clean.dims:
            ds_clean = ds_clean.expand_dims("step")
        cleaned.append(ds_clean)

    return xr.merge(cleaned, join="exact", compat="override", combine_attrs="override")


def _transform(ds: xr.Dataset, dims: Sequence[str]) -> xr.Dataset:
    """Rename, convert units, and reorder dimensions to match the target schema `dims`."""
    gen_proc = int(extract_grib_attr(ds, "generatingProcessIdentifier", -1))
    local_tbl = int(extract_grib_attr(ds, "localTablesVersion", -1))
    first_lat = float(extract_grib_attr(ds, "latitudeOfFirstGridPointInDegrees", np.nan))
    first_lon = float(extract_grib_attr(ds, "longitudeOfFirstGridPointInDegrees", np.nan))

    # Global grids may arrive on a 0-360 longitude; the schemas use -180 to 180. The UKV grid
    # stays on its native LAEA projection, so this only applies where longitude is a dimension.
    if "longitude" in ds.dims and ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")

    if "time" in ds.coords:
        ds = ds.rename({"time": "init_time"})
    if "init_time" in ds.coords and "init_time" not in ds.dims:
        ds = ds.expand_dims("init_time")

    ds, schema_vars = normalize_um_variables(ds)

    if "x_laea" in dims:
        ds = _assign_ukv_coords(ds)

    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)

    extra_coords = [c for c in ds.coords if c not in dims]
    if extra_coords:
        ds = ds.drop_vars(extra_coords, errors="ignore")

    n_init = ds.sizes["init_time"]
    meta_coords: dict[str, object] = {
        "generating_process_id": ("init_time", np.full(n_init, gen_proc, dtype=np.int16)),
        "local_tables_version": ("init_time", np.full(n_init, local_tbl, dtype=np.int16)),
        "ingested_at": utc_now_coord(n_init),
    }
    if "x_laea" in dims:
        meta_coords["grid_first_lat"] = ("init_time", np.full(n_init, first_lat, dtype=np.float64))
        meta_coords["grid_first_lon"] = ("init_time", np.full(n_init, first_lon, dtype=np.float64))
    ds = ds.assign_coords(meta_coords)

    return enforce_dim_order(ds, dims, keep_vars=schema_vars)


def process_metoffice(grib_path: Path | str, dims: Sequence[str]) -> xr.Dataset:
    """Read and transform a MetOffice DataHub GRIB file without schema validation.

    Use :func:`process_metoffice_westeurope`, :func:`process_metoffice_india`, or
    :func:`process_metoffice_ukv` to validate the returned dataset.
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
