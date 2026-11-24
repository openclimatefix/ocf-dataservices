from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr
from schemas.dim_order import enforce_dim_order
from schemas.nwp_coordinates import extract_grib_attr, utc_now_coord
from schemas.um_grib import normalize_um_variables, read_um_grib_datasets
from schemas.validation import validates

from .schema import CedaGlobalWestEuropeSchema, CedaUkvSchema

# CEDA UKV 2km OSGB grid coordinates (metres), from https://zenodo.org/record/7357056 (page 4).
# y runs north-to-south (704 points), x west-to-east (548 points).
_UKV_Y_OSGB = np.arange(1223000, -185000, -2000, dtype=np.float64)
_UKV_X_OSGB = np.arange(-239000, 857000, 2000, dtype=np.float64)


def _assign_ukv_osgb_coords(ds: xr.Dataset) -> xr.Dataset:
    """Reshape or reassign CEDA UKV spatial dimensions to `(y_osgb, x_osgb)` in metres."""
    ny, nx = len(_UKV_Y_OSGB), len(_UKV_X_OSGB)
    if "values" in ds.dims:
        if ds.sizes["values"] != ny * nx:
            raise ValueError(
                f"CEDA UKV 1D 'values' size ({ds.sizes['values']}) does not match "
                f"expected OSGB grid ({ny} * {nx} = {ny * nx})"
            )
        new_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for var_name, da in ds.data_vars.items():
            other_dims = [str(d) for d in da.dims if d != "values"]
            transposed = da.transpose(*other_dims, "values")
            reshaped = transposed.values.reshape(*[transposed.sizes[d] for d in other_dims], ny, nx)
            new_vars[str(var_name)] = ((*other_dims, "y_osgb", "x_osgb"), reshaped)

        kept_coords = {
            str(c): ds.coords[c].variable
            for c in ds.coords
            if "values" not in ds.coords[c].dims
            and c not in ("latitude", "longitude", "values")
        }
        kept_coords["y_osgb"] = xr.Variable(("y_osgb",), _UKV_Y_OSGB)
        kept_coords["x_osgb"] = xr.Variable(("x_osgb",), _UKV_X_OSGB)
        return xr.Dataset(data_vars=new_vars, coords=kept_coords, attrs=ds.attrs)

    if ds.sizes.get("y") != ny or ds.sizes.get("x") != nx:
        raise ValueError(
            f"CEDA UKV grid shape (y={ds.sizes.get('y')}, x={ds.sizes.get('x')}) does not match "
            f"expected OSGB grid (y={ny}, x={nx})"
        )
    return (
        ds.assign_coords(x=list(range(nx)), y=list(range(ny)))
        .sortby("y", ascending=False)
        .sortby("x")
        .assign_coords(x=_UKV_X_OSGB, y=_UKV_Y_OSGB)
        .rename({"x": "x_osgb", "y": "y_osgb"})
        .drop_vars(["latitude", "longitude"], errors="ignore")
    )


def _read_raw_grib(grib_path: Path | str) -> list[xr.Dataset]:
    """Read a concatenated CEDA GRIB file into a list of xarray Datasets."""
    return read_um_grib_datasets(
        grib_path,
        read_keys=[
            "generatingProcessIdentifier",
            "tablesVersion",
            "latitudeOfFirstGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
        ],
    )


def _combine_raw_datasets(
    raw: xr.Dataset | Sequence[xr.Dataset],
    is_ukv: bool,
) -> tuple[xr.Dataset, int, int, float, float]:
    """Combine per-tile or per-step-range GRIB hypercubes into a single Dataset."""
    dss = [raw] if isinstance(raw, xr.Dataset) else list(raw)
    if not dss:
        raise ValueError("No datasets provided to _transform.")

    gen_proc = -1
    tables_ver = -1
    first_lat = np.nan
    first_lon = np.nan
    for d in dss:
        if gen_proc == -1:
            gen_proc = int(extract_grib_attr(d, "generatingProcessIdentifier", -1))
        if tables_ver == -1:
            tables_ver = int(extract_grib_attr(d, "tablesVersion", -1))
        if np.isnan(first_lat):
            first_lat = float(extract_grib_attr(d, "latitudeOfFirstGridPointInDegrees", np.nan))
        if np.isnan(first_lon):
            first_lon = float(extract_grib_attr(d, "longitudeOfFirstGridPointInDegrees", np.nan))

    drop_scalar_coords = [
        "number",
        "surface",
        "heightAboveGround",
        "heightAboveGroundHighPrecision",
        "atmosphere",
        "meanSea",
        "valid_time",
    ]
    by_var: dict[str, list[xr.Dataset]] = {}
    for d in dss:
        if "t" in d.data_vars and "heightAboveGround" in d["t"].dims:
            d = d.sel(heightAboveGround=1)
        d = d.drop_vars(drop_scalar_coords, errors="ignore")
        if not is_ukv and "longitude" in d.dims:
            # CEDA AreaA..H tiles are 386 points wide where index -1 overlaps the next tile's index 0
            if d.sizes["longitude"] == 386:
                d = d.isel(longitude=slice(None, -1))
            if "latitude" in d.dims and d.sizes["latitude"] == 577 and float(d.latitude.max()) == 0.0:
                # Southern hemisphere tiles (AreaE..H) share latitude=0.0 with AreaA..D
                d = d.sel(latitude=d.latitude != 0.0)
            if float(d.longitude.max()) > 180:
                d = d.assign_coords(longitude=(((d.longitude + 180) % 360) - 180))
            d = d.sortby("longitude").sortby("latitude", ascending=False)
        for var in d.data_vars:
            by_var.setdefault(str(var), []).append(d[[var]])

    def _concat_dedup_steps(pieces: list[xr.Dataset]) -> xr.Dataset:
        if len(pieces) == 1:
            return pieces[0]
        cat = xr.concat(pieces, dim="step", combine_attrs="override").sortby("step")
        _, unique_idx = np.unique(cat["step"].values, return_index=True)
        return cat.isel(step=np.sort(unique_idx))

    merged_vars: list[xr.Dataset] = []
    for tiles in by_var.values():
        if len(tiles) == 1:
            merged_vars.append(tiles[0])
        elif is_ukv or "latitude" not in tiles[0].dims:
            merged_vars.append(_concat_dedup_steps(tiles))
        else:
            # Group by spatial tile bounds first, deduplicate steps within each tile, then stitch tiles
            by_spatial: dict[tuple[float, float, int, int], list[xr.Dataset]] = {}
            for t_ds in tiles:
                key = (
                    float(t_ds.latitude.values[0]),
                    float(t_ds.longitude.values[0]),
                    int(t_ds.sizes["latitude"]),
                    int(t_ds.sizes["longitude"]),
                )
                by_spatial.setdefault(key, []).append(t_ds)
            spatial_tiles = [_concat_dedup_steps(group) for group in by_spatial.values()]
            if len(spatial_tiles) == 1:
                merged_vars.append(spatial_tiles[0])
            else:
                combined = cast(
                    xr.Dataset,
                    xr.combine_by_coords(spatial_tiles, combine_attrs="override"),
                )
                merged_vars.append(combined)

    full = xr.merge(merged_vars, compat="override", combine_attrs="override")
    return full, gen_proc, tables_ver, first_lat, first_lon


def _transform(
    raw: xr.Dataset | Sequence[xr.Dataset],
    dims: Sequence[str],
    bbox_nwse: Sequence[float] | None = None,
    max_step_hours: int | None = None,
) -> xr.Dataset:
    """Rename, convert units, slice, and reorder dimensions to match the target CEDA schema."""
    is_ukv = "x_osgb" in dims
    ds, gen_proc, tables_ver, first_lat, first_lon = _combine_raw_datasets(raw, is_ukv=is_ukv)

    if bbox_nwse is not None and "latitude" in ds.dims and "longitude" in ds.dims:
        n, w, s, e = bbox_nwse
        lat_slice = (
            slice(s, n)
            if ds.latitude.size > 1 and float(ds.latitude[0]) < float(ds.latitude[-1])
            else slice(n, s)
        )
        ds = ds.sel(latitude=lat_slice, longitude=slice(w, e))
        if ds.sizes.get("latitude", 0) == 0 or ds.sizes.get("longitude", 0) == 0:
            raise ValueError(f"Empty spatial slice for bbox {bbox_nwse}: sizes={dict(ds.sizes)}")

    if "time" in ds.coords:
        ds = ds.rename({"time": "init_time"})
    if "init_time" in ds.coords and "init_time" not in ds.dims:
        ds = ds.expand_dims("init_time")

    ds, schema_vars = normalize_um_variables(ds)

    if is_ukv:
        ds = _assign_ukv_osgb_coords(ds)

    if max_step_hours is not None and "step" in ds.coords:
        ds = ds.sel(step=slice(None, np.timedelta64(max_step_hours, "h")))

    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)

    extra_coords = [c for c in ds.coords if c not in dims]
    if extra_coords:
        ds = ds.drop_vars(extra_coords, errors="ignore")

    n_init = ds.sizes["init_time"]
    meta_coords: dict[str, object] = {
        "generating_process_id": ("init_time", np.full(n_init, gen_proc, dtype=np.int16)),
        "grib_tables_version": ("init_time", np.full(n_init, tables_ver, dtype=np.int16)),
        "ingested_at": utc_now_coord(n_init),
    }
    if is_ukv:
        meta_coords["grid_first_lat"] = ("init_time", np.full(n_init, first_lat, dtype=np.float64))
        meta_coords["grid_first_lon"] = ("init_time", np.full(n_init, first_lon, dtype=np.float64))
    ds = ds.assign_coords(meta_coords)

    return enforce_dim_order(ds, dims, keep_vars=schema_vars)


@validates(CedaGlobalWestEuropeSchema)
def process_ceda_global_west_europe(
    grib_path: Path | str,
    bbox_nwse: Sequence[float] = (63, -12, 35, 26),
    max_step_hours: int = 144,
) -> xr.Dataset:
    """Process CEDA Global 17km GRIB data for West Europe, validated against CedaGlobalWestEuropeSchema."""
    return _transform(
        _read_raw_grib(grib_path),
        dims=CedaGlobalWestEuropeSchema.dims(),
        bbox_nwse=bbox_nwse,
        max_step_hours=max_step_hours,
    )


@validates(CedaUkvSchema)
def process_ceda_ukv(
    grib_path: Path | str,
    max_step_hours: int = 54,
) -> xr.Dataset:
    """Process CEDA UKV 2km GRIB data, validated against CedaUkvSchema."""
    return _transform(
        _read_raw_grib(grib_path),
        dims=CedaUkvSchema.dims(),
        max_step_hours=max_step_hours,
    )
