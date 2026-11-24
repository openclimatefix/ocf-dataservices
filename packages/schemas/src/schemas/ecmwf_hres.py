from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import xarray as xr
from pandera.typing.xarray import Coordinate

from .base import NwpDatasetSchema
from .dim_order import enforce_dim_order
from .nwp_coordinates import (
    extract_grib_attr,
    init_time_coord,
    latitude,
    longitude,
    parse_expver_int16,
    step,
    utc_now_coord,
)
from .nwp_variables import (
    direct_solar_radiation,
    downward_long_wave_radiation_flux_surface,
    downward_short_wave_radiation_flux_surface,
    high_cloud_cover,
    low_cloud_cover,
    medium_cloud_cover,
    scale_to_percent,
    snow_depth,
    temperature_2m,
    total_cloud_cover_atmosphere,
    total_precipitation_rate,
    uv_b_radiation,
    visibility,
    wind_u_10m,
    wind_u_100m,
    wind_u_200m,
    wind_v_10m,
    wind_v_100m,
    wind_v_200m,
)


class EcmwfHresSchema(NwpDatasetSchema):
    """Shared schema for 18-variable ECMWF HRES datasets (live S3 and MARS archive)."""

    _chunks: ClassVar[dict[str, int]] = {"init_time": 1, "step": 1, "latitude": -1, "longitude": -1}
    _shards: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": -1,
        "latitude": -1,
        "longitude": -1,
    }

    _dims = ("init_time", "step", "latitude", "longitude")

    # Dimensions
    init_time: Coordinate[np.datetime64] = init_time_coord()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=84)
    longitude: Coordinate[np.float64] = longitude(ge=-180, le=180)
    latitude: Coordinate[np.float64] = latitude(ge=-90, le=90)

    # Auxiliary metadata coordinates on init_time
    generating_process_id: Coordinate[np.int16] = init_time_coord()
    expver: Coordinate[np.int16] = init_time_coord()
    grib_tables_version: Coordinate[np.int16] = init_time_coord()
    ingested_at: Coordinate[np.datetime64] = init_time_coord()

    # Variables (18 parameters)
    wind_u_10m: np.float32 = wind_u_10m(dims=_dims, nullable=True)
    wind_v_10m: np.float32 = wind_v_10m(dims=_dims, nullable=True)
    wind_u_100m: np.float32 = wind_u_100m(dims=_dims, nullable=False)
    wind_v_100m: np.float32 = wind_v_100m(dims=_dims, nullable=False)
    temperature_2m: np.float32 = temperature_2m(dims=_dims, nullable=False)
    wind_u_200m: np.float32 = wind_u_200m(dims=_dims, nullable=False)
    wind_v_200m: np.float32 = wind_v_200m(dims=_dims, nullable=False)
    direct_solar_radiation: np.float32 = direct_solar_radiation(dims=_dims, nullable=True)
    uv_b_radiation: np.float32 = uv_b_radiation(dims=_dims, nullable=True)
    high_cloud_cover: np.float32 = high_cloud_cover(dims=_dims, nullable=False)
    low_cloud_cover: np.float32 = low_cloud_cover(dims=_dims, nullable=False)
    medium_cloud_cover: np.float32 = medium_cloud_cover(dims=_dims, nullable=False)
    total_precipitation_rate: np.float32 = total_precipitation_rate(dims=_dims, nullable=True)
    snow_depth: np.float32 = snow_depth(dims=_dims, nullable=True)
    downward_long_wave_radiation_flux_surface: np.float32 = (
        downward_long_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    downward_short_wave_radiation_flux_surface: np.float32 = (
        downward_short_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    total_cloud_cover_atmosphere: np.float32 = total_cloud_cover_atmosphere(
        dims=_dims, nullable=False
    )
    visibility: np.float32 = visibility(dims=_dims, nullable=True)


_HRES_VAR_MAPPING: dict[str, str] = {
    "10u": "wind_u_10m",
    "u10": "wind_u_10m",
    "10v": "wind_v_10m",
    "v10": "wind_v_10m",
    "100u": "wind_u_100m",
    "u100": "wind_u_100m",
    "100v": "wind_v_100m",
    "v100": "wind_v_100m",
    "t2m": "temperature_2m",
    "2t": "temperature_2m",
    "200u": "wind_u_200m",
    "u200": "wind_u_200m",
    "200v": "wind_v_200m",
    "v200": "wind_v_200m",
    "dsrp": "direct_solar_radiation",
    "uvb": "uv_b_radiation",
    "hcc": "high_cloud_cover",
    "lcc": "low_cloud_cover",
    "mcc": "medium_cloud_cover",
    "tprate": "total_precipitation_rate",
    "sd": "snow_depth",
    "strd": "downward_long_wave_radiation_flux_surface",
    "ssrd": "downward_short_wave_radiation_flux_surface",
    "tcc": "total_cloud_cover_atmosphere",
    "vis": "visibility",
}


def transform_ecmwf_hres(
    ds: xr.Dataset,
    bbox_nwse: Sequence[float],
    max_step_hours: int = 84,
) -> xr.Dataset:
    """Slice to region/step, rename, convert units, de-accumulate radiation, and order dims."""
    cycle_val = int(extract_grib_attr(ds, "generatingProcessIdentifier", -1))
    expver_val = parse_expver_int16(extract_grib_attr(ds, "expver", -1))
    tables_val = int(extract_grib_attr(ds, "tablesVersion", -1))

    if "longitude" in ds.coords and float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")

    n, w, s, e = bbox_nwse
    if "latitude" in ds.coords and "longitude" in ds.coords:
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

    rename_vars = {k: v for k, v in _HRES_VAR_MAPPING.items() if k in ds.data_vars}
    ds = ds.rename(rename_vars)

    if "temperature_2m" in ds.data_vars:
        ds["temperature_2m"] = ds["temperature_2m"] - 273.15

    for cloud_var in (
        "high_cloud_cover",
        "medium_cloud_cover",
        "low_cloud_cover",
        "total_cloud_cover_atmosphere",
    ):
        if cloud_var in ds.data_vars:
            ds[cloud_var] = scale_to_percent(ds[cloud_var])

    rad_vars = [
        v
        for v in (
            "downward_short_wave_radiation_flux_surface",
            "downward_long_wave_radiation_flux_surface",
            "direct_solar_radiation",
            "uv_b_radiation",
        )
        if v in ds.data_vars
    ]
    if rad_vars:
        step_dt = (ds.step.shift(step=-1) - ds.step).dt.total_seconds()
        for rad_var in rad_vars:
            diff_var = ds[rad_var].shift(step=-1) - ds[rad_var]
            ds[rad_var] = (diff_var / step_dt).clip(min=0.0)
        ds = ds.isel(step=slice(0, -1))

    if "step" in ds.coords:
        ds = ds.sel(step=slice(None, np.timedelta64(max_step_hours, "h")))

    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)

    ordered_dims = EcmwfHresSchema.dims()
    extra_coords = [c for c in ds.coords if c not in ordered_dims]
    if extra_coords:
        ds = ds.drop_vars(extra_coords, errors="ignore")

    n_init = ds.sizes["init_time"]
    ds = ds.assign_coords(
        generating_process_id=("init_time", np.full(n_init, cycle_val, dtype=np.int16)),
        expver=("init_time", np.full(n_init, expver_val, dtype=np.int16)),
        grib_tables_version=("init_time", np.full(n_init, tables_val, dtype=np.int16)),
        ingested_at=utc_now_coord(n_init),
    )

    schema_vars = list(set(_HRES_VAR_MAPPING.values()))
    return enforce_dim_order(ds, ordered_dims, keep_vars=schema_vars)
