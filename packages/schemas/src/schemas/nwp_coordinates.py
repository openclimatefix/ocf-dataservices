import datetime as dt
from typing import Any

import numpy as np
import pandera.xarray as pa
import xarray as xr


def init_time_coord() -> Any:
    """Coordinate indexed by `('init_time',)` (used for `init_time` and run-level metadata coords)."""
    return pa.Field(dims=("init_time",), nullable=False)


def step(ge_hours: int = 0, le_hours: int = 85) -> Any:
    return pa.Field(
        dims=("step",),
        nullable=False,
        ge=np.timedelta64(ge_hours, "h"),
        le=np.timedelta64(le_hours, "h"),
    )


def ensemble_member(ge: int = 1, le: int = 50) -> Any:
    return pa.Field(dims=("ensemble_member",), nullable=False, ge=ge, le=le)


def longitude(ge: float = -180, le: float = 180) -> Any:
    return pa.Field(dims=("longitude",), nullable=False, ge=ge, le=le)


def latitude(ge: float = -90, le: float = 90) -> Any:
    return pa.Field(dims=("latitude",), nullable=False, ge=ge, le=le)


def x_laea(ge: float = -2e6, le: float = 2e6) -> Any:
    return pa.Field(dims=("x_laea",), nullable=False, ge=ge, le=le)


def y_laea(ge: float = -2e6, le: float = 2e6) -> Any:
    return pa.Field(dims=("y_laea",), nullable=False, ge=ge, le=le)


def grid_first_lat() -> Any:
    """Latitude of first grid point in degrees from the raw GRIB grid definition."""
    return pa.Field(dims=("init_time",), nullable=True, ge=-90, le=90)


def grid_first_lon() -> Any:
    """Longitude of first grid point in degrees from the raw GRIB grid definition."""
    return pa.Field(dims=("init_time",), nullable=True, ge=-180, le=360)


def extract_grib_attr(ds: xr.Dataset, key: str, default: Any) -> Any:
    """Extract a GRIB header attribute (`GRIB_<key>` or `<key>`) from `ds` or its data_vars."""
    for candidate in (f"GRIB_{key}", key):
        if candidate in ds.attrs and ds.attrs[candidate] not in (None, "undef"):
            return ds.attrs[candidate]
        for da in ds.data_vars.values():
            if candidate in da.attrs and da.attrs[candidate] not in (None, "undef"):
                return da.attrs[candidate]
    return default


def parse_expver_int16(raw: Any, default: int = -1) -> int:
    """Convert an ECMWF `expver` value (e.g. `'0001'` -> `1`, `'h9be'` -> `-1`) to int16."""
    if raw is None or raw == "undef":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def utc_now_coord(n_init: int) -> tuple[tuple[str], np.ndarray]:
    """Build an `('init_time',)` coordinate array stamped with the current UTC time."""
    now_ns = np.datetime64(dt.datetime.now(dt.UTC).replace(tzinfo=None), "ns")
    return (("init_time",), np.full(n_init, now_ns, dtype="datetime64[ns]"))
