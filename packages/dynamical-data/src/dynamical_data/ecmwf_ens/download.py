import concurrent.futures
from datetime import datetime, timezone
from typing import Final, Literal, cast

import dynamical_catalog
import numpy as np
import xarray as xr


class NwpRunNotYetAvailable(Exception):
    """Raised when ``nwp_init_time`` isn't in the catalog yet (Dynamical hasn't published it)."""


_ECMWF_ENS_VARS_TO_DOWNLOAD: Final[tuple[str, ...]] = (
    "temperature_2m",
    "dew_point_temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
    "total_cloud_cover_atmosphere",
    "pressure_reduced_to_mean_sea_level",
    "downward_long_wave_radiation_flux_surface",
    "downward_short_wave_radiation_flux_surface",
    "precipitation_surface",
    "categorical_precipitation_type_surface",
)


def open_it(
    nwp_init_time: datetime,
    bbox_nwse: tuple[float, float, float, float] = (90, 180, -90, -180)
) -> xr.Dataset:
    """Lazily open the ECMWF ENS Icechunk store for a given init time.

    No data is downloaded: the returned dataset is still backed by lazy Dask/Zarr arrays.
    Call :func:`download_ecmwf_ens_data` to actually fetch the data.

    Args:
        nwp_init_time: The initialization time to open. Must be timezone aware.
        bbox_nwse: Optional bounding box to slice the dataset.
            Format is (max_lat, max_lon, min_lat, min_lon).
            If None, the entire dataset is returned.
    """
    if nwp_init_time.utcoffset() is None:
        raise ValueError(f"nwp_init_time must be timezone aware. {nwp_init_time.tzinfo=}")

    # We need to make nwp_init_time tz-naive for the xarray selection.
    utc_nwp_init_time = np.datetime64(nwp_init_time.astimezone(timezone.utc).replace(tzinfo=None))

    ds = dynamical_catalog.open("ecmwf-ifs-ens-forecast-15-day-0-25-degree", chunks=None)

    # Cast to xr.Dataset to satisfy the type checker, as indexing with a list is misidentified as
    # returning a DataArray.
    ds = cast(xr.Dataset, ds[list(_ECMWF_ENS_VARS_TO_DOWNLOAD)])

    if utc_nwp_init_time not in ds.init_time.values:
        raise NwpRunNotYetAvailable(f"{utc_nwp_init_time} is not in ds.init_time.values")

    # Check for empty coordinates before computing bounds to fail gracefully.
    if ds.longitude.size == 0 or ds.latitude.size == 0:
        raise ValueError("Dataset has empty longitude or latitude coordinates.")

    # Validate longitude range.
    # NOTE: Dynamical.org converts the longitude range to [-180, 180].
    if ds.longitude.min() < -180 or ds.longitude.max() > 180:
        raise ValueError("Dataset longitude must be in the range [-180, 180]")

    lat_slice = _calc_slice_for_lat_or_lng("latitude", ds, bbox_nwse)
    lon_slice = _calc_slice_for_lat_or_lng("longitude", ds, bbox_nwse)

    # NOTE: This will fail if the region crosses the anti-meridian. But we do not anticipate
    # forecasting near the anti-meridian.
    ds_sliced = ds.sel(latitude=lat_slice, longitude=lon_slice, init_time=utc_nwp_init_time)

    # Explicitly check for an empty spatial intersection after slicing.
    # This prevents downstream KeyErrors during DataFrame conversion.
    if ds_sliced.longitude.size == 0 or ds_sliced.latitude.size == 0:
        raise ValueError(
            "No spatial overlap found between bounding box and NWP dataset. " + \
            f"Dataset covers ({ds.latitude.max().item()}, {ds.longitude.min().item()}, " + \
            f"{ds.latitude.min().item()}, {ds.longitude.max().item()}); " + \
            f"Bounding box is {bbox_nwse}. " + \
            f"{ds_sliced=}"
        )

    # TODO: Ensure dimensions match expected OCF dimensions of
    # (init_time, step, ensemble_member, latitude, longitude).
    # Currently sliced dataset has dimensions (ensemble_member, latitude, lead_time, longitude)

    return ds_sliced


def download(ds_sliced: xr.Dataset) -> xr.Dataset:
    """Download (compute) a lazily-opened, already-sliced ECMWF ENS dataset.

    Args:
        ds_sliced: A lazy dataset as returned by :func:`open_ecmwf_ens_run`.
    """

    def download_array(var_name: str) -> dict[str, xr.DataArray]:
        return {var_name: ds_sliced[var_name].compute()}

    # The download is I/O bound (S3 network requests). We use a ThreadPoolExecutor to parallelize
    # network latency across multiple variables. A ProcessPoolExecutor would be less efficient here
    # due to the high serialization overhead of Xarray objects between processes.
    #
    # max_workers is capped rather than left at the default (one thread per variable, i.e. 13).
    # Investigation of issue #276 found that 13 concurrent chunked-zarr fetches self-contend badly
    # (S3 rate limiting or connection-pool starvation): most variables finish in 5-20s, but a few
    # straggle for minutes, making the whole download 600s+. Capping at 4 removed the stragglers
    # entirely and cut a real download from 645s to 22.5s.
    #
    # This is a recent regression, not a pre-existing property of the download: Dagster's run
    # history shows per-partition downloads holding a steady ~48-54s right up to 2026-06-30
    # 12:26 UTC, then every run afterwards (2026-07-01 onwards) taking 3-12 min. That boundary
    # lines up exactly with an `icechunk` 2.0.6 -> 2.1.0 bump in the same `uv.lock` update
    # (commit b46d145, 2026-06-30 12:26:50 UTC) — the leading theory is a change in icechunk's
    # underlying S3 client (connection pooling/concurrency handling) between those two versions,
    # though this hasn't been confirmed by pinning back to 2.0.6 and re-testing.
    data_arrays: dict[str, xr.DataArray] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(download_array, str(name)) for name in ds_sliced.data_vars.keys()
        ]
        for future in concurrent.futures.as_completed(futures):
            data_arrays.update(future.result())

    return xr.Dataset(data_arrays)


def _calc_slice_for_lat_or_lng(
    coord_name: Literal["latitude", "longitude"],
    ds: xr.Dataset,
    bbox_nwse: tuple[float, float, float, float],
) -> slice:
    """Robust slicing: xarray slice(a, b) is sensitive to coordinate direction.

    We determine if latitude/longitude are ascending or descending. In ECMWF ENS on Dynamical,
    latitude goes from +90 to -90, and longitude goes from -180 to +179.8.
    """
    min_coord, max_coord = (
        bbox_nwse[2] if coord_name == "latitude" else bbox_nwse[1],
        bbox_nwse[0] if coord_name == "latitude" else bbox_nwse[3],
    )

    if min_coord == max_coord:
        raise ValueError(f"{min_coord=} cannot be equal to {max_coord=} for {coord_name}")

    coord_array = ds[coord_name].values
    if len(coord_array) <= 1:
        raise ValueError(
            f"ds.{coord_name}.values must have multiple values. Found {len(coord_array)} values"
        )

    is_ascending = coord_array[0] < coord_array[-1]
    return slice(min_coord, max_coord) if is_ascending else slice(max_coord, min_coord)
