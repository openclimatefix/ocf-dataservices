import datetime as dt
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr
from schemas.dim_order import enforce_dim_order
from schemas.validation import validates

from .client import MarsClient, MarsRequest
from .schema import MarsEcmwfEnsSchema

_MARS_PARAMETERS: Final[list[str]] = ["167", "169", "186", "187", "188"]


def download_raw(
    client: MarsClient,
    init_time: dt.datetime,
    bbox_nwse: list[int],
    steps: list[int],
    numbers: list[int],
    target_path: Path | str,
) -> None:
    """Download raw ECMWF MARS ENS data to a local GRIB file.

    Args:
        client: Authenticated MarsClient instance.
        init_time: Initialization time of the forecast.
        bbox_nwse: Bounding box in North, West, South, East order.
        steps: Forecast steps (lead times) in hours.
        numbers: Ensemble member numbers to download.
        target_path: Path where the raw GRIB file will be saved.
    """
    request = MarsRequest.ens(
        params=_MARS_PARAMETERS,
        init_time=init_time,
        steps=steps,
        bbox_nwse=bbox_nwse,
        number=numbers,
    )

    with open(target_path, "wb") as f:
        client.execute(request, f)


def _read_raw_grib(grib_path: Path | str) -> xr.Dataset:
    """Read a raw MARS ENS GRIB file into an Xarray Dataset with its original variable/coord names.

    Args:
        grib_path: Path to the raw GRIB file downloaded via `download_raw`.
    """
    # Open instantaneous and accumulated fields separately to avoid stepType conflicts
    # in cfgrib, then merge them.
    ds_inst = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"stepType": "instant"}},
    )
    try:
        ds_accum = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"stepType": "accum"}},
        )
        return xr.merge([ds_inst, ds_accum], compat="override")
    except Exception:  # noqa: BLE001
        # If there are no accumulated fields (e.g. no ssrd) in the file, open_dataset might fail
        return ds_inst


def _transform(ds: xr.Dataset) -> xr.Dataset:
    """Rename, convert units, and reorder dimensions to match MarsEcmwfEnsSchema."""
    # Rename coordinates
    rename_coords = {}
    if "time" in ds.coords:
        rename_coords["time"] = "init_time"
    if "number" in ds.coords:
        rename_coords["number"] = "ensemble_member"

    if rename_coords:
        ds = ds.rename(rename_coords)

    # Make sure init_time is a dimension
    if "init_time" in ds.coords and "init_time" not in ds.dims:
        ds = ds.expand_dims("init_time")

    # Mapping of shortName to schema variable name
    var_mapping = {
        "t2m": "temperature_2m",
        "ssrd": "downward_short_wave_radiation_flux_surface",
        "hcc": "high_cloud_cover",
        "mcc": "medium_cloud_cover",
        "lcc": "low_cloud_cover",
    }

    # Rename variables present in the dataset
    rename_vars = {k: v for k, v in var_mapping.items() if k in ds.data_vars}
    ds = ds.rename(rename_vars)

    # Unit conversions
    if "temperature_2m" in ds.data_vars:
        ds["temperature_2m"] = ds["temperature_2m"] - 273.15  # K -> °C

    # ECMWF MARS radiation steps are accumulated since the start of the forecast.
    # Apply a forward difference to extract the accumulation over each step interval,
    # then divide by the interval duration in seconds to convert J m-2 to W m-2.
    # Taken from the existing forecast app.
    if "downward_short_wave_radiation_flux_surface" in ds.data_vars:
        rad_var = "downward_short_wave_radiation_flux_surface"

        dt = (ds.step.shift(step=-1) - ds.step).dt.total_seconds()

        # Forward difference: Accumulation(T+dt) - Accumulation(T)
        diff_var = ds[rad_var].shift(step=-1) - ds[rad_var]

        # Convert J m-2 to W m-2
        flux = diff_var / dt

        # Prevent negative values due to spectral ringing
        ds[rad_var] = np.clip(flux, a_min=0, a_max=None)

        # Drop the last step since its forward difference is NaN
        ds = ds.isel(step=slice(0, -1))

    for cloud_var in ["high_cloud_cover", "medium_cloud_cover", "low_cloud_cover"]:
        if cloud_var in ds.data_vars:
            # Cloud cover gets clipped because GRIB packing can produce values slightly above 1
            ds[cloud_var] = np.clip(ds[cloud_var] * 100, a_min=0, a_max=100)

    # Ensure all data variables are float32
    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)

    if "ensemble_member" in ds.coords:
        ds["ensemble_member"] = ds["ensemble_member"].astype(np.int16)

    ordered_dims = MarsEcmwfEnsSchema.dims()
    extra_coords = [c for c in ds.coords if c not in ordered_dims]
    if extra_coords:
        ds = ds.drop_vars(extra_coords, errors="ignore")

    # Reorder dimensions exactly as required by the schema, and drop any variables not in the
    # schema (like 'surface').
    return enforce_dim_order(ds, ordered_dims, keep_vars=list(var_mapping.values()))


@validates(MarsEcmwfEnsSchema)
def convert_to_dataset(grib_path: Path | str) -> xr.Dataset:
    """Convert a raw ECMWF MARS ENS GRIB file to an Xarray Dataset matching MarsEcmwfEnsSchema.

    The returned dataset is validated against MarsEcmwfEnsSchema before being returned.

    Args:
        grib_path: Path to the raw GRIB file downloaded via `download_raw`.

    Returns:
        An xarray Dataset conforming to the MarsEcmwfEnsSchema.
    """
    return _transform(_read_raw_grib(grib_path))
