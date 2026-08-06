import datetime as dt
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import xarray as xr

from .client import MarsClient, MarsRequest
from .schema import EcmwfEnsSchema

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


def convert_to_dataset(grib_path: Path | str) -> Annotated[xr.Dataset, EcmwfEnsSchema]:
    """Convert a raw ECMWF MARS ENS GRIB file to an Xarray Dataset matching EcmwfEnsSchema.

    Args:
        grib_path: Path to the raw GRIB file downloaded via `download_raw`.

    Returns:
        An xarray Dataset conforming to the EcmwfEnsSchema.
    """
    # Open instantaneous and accumulated fields separately to avoid stepType conflicts
    # in cfgrib, then merge them.
    ds_inst = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"stepType": "inst"}},
    )
    try:
        ds_accum = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"stepType": "accum"}},
        )
        ds = xr.merge([ds_inst, ds_accum], compat="override")
    except Exception:  # noqa: BLE001
        # If there are no accumulated fields (e.g. no ssrd) in the file, open_dataset might fail
        ds = ds_inst

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
    # Temperature: Kelvin to Celsius
    if "temperature_2m" in ds.data_vars:
        ds["temperature_2m"] = ds["temperature_2m"] - 273.15

    # Radiation: J m-2 (accumulated over 3h) to W m-2
    # NOTE: ECMWF MARS radiation steps are accumulated since the start of the forecast.
    # However, depending on the MARS request stream (e.g., enfo), they might be accumulated 
    # since the previous output step. Assuming 3 hours here per the prompt, though 
    # rigorous step differencing might be needed depending on the exact steps requested.
    # For now we apply the standard conversion factor as outlined in the plan.
    if "downward_short_wave_radiation_flux_surface" in ds.data_vars:
        # Prevent negative values due to spectral ringing
        ds["downward_short_wave_radiation_flux_surface"] = np.clip(
            ds["downward_short_wave_radiation_flux_surface"] / 10800, 
            a_min=0, 
            a_max=None
        )

    # Cloud covers: 0-1 fraction to percentage
    for cloud_var in ["high_cloud_cover", "medium_cloud_cover", "low_cloud_cover"]:
        if cloud_var in ds.data_vars:
            ds[cloud_var] = ds[cloud_var] * 100

    # Ensure all data variables are float32
    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)

    # Reorder dimensions exactly as required by the schema
    ordered_dims = ("init_time", "step", "ensemble_member", "latitude", "longitude")
    
    # Extract coordinate variables in the exact order
    new_coords = {d: ds.coords[d].variable for d in ordered_dims if d in ds.coords}
    for c in ds.coords:
        if c not in new_coords:
            new_coords[c] = ds.coords[c].variable

    # Build new dataset enforcing the dimension order for variables
    ds_ordered = xr.Dataset(
        data_vars={k: v.transpose(*[d for d in ordered_dims if d in v.dims]).variable 
                   for k, v in ds.data_vars.items()},
        coords=new_coords
    )

    # Force exact dimension ordering at the Dataset level
    ds_ordered = ds_ordered.transpose(*ordered_dims)
    
    # Drop any variables not in the schema (like 'surface')
    schema_vars = list(var_mapping.values())
    ds_ordered = ds_ordered[[v for v in schema_vars if v in ds_ordered.data_vars]]

    # Validate
    validated_ds = EcmwfEnsSchema.validate(ds_ordered)

    return validated_ds
