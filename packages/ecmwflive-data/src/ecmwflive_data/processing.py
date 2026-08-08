from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr
from schemas.dim_order import enforce_dim_order
from schemas.validation import validates

from .schema import EcmwfLiveNlSchema, EcmwfLiveUkIndiaSchema


def process_ecmwf_live(
    grib_path: Path | str,
    bbox_nwse: list[float],
    max_step_hours: int,
) -> xr.Dataset:
    """
    Process raw ECMWF Live GRIB data into an Xarray Dataset.

    This is region-agnostic and does not validate its output; callers should use
    :func:`process_ecmwf_live_uk_india` or :func:`process_ecmwf_live_nl`, which wrap this function
    and validate the result against the appropriate schema.
    """
    # Open using cfgrib with ignore_keys to prevent DatasetBuildError
    # from conflicting vertical levels and stepTypes.
    # The GRIB file contains different resolutions (numberOfPoints), so we use cfgrib.open_datasets
    # to open all hypercubes, then filter for the one we want.
    try:
        datasets = cfgrib.open_datasets(
            grib_path,
            backend_kwargs={
                "indexpath": "",
                "ignore_keys": ["valid_time", "stepType", "typeOfLevel", "surface", "heightAboveGround", "meanSea"],
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
        d for d in datasets 
        if "latitude" in d.coords and "longitude" in d.coords and d.latitude.ndim == 1 and d.longitude.ndim == 1
    ]

    if not valid_datasets:
        raise ValueError(f"No regular lat/lon grid datasets found in {grib_path}")

    # Usually there is only one dominant grid resolution containing the fields we need.
    # We group datasets by their lat/lon shapes and merge within the group that has the most variables.
    # For now, let's merge everything on the same grid shape.
    grid_shapes = {}
    for d in valid_datasets:
        shape = (len(d.latitude), len(d.longitude))
        if shape not in grid_shapes:
            grid_shapes[shape] = []
        grid_shapes[shape].append(d)

    # Pick the grid shape that has the most data variables (the primary forecast data)
    best_shape = max(grid_shapes.keys(), key=lambda s: sum(len(d.data_vars) for d in grid_shapes[s]))
    
    ds = xr.merge(grid_shapes[best_shape], compat="override")

    # Drop the 'number' scalar coordinate that cfgrib automatically adds for ECMWF data.
    # The schemas do not expect it.
    ds = ds.drop_vars(["number"], errors="ignore")

    # Spatial slicing
    # ECMWF S3 files typically use 0-360 longitude, but our schemas use -180 to 180.
    if "longitude" in ds.coords and ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")
        
    # bbox_nwse is North, West, South, East
    n, w, s, e = bbox_nwse
    ds = ds.sel(latitude=slice(n, s), longitude=slice(w, e))
    
    # Time slicing - strictly truncate to max_step_hours to remove non-hourly jumps
    if "step" in ds.coords:
        max_step = np.timedelta64(max_step_hours, "h")
        ds = ds.sel(step=slice(None, max_step))

    # Rename coordinates
    rename_coords = {}
    if "time" in ds.coords:
        rename_coords["time"] = "init_time"
    if rename_coords:
        ds = ds.rename(rename_coords)

    if "init_time" in ds.coords and "init_time" not in ds.dims:
        ds = ds.expand_dims("init_time")

    # Rename variables to match schema
    var_mapping = {
        "10u": "wind_u_10m",
        "u10": "wind_u_10m",
        "10v": "wind_v_10m",
        "v10": "wind_v_10m",
        "100u": "wind_u_100m",
        "u100": "wind_u_100m",
        "100v": "wind_v_100m",
        "v100": "wind_v_100m",
        "t2m": "temperature_2m",
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
    
    rename_vars = {k: v for k, v in var_mapping.items() if k in ds.data_vars}
    ds = ds.rename(rename_vars)

    # Unit conversions
    if "temperature_2m" in ds.data_vars:
        ds["temperature_2m"] = ds["temperature_2m"] - 273.15

    for cloud_var in ["high_cloud_cover", "medium_cloud_cover", "low_cloud_cover", "total_cloud_cover_atmosphere"]:
        if cloud_var in ds.data_vars and ds[cloud_var].max() <= 1.0:
            ds[cloud_var] = ds[cloud_var] * 100.0

    # Accumulation variables to Flux
    # ECMWF HRES Live and ENS operational data accumulate radiation from the start of the forecast.
    # We apply a forward difference to extract the accumulation over each step interval,
    # then divide by the interval duration in seconds to convert J m-2 to W m-2.
    rad_vars = [
        "downward_short_wave_radiation_flux_surface",
        "downward_long_wave_radiation_flux_surface",
        "direct_solar_radiation",
        "uv_b_radiation",
    ]
    present_rad_vars = [v for v in rad_vars if v in ds.data_vars]
    
    if present_rad_vars:
        # Duration of each step interval in seconds (step[i+1] - step[i])
        dt = (ds.step.shift(step=-1) - ds.step).dt.total_seconds()
        
        for rad_var in present_rad_vars:
            # Forward difference: Accumulation(T+dt) - Accumulation(T)
            diff_var = ds[rad_var].shift(step=-1) - ds[rad_var]
            
            # Convert J m-2 to W m-2
            flux = diff_var / dt
            
            # Clip negative numerical noise
            ds[rad_var] = np.clip(flux, a_min=0, a_max=None)
            
        # The forward difference leaves the last step as NaN. We drop it.
        ds = ds.isel(step=slice(0, -1))

    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)

    ordered_dims = ("init_time", "step", "latitude", "longitude")
    schema_vars = list(set(var_mapping.values()))
    ds_ordered = enforce_dim_order(ds, ordered_dims, keep_vars=schema_vars)

    return ds_ordered


@validates(EcmwfLiveUkIndiaSchema)
def process_ecmwf_live_uk_india(
    grib_path: Path | str,
    bbox_nwse: list[float],
    max_step_hours: int,
) -> xr.Dataset:
    """Process raw ECMWF Live GRIB data for the UK/India regions, validated against EcmwfLiveUkIndiaSchema."""
    return process_ecmwf_live(grib_path=grib_path, bbox_nwse=bbox_nwse, max_step_hours=max_step_hours)


@validates(EcmwfLiveNlSchema)
def process_ecmwf_live_nl(
    grib_path: Path | str,
    bbox_nwse: list[float],
    max_step_hours: int,
) -> xr.Dataset:
    """Process raw ECMWF Live GRIB data for the NL region, validated against EcmwfLiveNlSchema."""
    return process_ecmwf_live(grib_path=grib_path, bbox_nwse=bbox_nwse, max_step_hours=max_step_hours)
