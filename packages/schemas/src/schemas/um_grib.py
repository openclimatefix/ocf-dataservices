from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from .nwp_variables import scale_to_percent

UM_UNKNOWN_PARAMETER_NUMBERS: dict[int, str] = {
    1: "tcc",
    192: "u10",
    193: "v10",
    194: "wdir",
    195: "si10",
}

UM_VAR_MAPPING: dict[str, str] = {
    "tcc": "total_cloud_cover_atmosphere",
    "clt": "total_cloud_cover_atmosphere",
    "hcc": "high_cloud_cover",
    "mcc": "medium_cloud_cover",
    "lcc": "low_cloud_cover",
    "vis": "visibility",
    "r": "relative_humidity_2m",
    "r2": "relative_humidity_2m",
    "prmsl": "pressure_reduced_to_mean_sea_level",
    "msl": "pressure_reduced_to_mean_sea_level",
    "mslp": "pressure_reduced_to_mean_sea_level",
    "sd": "snow_depth",
    "sdwe": "snow_depth",
    "sde": "snow_depth",
    "swavr": "downward_short_wave_radiation_flux_surface",
    "dswrf": "downward_short_wave_radiation_flux_surface",
    "sdswrf": "downward_short_wave_radiation_flux_surface",
    "ssrd": "downward_short_wave_radiation_flux_surface",
    "dlwrf": "downward_long_wave_radiation_flux_surface",
    "sdlwrf": "downward_long_wave_radiation_flux_surface",
    "strd": "downward_long_wave_radiation_flux_surface",
    "t2m": "temperature_2m",
    "t": "temperature_2m",
    "tas": "temperature_2m",
    "u10": "wind_u_10m",
    "10u": "wind_u_10m",
    "u": "wind_u_10m",
    "uas": "wind_u_10m",
    "v10": "wind_v_10m",
    "10v": "wind_v_10m",
    "v": "wind_v_10m",
    "vas": "wind_v_10m",
    "si10": "wind_speed_10m",
    "10si": "wind_speed_10m",
    "wdir": "wind_direction_10m",
    "wdir10": "wind_direction_10m",
    "10wdir": "wind_direction_10m",
    "prate": "total_precipitation_rate",
    "tprate": "total_precipitation_rate",
    "rprate": "total_precipitation_rate",
    "lsrr": "total_precipitation_rate",
}

UM_SCHEMA_VARS: list[str] = list(
    set(UM_VAR_MAPPING.values()) - {"wind_speed_10m", "wind_direction_10m"}
)


def read_um_grib_datasets(
    grib_path: Path | str,
    read_keys: Sequence[str],
) -> list[xr.Dataset]:
    """Read a MetOffice Unified Model GRIB2 file, resolving local `'unknown'` parameter numbers."""
    import cfgrib

    datasets: list[xr.Dataset] = []
    for param_num, target_name in UM_UNKNOWN_PARAMETER_NUMBERS.items():
        try:
            ds_params = cfgrib.open_datasets(
                grib_path,
                backend_kwargs={
                    "indexpath": "",
                    "read_keys": list(read_keys),
                    "filter_by_keys": {"shortName": "unknown", "parameterNumber": param_num},
                },
            )
            for ds_param in ds_params:
                if "unknown" in ds_param.data_vars:
                    datasets.append(ds_param.rename({"unknown": target_name}))
        except Exception:  # noqa: S110, BLE001
            pass

    try:
        opened = cfgrib.open_datasets(
            grib_path,
            backend_kwargs={"indexpath": "", "read_keys": list(read_keys)},
        )
    except Exception as e:
        raise ValueError(f"Failed to open datasets from {grib_path}: {e}") from e

    for ds in opened:
        if "unknown" in ds.data_vars:
            ds = ds.drop_vars("unknown")
        if len(ds.data_vars) > 0:
            datasets.append(ds)

    if not datasets:
        raise ValueError(f"No valid datasets found in {grib_path}")

    return datasets


def normalize_um_variables(ds: xr.Dataset) -> tuple[xr.Dataset, list[str]]:
    """Rename MetOffice UM variables to canonical names and convert Kelvin, %, radiation, and wind."""
    rename_vars = {k: v for k, v in UM_VAR_MAPPING.items() if k in ds.data_vars}
    ds = ds.rename(rename_vars)

    if "temperature_2m" in ds.data_vars:
        ds["temperature_2m"] = ds["temperature_2m"] - 273.15

    for pct_var in (
        "total_cloud_cover_atmosphere",
        "high_cloud_cover",
        "medium_cloud_cover",
        "low_cloud_cover",
        "relative_humidity_2m",
    ):
        if pct_var in ds.data_vars:
            cap = 120.0 if pct_var == "relative_humidity_2m" else 100.0
            ds[pct_var] = scale_to_percent(ds[pct_var], max_val=cap)

    for rad_var in (
        "downward_short_wave_radiation_flux_surface",
        "downward_long_wave_radiation_flux_surface",
    ):
        if rad_var in ds.data_vars:
            ds[rad_var] = ds[rad_var].clip(min=0.0)

    if "wind_speed_10m" in ds.data_vars and "wind_direction_10m" in ds.data_vars:
        if "wind_u_10m" not in ds.data_vars or "wind_v_10m" not in ds.data_vars:
            rad = np.deg2rad(ds["wind_direction_10m"])
            ds["wind_u_10m"] = -ds["wind_speed_10m"] * np.sin(rad)
            ds["wind_v_10m"] = -ds["wind_speed_10m"] * np.cos(rad)
        ds = ds.drop_vars(["wind_speed_10m", "wind_direction_10m"])

    return ds, UM_SCHEMA_VARS
