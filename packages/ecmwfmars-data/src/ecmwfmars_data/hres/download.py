import datetime as dt
from pathlib import Path
from typing import Final

import cfgrib
import numpy as np
import xarray as xr
from schemas.dim_order import enforce_dim_order
from schemas.nwp_coordinates import extract_grib_attr, parse_expver_int16, utc_now_coord
from schemas.validation import is_valid_grib_file, validates

from ..ens.client import MarsClient, MarsRequest
from .schema import MarsEcmwfHresSchema

# The 18 ECMWF MARS paramIds matching EcmwfLiveSchema:
# 165=10u, 166=10v, 228246=100u, 228247=100v, 167=2t, 228239=200u, 228240=200v,
# 47=dsrp, 57=uvb, 188=hcc, 186=lcc, 187=mcc, 260048=tprate, 141=sd, 175=strd,
# 169=ssrd, 164=tcc, 3020=vis
HRES_MARS_PARAMETERS: Final[list[str]] = [
    "165",
    "166",
    "228246",
    "228247",
    "167",
    "228239",
    "228240",
    "47",
    "57",
    "188",
    "186",
    "187",
    "260048",
    "141",
    "175",
    "169",
    "164",
    "3020",
]
HRES_WEST_EUROPE_BBOX_NWSE: Final[list[int]] = [63, -12, 35, 26]
HRES_STEPS: Final[list[int]] = list(range(86))


def download_raw_hres(
    client: MarsClient,
    init_time: dt.datetime,
    bbox_nwse: list[int],
    steps: list[int],
    target_path: Path | str,
) -> None:
    """Download raw ECMWF MARS HRES data to a local GRIB file."""
    if is_valid_grib_file(target_path):
        return

    request = MarsRequest.hres(
        params=HRES_MARS_PARAMETERS,
        init_time=init_time,
        steps=steps,
        bbox_nwse=bbox_nwse,
    )

    with open(target_path, "wb") as f:
        client.execute(request, f)

    if not is_valid_grib_file(target_path):
        Path(target_path).unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file {target_path} is not a valid GRIB file.")


def _read_raw_hres_grib(grib_path: Path | str) -> xr.Dataset:
    """Read a raw MARS HRES GRIB file into an xarray Dataset."""
    try:
        datasets = cfgrib.open_datasets(
            grib_path,
            backend_kwargs={
                "indexpath": "",
                "read_keys": ["generatingProcessIdentifier", "expver", "tablesVersion"],
                "ignore_keys": [
                    "valid_time",
                    "stepType",
                    "typeOfLevel",
                    "surface",
                    "heightAboveGround",
                    "meanSea",
                ],
            },
        )
    except Exception as e:
        raise ValueError(f"Failed to open datasets from {grib_path}: {e}") from e

    if not datasets:
        raise ValueError(f"No datasets found in {grib_path}")

    return xr.merge(datasets, compat="override", combine_attrs="override")


def _transform_hres(ds: xr.Dataset, max_step_hours: int = 84) -> xr.Dataset:
    """Rename, convert units, de-accumulate radiation, and reorder dimensions to MarsEcmwfHresSchema."""
    cycle_val = int(extract_grib_attr(ds, "generatingProcessIdentifier", -1))
    expver_val = parse_expver_int16(extract_grib_attr(ds, "expver", -1))
    tables_val = int(extract_grib_attr(ds, "tablesVersion", -1))

    if "longitude" in ds.coords and float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")

    if "time" in ds.coords:
        ds = ds.rename({"time": "init_time"})
    if "init_time" in ds.coords and "init_time" not in ds.dims:
        ds = ds.expand_dims("init_time")

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

    rename_vars = {k: v for k, v in var_mapping.items() if k in ds.data_vars}
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
            if float(ds[cloud_var].max()) <= 1.1:
                ds[cloud_var] = ds[cloud_var] * 100.0
            ds[cloud_var] = np.clip(ds[cloud_var], a_min=0.0, a_max=100.0)

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
            ds[rad_var] = np.clip(diff_var / step_dt, a_min=0, a_max=None)
        ds = ds.isel(step=slice(0, -1))

    if "step" in ds.coords:
        ds = ds.sel(step=slice(None, np.timedelta64(max_step_hours, "h")))

    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)

    ordered_dims = MarsEcmwfHresSchema.dims()
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

    schema_vars = list(set(var_mapping.values()))
    return enforce_dim_order(ds, ordered_dims, keep_vars=schema_vars)


@validates(MarsEcmwfHresSchema)
def convert_hres_to_dataset(grib_path: Path | str, max_step_hours: int = 84) -> xr.Dataset:
    """Convert a raw ECMWF MARS HRES GRIB file to an xarray Dataset matching MarsEcmwfHresSchema."""
    return _transform_hres(_read_raw_hres_grib(grib_path), max_step_hours=max_step_hours)
