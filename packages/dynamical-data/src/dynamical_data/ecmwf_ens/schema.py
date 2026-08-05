from typing import ClassVar

import numpy as np
import pandera.xarray as pa
import xarray as xr
from pandera import check
from pandera.typing.xarray import Coordinate


class EcmwfEnsSchema(pa.DatasetModel):
    _chunks: ClassVar[dict[str, int]] = {"init_time": 1, "step": 1, "ensemble_member": 1, "latitude": -1, "longitude": -1}
    _shards: ClassVar[dict[str, int]] = {"init_time": 1, "step": -1, "ensemble_member": -1, "latitude": -1, "longitude": -1}

    _dims = ("init_time", "step", "ensemble_member", "latitude", "longitude")

    # Dimensions
    init_time: Coordinate[np.datetime64] = pa.Field(
        dims=("init_time",),
        nullable=False,
    )
    step: Coordinate[np.timedelta64] = pa.Field(
        dims=("step",),
        nullable=False,
        ge=0,
        le=85,
    )
    ensemble_member: Coordinate[np.int16] = pa.Field(
        dims=("ensemble_member",),
        nullable=False,
        ge=1,
        le=51,
    )
    longitude: Coordinate[np.float64] = pa.Field(
        dims=("longitude",),
        nullable=False,
        ge=-180,
        le=180,
    )
    latitude: Coordinate[np.float64] = pa.Field(
        dims=("latitude",),
        nullable=False,
        ge=-90,
        le=90,
    )

    # Variables
    temperature_2m: np.float32 = pa.Field(
        dims=_dims,
        ge=-100, # degrees Celsius
        le=100,
        nullable=False,
    )
    dew_point_temperature_2m: np.float32 = pa.Field(
        dims=_dims,
        ge=-100, # degrees Celsius
        le=100,
        nullable=False,
    )
    wind_u_100m: np.float32 = pa.Field(
        dims=_dims,
        ge=-115, # m/s
        le=115,
        nullable=False,
    )
    wind_v_100m: np.float32 = pa.Field(
        dims=_dims,
        ge=-115, # m/s
        le=115,
        nullable=False,
    )
    pressure_reduced_to_mean_sea_level: np.float32 = pa.Field(
        dims=_dims,
        ge=80000, # Pa
        le=115000,
        nullable=False,
    )
    total_cloud_cover_atmosphere: np.float32 = pa.Field(
        dims=_dims,
        ge=0, # percent
        le=100,
        nullable=False,
    )
    downward_long_wave_radiation_flux_surface: np.float32 = pa.Field(
        dims=_dims,
        ge=0, # W m-2
        le=2000,
        nullable=True,
    )
    downward_short_wave_radiation_flux_surface: np.float32 = pa.Field(
        dims=_dims,
        ge=0, # W m-2
        le=2000,
        nullable=True,
    )
    precipitation_surface: np.float32 = pa.Field(
        dims=_dims,
        ge=0, # kg m-2 s-1
        le=1,
        nullable=True,
    )
    categorical_precipitation_type_surface: np.float32 = pa.Field(
        dims=_dims,
        ge=0,
        le=255,
        nullable=True,
    )
    wind_v_10m: np.float32 = pa.Field(
        dims=_dims,
        ge=-115, # m/s
        le=115,
        nullable=True,
    )
    wind_u_10m: np.float32 = pa.Field(
        dims=_dims,
        ge=-115, # m/s
        le=115,
        nullable=True,
    )

    @check(
        "downward_long_wave_radiation_flux_surface",
        "downward_short_wave_radiation_flux_surface",
        "precipitation_surface",
        "categorical_precipitation_type_surface",
        "wind_v_10m",
        "wind_u_10m",
    )
    def max_2_percent_null(cls, da: xr.DataArray) -> bool:
        return float(da.isnull().mean()) < 0.02

    class Config:
        strict = "filter" # Drops unlisted variables
        strict_coords = "filter" # Drops unlisted coordinates
        chunked=True # Ensures the dataset is chunked
