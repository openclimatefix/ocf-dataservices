from typing import ClassVar

import numpy as np
import xarray as xr
from pandera import check
from pandera.typing.xarray import Coordinate
from schemas.base import NwpDatasetSchema
from schemas.nwp_coordinates import ensemble_member, init_time, latitude, longitude, step
from schemas.nwp_variables import (
    categorical_precipitation_type_surface,
    downward_long_wave_radiation_flux_surface,
    downward_short_wave_radiation_flux_surface,
    precipitation_surface,
    pressure_reduced_to_mean_sea_level,
    temperature_2m,
    total_cloud_cover_atmosphere,
    wind_u_10m,
    wind_u_100m,
    wind_v_10m,
    wind_v_100m,
)


class DynamicalEcmwfEnsSchema(NwpDatasetSchema):
    _chunks: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": 1,
        "ensemble_member": 1,
        "latitude": -1,
        "longitude": -1,
    }
    _shards: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": -1,
        "ensemble_member": -1,
        "latitude": -1,
        "longitude": -1,
    }

    _dims = ("init_time", "step", "ensemble_member", "latitude", "longitude")

    # Dimensions
    init_time: Coordinate[np.datetime64] = init_time()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=85)
    ensemble_member: Coordinate[np.int16] = ensemble_member(ge=1, le=50)
    longitude: Coordinate[np.float64] = longitude(ge=-180, le=180)
    latitude: Coordinate[np.float64] = latitude(ge=-90, le=90)

    # Variables
    temperature_2m: np.float32 = temperature_2m(dims=_dims, nullable=False)
    wind_u_100m: np.float32 = wind_u_100m(dims=_dims, nullable=False)
    wind_v_100m: np.float32 = wind_v_100m(dims=_dims, nullable=False)
    pressure_reduced_to_mean_sea_level: np.float32 = pressure_reduced_to_mean_sea_level(
        dims=_dims, nullable=False
    )
    total_cloud_cover_atmosphere: np.float32 = total_cloud_cover_atmosphere(
        dims=_dims, nullable=False
    )
    downward_long_wave_radiation_flux_surface: np.float32 = (
        downward_long_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    downward_short_wave_radiation_flux_surface: np.float32 = (
        downward_short_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    precipitation_surface: np.float32 = precipitation_surface(dims=_dims, nullable=True)
    categorical_precipitation_type_surface: np.float32 = categorical_precipitation_type_surface(
        dims=_dims, nullable=True
    )
    wind_v_10m: np.float32 = wind_v_10m(dims=_dims, nullable=True)
    wind_u_10m: np.float32 = wind_u_10m(dims=_dims, nullable=True)

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

    class Config(NwpDatasetSchema.Config):
        chunked = True  # Ensures the dataset is chunked
