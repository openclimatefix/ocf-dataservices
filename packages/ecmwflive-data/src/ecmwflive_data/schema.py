from typing import ClassVar

import numpy as np
import pandera.xarray as pa
from pandera.typing.xarray import Coordinate
from schemas.nwp_coordinates import init_time, latitude, longitude, step
from schemas.nwp_variables import (
    direct_solar_radiation,
    downward_long_wave_radiation_flux_surface,
    downward_short_wave_radiation_flux_surface,
    high_cloud_cover,
    low_cloud_cover,
    medium_cloud_cover,
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


class EcmwfLiveUkIndiaSchema(pa.DatasetModel):
    _chunks: ClassVar[dict[str, int]] = {"init_time": 1, "step": 1, "latitude": -1, "longitude": -1}
    _shards: ClassVar[dict[str, int]] = {"init_time": 1, "step": -1, "latitude": -1, "longitude": -1}

    _dims = ("init_time", "step", "latitude", "longitude")

    # Dimensions
    init_time: Coordinate[np.datetime64] = init_time()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=84)
    longitude: Coordinate[np.float64] = longitude(ge=-180, le=180)
    latitude: Coordinate[np.float64] = latitude(ge=-90, le=90)

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
    downward_long_wave_radiation_flux_surface: np.float32 = downward_long_wave_radiation_flux_surface(dims=_dims, nullable=True)
    downward_short_wave_radiation_flux_surface: np.float32 = downward_short_wave_radiation_flux_surface(dims=_dims, nullable=True)
    total_cloud_cover_atmosphere: np.float32 = total_cloud_cover_atmosphere(dims=_dims, nullable=False)
    visibility: np.float32 = visibility(dims=_dims, nullable=True)

    class Config:
        strict = "filter"
        strict_coords = "filter"
        

class EcmwfLiveNlSchema(pa.DatasetModel):
    _chunks: ClassVar[dict[str, int]] = {"init_time": 1, "step": 1, "latitude": -1, "longitude": -1}
    _shards: ClassVar[dict[str, int]] = {"init_time": 1, "step": -1, "latitude": -1, "longitude": -1}

    _dims = ("init_time", "step", "latitude", "longitude")

    # Dimensions
    init_time: Coordinate[np.datetime64] = init_time()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=56)
    longitude: Coordinate[np.float64] = longitude(ge=-180, le=180)
    latitude: Coordinate[np.float64] = latitude(ge=-90, le=90)

    # Variables (18 parameters)
    wind_u_10m: np.float32 = wind_u_10m(dims=_dims, nullable=True)
    wind_v_10m: np.float32 = wind_v_10m(dims=_dims, nullable=True)
    wind_u_100m: np.float32 = wind_u_100m(dims=_dims, nullable=False)
    wind_v_100m: np.float32 = wind_v_100m(dims=_dims, nullable=False)
    temperature_2m: np.float32 = temperature_2m(dims=_dims, nullable=False)
    wind_u_200m: np.float32 = wind_u_200m(dims=_dims, nullable=False)
    wind_v_200m: np.float32 = wind_v_200m(dims=_dims, nullable=False)
    high_cloud_cover: np.float32 = high_cloud_cover(dims=_dims, nullable=False)
    low_cloud_cover: np.float32 = low_cloud_cover(dims=_dims, nullable=False)
    medium_cloud_cover: np.float32 = medium_cloud_cover(dims=_dims, nullable=False)
    snow_depth: np.float32 = snow_depth(dims=_dims, nullable=True)
    direct_solar_radiation: np.float32 = direct_solar_radiation(dims=_dims, nullable=True)
    uv_b_radiation: np.float32 = uv_b_radiation(dims=_dims, nullable=True)
    downward_long_wave_radiation_flux_surface: np.float32 = downward_long_wave_radiation_flux_surface(dims=_dims, nullable=True)
    downward_short_wave_radiation_flux_surface: np.float32 = downward_short_wave_radiation_flux_surface(dims=_dims, nullable=True)
    total_precipitation_rate: np.float32 = total_precipitation_rate(dims=_dims, nullable=True)
    total_cloud_cover_atmosphere: np.float32 = total_cloud_cover_atmosphere(dims=_dims, nullable=False)
    visibility: np.float32 = visibility(dims=_dims, nullable=True)

    class Config:
        strict = "filter"
        strict_coords = "filter"
        
