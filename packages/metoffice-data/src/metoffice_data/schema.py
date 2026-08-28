from typing import ClassVar

import numpy as np
from pandera.typing.xarray import Coordinate
from schemas.base import NwpDatasetSchema
from schemas.nwp_coordinates import (
    init_time,
    latitude,
    longitude,
    step,
    x_laea,
    y_laea,
)
from schemas.nwp_variables import (
    downward_long_wave_radiation_flux_surface,
    downward_short_wave_radiation_flux_surface,
    high_cloud_cover,
    low_cloud_cover,
    medium_cloud_cover,
    pressure_reduced_to_mean_sea_level,
    relative_humidity_2m,
    snow_depth,
    temperature_2m,
    total_cloud_cover_atmosphere,
    total_precipitation_rate,
    visibility,
    wind_direction_10m,
    wind_speed_10m,
    wind_u_10m,
    wind_v_10m,
)


class MetOfficeGlobalWesteuropeSchema(NwpDatasetSchema):
    _chunks: ClassVar[dict[str, int]] = {"init_time": 1, "step": 1, "latitude": -1, "longitude": -1}
    _shards: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": -1,
        "latitude": -1,
        "longitude": -1,
    }

    _dims = ("init_time", "step", "latitude", "longitude")

    # Dimensions
    init_time: Coordinate[np.datetime64] = init_time()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=54)
    longitude: Coordinate[np.float64] = longitude(ge=-180, le=180)
    latitude: Coordinate[np.float64] = latitude(ge=-90, le=90)

    # Variables (12 parameters)
    total_cloud_cover_atmosphere: np.float32 = total_cloud_cover_atmosphere(
        dims=_dims, nullable=False
    )
    high_cloud_cover: np.float32 = high_cloud_cover(dims=_dims, nullable=False)
    medium_cloud_cover: np.float32 = medium_cloud_cover(dims=_dims, nullable=False)
    low_cloud_cover: np.float32 = low_cloud_cover(dims=_dims, nullable=False)
    visibility: np.float32 = visibility(dims=_dims, nullable=True)
    relative_humidity_2m: np.float32 = relative_humidity_2m(dims=_dims, nullable=False)
    pressure_reduced_to_mean_sea_level: np.float32 = pressure_reduced_to_mean_sea_level(
        dims=_dims, nullable=False
    )
    snow_depth: np.float32 = snow_depth(dims=_dims, nullable=True)
    downward_short_wave_radiation_flux_surface: np.float32 = (
        downward_short_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    temperature_2m: np.float32 = temperature_2m(dims=_dims, nullable=False)
    wind_u_10m: np.float32 = wind_u_10m(dims=_dims, nullable=True)
    wind_v_10m: np.float32 = wind_v_10m(dims=_dims, nullable=True)

    class Config(NwpDatasetSchema.Config):
        pass


class MetOfficeGlobalIndiaSchema(NwpDatasetSchema):
    _chunks: ClassVar[dict[str, int]] = {"init_time": 1, "step": 1, "latitude": -1, "longitude": -1}
    _shards: ClassVar[dict[str, int]] = {
        "init_time": 1,
        "step": -1,
        "latitude": -1,
        "longitude": -1,
    }

    _dims = ("init_time", "step", "latitude", "longitude")

    # Dimensions
    init_time: Coordinate[np.datetime64] = init_time()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=54)
    longitude: Coordinate[np.float64] = longitude(ge=-180, le=180)
    latitude: Coordinate[np.float64] = latitude(ge=-90, le=90)

    # Variables (11 parameters)
    total_cloud_cover_atmosphere: np.float32 = total_cloud_cover_atmosphere(
        dims=_dims, nullable=False
    )
    high_cloud_cover: np.float32 = high_cloud_cover(dims=_dims, nullable=False)
    medium_cloud_cover: np.float32 = medium_cloud_cover(dims=_dims, nullable=False)
    low_cloud_cover: np.float32 = low_cloud_cover(dims=_dims, nullable=False)
    visibility: np.float32 = visibility(dims=_dims, nullable=True)
    relative_humidity_2m: np.float32 = relative_humidity_2m(dims=_dims, nullable=False)
    snow_depth: np.float32 = snow_depth(dims=_dims, nullable=True)
    downward_short_wave_radiation_flux_surface: np.float32 = (
        downward_short_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    temperature_2m: np.float32 = temperature_2m(dims=_dims, nullable=False)
    wind_u_10m: np.float32 = wind_u_10m(dims=_dims, nullable=True)
    wind_v_10m: np.float32 = wind_v_10m(dims=_dims, nullable=True)

    class Config(NwpDatasetSchema.Config):
        pass


class MetOfficeUkvSchema(NwpDatasetSchema):
    _chunks: ClassVar[dict[str, int]] = {"init_time": 1, "step": 1, "y_laea": -1, "x_laea": -1}
    _shards: ClassVar[dict[str, int]] = {"init_time": 1, "step": -1, "y_laea": -1, "x_laea": -1}

    _dims = ("init_time", "step", "y_laea", "x_laea")

    # Dimensions. The UKV model is kept on its native Lambert Azimuthal Equal Area grid
    # (metres), not remapped to lat/lon.
    init_time: Coordinate[np.datetime64] = init_time()
    step: Coordinate[np.timedelta64] = step(ge_hours=0, le_hours=42)
    y_laea: Coordinate[np.float64] = y_laea(ge=-576000, le=700000)
    x_laea: Coordinate[np.float64] = x_laea(ge=-576000, le=332000)

    # Variables (12 parameters)
    high_cloud_cover: np.float32 = high_cloud_cover(dims=_dims, nullable=False)
    medium_cloud_cover: np.float32 = medium_cloud_cover(dims=_dims, nullable=False)
    low_cloud_cover: np.float32 = low_cloud_cover(dims=_dims, nullable=False)
    visibility: np.float32 = visibility(dims=_dims, nullable=True)
    relative_humidity_2m: np.float32 = relative_humidity_2m(dims=_dims, nullable=False)
    total_precipitation_rate: np.float32 = total_precipitation_rate(dims=_dims, nullable=True)
    snow_depth: np.float32 = snow_depth(dims=_dims, nullable=True)
    downward_long_wave_radiation_flux_surface: np.float32 = (
        downward_long_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    downward_short_wave_radiation_flux_surface: np.float32 = (
        downward_short_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    temperature_2m: np.float32 = temperature_2m(dims=_dims, nullable=False)
    wind_speed_10m: np.float32 = wind_speed_10m(dims=_dims, nullable=True)
    wind_direction_10m: np.float32 = wind_direction_10m(dims=_dims, nullable=True)

    class Config(NwpDatasetSchema.Config):
        pass
