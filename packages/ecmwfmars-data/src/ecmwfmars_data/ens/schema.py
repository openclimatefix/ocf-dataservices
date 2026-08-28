from typing import ClassVar

import numpy as np
from pandera.typing.xarray import Coordinate
from schemas.base import NwpDatasetSchema
from schemas.nwp_coordinates import ensemble_member, init_time, latitude, longitude, step
from schemas.nwp_variables import (
    downward_short_wave_radiation_flux_surface,
    high_cloud_cover,
    low_cloud_cover,
    medium_cloud_cover,
    temperature_2m,
)


class MarsEcmwfEnsSchema(NwpDatasetSchema):
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
    downward_short_wave_radiation_flux_surface: np.float32 = (
        downward_short_wave_radiation_flux_surface(dims=_dims, nullable=True)
    )
    high_cloud_cover: np.float32 = high_cloud_cover(dims=_dims, nullable=False)
    medium_cloud_cover: np.float32 = medium_cloud_cover(dims=_dims, nullable=False)
    low_cloud_cover: np.float32 = low_cloud_cover(dims=_dims, nullable=False)

    class Config(NwpDatasetSchema.Config):
        pass
