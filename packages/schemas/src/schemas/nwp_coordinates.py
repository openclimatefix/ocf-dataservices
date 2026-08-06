from typing import Any

import numpy as np
import pandera.xarray as pa


def init_time() -> Any:
    return pa.Field(dims=("init_time",), nullable=False)

def step(ge_hours: int = 0, le_hours: int = 85) -> Any:
    return pa.Field(
        dims=("step",),
        nullable=False,
        ge=np.timedelta64(ge_hours, 'h'),
        le=np.timedelta64(le_hours, 'h'),
    )

def ensemble_member(ge: int = 1, le: int = 51) -> Any:
    return pa.Field(dims=("ensemble_member",), nullable=False, ge=ge, le=le)

def longitude(ge: float = -180, le: float = 180) -> Any:
    return pa.Field(dims=("longitude",), nullable=False, ge=ge, le=le)

def latitude(ge: float = -90, le: float = 90) -> Any:
    return pa.Field(dims=("latitude",), nullable=False, ge=ge, le=le)
