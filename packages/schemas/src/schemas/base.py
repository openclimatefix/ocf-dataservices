from typing import Any, ClassVar

import numpy as np
import pandera.xarray as pa
import xarray as xr
from pandera import check, dataframe_check
from pandera.api.xarray.model_config import DatasetConfig


class _NwpDatasetConfig(DatasetConfig):
    strict = "filter"  # Drops unlisted variables
    strict_coords = "filter"  # Drops unlisted coordinates


class NwpDatasetSchema(pa.DatasetModel):
    """Base Pandera schema for NWP `xarray.Dataset` objects.

    Subclasses must declare non-empty `_dims`, `_chunks`, and `_shards` class attributes with
    identical dimension keys.

    This schema validates dataset dimension order, coordinate types, nullability, and physical unit
    bounds. Spatial coordinates (`latitude`, `longitude`) are checked against global bounds only;
    provider packages must slice data to the target regional bounding box before validation runs.
    """

    _dims: ClassVar[tuple[str, ...]] = ()
    _chunks: ClassVar[dict[str, int]] = {}
    _shards: ClassVar[dict[str, int]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if not cls._dims:
            raise ValueError(f"{cls.__name__} must declare a non-empty `_dims` tuple.")
        if not cls._chunks or tuple(cls._chunks.keys()) != tuple(cls._dims):
            raise ValueError(
                f"{cls.__name__}._chunks keys {list(cls._chunks.keys())} must match `_dims` "
                f"{list(cls._dims)}."
            )
        if not cls._shards or tuple(cls._shards.keys()) != tuple(cls._dims):
            raise ValueError(
                f"{cls.__name__}._shards keys {list(cls._shards.keys())} must match `_dims` "
                f"{list(cls._dims)}."
            )
        cls.Config = type(
            "Config",
            (cls.Config,),
            {"dims": tuple(cls._dims), "ordered_dims": True},
        )
        super().__init_subclass__(**kwargs)

    @classmethod
    def dims(cls) -> tuple[str, ...]:
        return cls._dims

    @classmethod
    def append_dim(cls) -> str:
        return cls._dims[0]

    @dataframe_check
    def _dims_ordered_and_non_empty(cls, ds: xr.Dataset) -> bool:
        """Verify that the dataset dimension order matches `_dims` and no dimension is empty."""
        return tuple(ds.dims) == tuple(cls._dims) and all(ds.sizes[d] > 0 for d in cls._dims)

    @check("step", "init_time", "ensemble_member", regex=True)
    def _coord_strictly_increasing(cls, da: xr.DataArray) -> bool:
        """Verify that time and ensemble coordinates are strictly increasing."""
        if da.size <= 1:
            return True
        return bool(np.all(da.values[:-1] < da.values[1:]))

    @check(
        "total_cloud_cover_atmosphere",
        "high_cloud_cover",
        "medium_cloud_cover",
        "low_cloud_cover",
        "relative_humidity_2m",
        regex=True,
    )
    def _is_percent_not_fraction(cls, da: xr.DataArray) -> bool:
        """Verify that percentage fields are scaled to `[0, 100]` rather than `[0, 1]`."""
        if da.size == 0 or bool(da.isnull().all()):
            return False
        max_val = float(da.max())
        return max_val > 1.1 or max_val == 0.0

    @check(
        "downward_short_wave_radiation_flux_surface",
        "downward_long_wave_radiation_flux_surface",
        "direct_solar_radiation",
        "uv_b_radiation",
        "total_precipitation_rate",
        regex=True,
    )
    def _is_instantaneous_not_cumulative(cls, da: xr.DataArray) -> bool:
        """Reject fields that monotonically accumulate along `step` instead of representing rates."""
        if "step" not in da.dims or da.sizes["step"] < 2:
            return True
        diffs = da.diff("step")
        if not bool(diffs.notnull().any()):
            return True
        never_decreases = not bool((diffs < 0).any())
        strictly_increases = bool((diffs > 0).any())
        return not (never_decreases and strictly_increases)

    Config: type[DatasetConfig] = _NwpDatasetConfig
