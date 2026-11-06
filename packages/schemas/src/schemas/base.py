from typing import Any, ClassVar

import numpy as np
import pandera.xarray as pa
import xarray as xr
from pandera import check, dataframe_check


class NwpDatasetSchema(pa.DatasetModel):
    """Shared base for NWP dataset schemas.

    All NWP datasets should have `_dims`, `_chunks`, and `_shards` declared. This base class
    enforces that.

    The schema should validate a dataset's shape (declared dims/coords present, in the declared
    dtype), nullability, and physical unit bounds (e.g. a temperature field's plausible range).

    However, note that the spatial fields are checked against global bounds (e.g. -90/90,
    -180/180 in the case of lat/lon), NOT the given asset's bounding box. The spatial region is
    it is a provider-package data-selection responsibility, applied by slicing the data before
    schema validation runs (see e.g. `open_it`/`download` in `dynamical_data.ecmwf_ens.download`,
    or the `bbox_nwse` handling in `ecmwflive_data.processing`). Passing schema validation confirms
    the data is physically plausible and correctly shaped; it does not confirm the data covers the
    region a particular asset expects.
    """

    _dims: ClassVar[tuple[str, ...]] = ()
    _chunks: ClassVar[dict[str, int]] = {}
    _shards: ClassVar[dict[str, int]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
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

    @classmethod
    def dims(cls) -> tuple[str, ...]:
        return cls._dims

    @classmethod
    def append_dim(cls) -> str:
        return cls._dims[0]

    @dataframe_check
    def _dims_ordered_and_non_empty(cls, ds: xr.Dataset) -> bool:
        """Dataset-level dimension order must match `_dims`, and no dimension may be empty."""
        return tuple(ds.dims) == tuple(cls._dims) and all(ds.sizes[d] > 0 for d in cls._dims)

    @check("step", "init_time", "ensemble_member", regex=True)
    def _coord_strictly_increasing(cls, da: xr.DataArray) -> bool:
        """Time and ensemble coordinates must be strictly increasing (sorted and unique)."""
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
        """Checks whether the max of the data array is greater than 1.1 or equal to 0.0.
        If max is (0, 1.1] (allowing for GRIB simple-packing overshoot above 1.0), the data unit is
        the unit interval [0, 1], not a percentage [0, 100].
        """
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

    class Config:
        strict = "filter"  # Drops unlisted variables
        strict_coords = "filter"  # Drops unlisted coordinates
