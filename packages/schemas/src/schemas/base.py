from typing import ClassVar

import pandera.xarray as pa
import xarray as xr
from pandera import check


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

    @classmethod
    def dims(cls) -> tuple[str, ...]:
        return cls._dims

    @classmethod
    def append_dim(cls) -> str:
        return cls._dims[0]

    @check(
        "total_cloud_cover_atmosphere",
        "high_cloud_cover",
        "medium_cloud_cover",
        "low_cloud_cover",
        "relative_humidity_2m",
        regex=True,
    )
    def _is_percent_not_fraction(cls, da: xr.DataArray) -> bool:
        """Checks whether the max of the data array is greater than 1.0 or equal to 0.0.
        If max is (0, 1.0], it's highly likely the data unit is the Unit Interval, not a percentage.
        """
        # If the variable doesn't exist on the subclass (regex match skipped), just pass.
        return float(da.max()) > 1.0 or float(da.max()) == 0.0

    class Config:
        strict = "filter"  # Drops unlisted variables
        strict_coords = "filter"  # Drops unlisted coordinates
