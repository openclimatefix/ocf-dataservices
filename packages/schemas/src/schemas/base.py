from typing import ClassVar

import pandera.xarray as pa


class NwpDatasetSchema(pa.DatasetModel):
    """Shared base for NWP dataset schemas.

    Concrete schemas declare `_dims`, `_chunks`, and `_shards`; `_dims[0]` is the convention used
    for the append dimension of partitioned Icechunk writes, so it doesn't need repeating.

    Scope: a schema validates a dataset's shape (declared dims/coords present, in the declared
    dtype), nullability, and physical unit bounds (e.g. a temperature field's plausible range).
    `latitude`/`longitude` fields are checked against global bounds (-90/90, -180/180) as a basic
    sanity check, not as a stand-in for an asset's actual region.

    Geographic region (the `bbox_nwse` a given asset covers) is deliberately NOT a schema concern:
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

    class Config:
        strict = "filter"  # Drops unlisted variables
        strict_coords = "filter"  # Drops unlisted coordinates
