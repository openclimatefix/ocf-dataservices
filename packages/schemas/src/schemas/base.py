from typing import ClassVar

import pandera.xarray as pa


class NwpDatasetSchema(pa.DatasetModel):
    """Shared base for NWP dataset schemas.

    Concrete schemas declare `_dims`, `_chunks`, and `_shards`; `_dims[0]` is the convention used
    for the append dimension of partitioned Icechunk writes, so it doesn't need repeating.
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
