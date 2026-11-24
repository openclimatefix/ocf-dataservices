# schemas

Shared [Pandera](https://pandera.readthedocs.io/) field builders, coordinate helpers, and the base
model (`NwpDatasetSchema`) used to validate NWP `xarray.Dataset` objects across all provider
packages.

## Validation scope

A subclass of `NwpDatasetSchema` validates:

- **Dimensions and shape**: The dataset dimension order matches `_dims`, and no dimension is empty.
- **Coordinates**: Dimension coordinates and `init_time` metadata coordinates exist with the
  declared data types and monotonic order.
- **Nullability**: Non-nullable variables contain no `NaN` values.
- **Physical unit bounds**: Variables fall within physical ranges (for example, temperature in
  degrees Celsius, cloud cover and relative humidity in percentages `[0, 100]`, and non-cumulative
  radiation and precipitation rates).

## Regional slicing responsibility

`latitude` and `longitude` coordinates are validated against global bounds (`[-90, 90]` and
`[-180, 180]`), not asset-specific bounding boxes. Provider packages must slice spatial regions
before schema validation runs.

## Shared components

- `schemas.base.NwpDatasetSchema`: Base `pa.DatasetModel` that enforces `_dims`, `_chunks`, and
  `_shards` declarations and checks dimension order, monotonic coordinates, percentage scaling, and
  de-accumulated flux rates.
- `schemas.dim_order.enforce_dim_order`: Rebuilds an `xarray.Dataset` with an explicit dimension
  order while preserving auxiliary coordinates.
- `schemas.validation.validates`: Decorator that validates a function's returned `xarray.Dataset`
  against a schema and reports per-variable null counts on failure.
- `schemas.validation.is_valid_grib_file`: Verifies that a local file is at least 100 KB, starts
  with `b"GRIB"`, and ends with `b"7777"`.
