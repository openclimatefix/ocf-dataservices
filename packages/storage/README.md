# storage

Shared Icechunk storage writer (`storage.store`) and standalone CLI runner (`storage.cli`) used by
both Dagster (`XarrayIcechunkIOManager`) and the standalone `*-data` package entrypoints.

## Components

- `storage.store.write_to_icechunk`: Applies Veltkamp mantissa bit-rounding, writes or appends new
  `init_time` partitions to a Zarr v3 dataset in an Icechunk repository (local disk or S3),
  optionally updates the monthly `stats` Zarr group using parallel variance, and retries on
  concurrent Icechunk commit conflicts.
- `storage.store.existing_partition` / `storage.store.has_partition`: Checks whether an `init_time`
  partition already exists in the target Icechunk repository before downloading upstream data.
- `storage.cli.run_cli`: Provides a unified command-line interface (`--product`, `--store-path`,
  `--init-time`, `--start`, `--end`, `--step-hours`, `--write-stats`) for all `*-data` packages.
