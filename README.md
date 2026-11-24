# ocf-dataservices

`ocf-dataservices` ingests, validates, and converts Numerical Weather Prediction (NWP) datasets
into a standardized [Icechunk](https://icechunk.io) (Zarr v3) storage format for Open Climate Fix
machine learning pipelines.

Each data module runs in two modes:
1. **Dagster mode**: Orchestrated archive backfills and scheduled runs via [Dagster](https://dagster.io).
2. **Standalone CLI mode**: Direct container execution outside Dagster using the shared CLI runner
   in `packages/storage`.

## Architecture

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with
three layers:

- **`src/ocf_dataservices`**: The Dagster orchestration layer. It defines assets (`defs/`), IO
  managers, and resources (`resources/`), and registers them in `definitions.py`. It contains no
  provider-specific parsing logic.
- **`packages/*-data`**: Provider data modules. Each module is an isolated workspace package with
  its own dependencies, schema, download client, transform logic, and standalone CLI entrypoint.
- **`packages/storage`**: Shared Icechunk storage writer (`storage.store`) and standalone CLI
  runner (`storage.cli`). Both Dagster (`XarrayIcechunkIOManager`) and the standalone package CLIs
  call `storage.store.write_to_icechunk`.
- **`packages/schemas`**: Shared [Pandera](https://pandera.readthedocs.io/) validation base model
  (`NwpDatasetSchema`), coordinate and variable field builders, dimension ordering helpers, and GRIB
  file verification.

Dependencies flow in one direction:
`ocf_dataservices` -> `packages/*-data` -> `packages/storage` -> `packages/schemas`.

### Provider packages

| Package | CLI command | `--product` values | Source |
|---|---|---|---|
| `ceda-data` | `ceda-data` | `ukv_uk`, `global_west_europe` | CEDA DAP MetOffice archive |
| `dynamical-data` | `dynamical-data` | `ens_uk` | Dynamical.org public Zarr archive |
| `ecmwflive-data` | `ecmwflive-data` | `hres_west_europe`, `hres_india` | ECMWF real-time S3 GRIB dissemination |
| `ecmwfmars-data` | `ecmwfmars-data` | `ens_uk`, `hres_west_europe` | ECMWF MARS archive API |
| `metoffice-data` | `metoffice-data` | `global_west_europe`, `global_india`, `ukv_uk` | MetOffice Weather DataHub order API |

## Asset tiers: L0 / L1 / L2

Dagster assets are organized into three tiers:

- **L0 (raw)**: Data stored in its original format (such as a multi-message GRIB file) via
  `RawFileIOManager`. Each L0 file is verified for minimum byte size (`>= 100 KB`), the leading
  `GRIB` magic header, and the trailing `7777` end-of-message marker.
- **L1 (validated)**: Data parsed into an `xarray.Dataset`, renamed to canonical variable names,
  converted to standard units, and validated against a `pandera` schema. `XarrayIcechunkIOManager`
  writes the dataset to an Icechunk repository and updates the monthly `stats` group. Machine
  learning pipelines read from this tier.
- **L2 (derived)**: Reserved for derived datasets such as multi-provider blends or resampled grids.

## Standalone container usage

The container image built from `Containerfile` installs all workspace packages and their CLI
entrypoints into `/app/.venv/bin`.

To run a single forecast initialization time in a container:

```bash
docker run --rm --env-file .env \
  ghcr.io/openclimatefix/ocf-dataservices:devsjc-rewrite \
  ecmwfmars-data \
  --product hres_west_europe \
  --init-time 2026-01-10T12:00 \
  --store-path s3://my-example-bucket/west-europe.icechunk
```

To run a range of forecast initialization times:

```bash
docker run --rm --env-file .env \
  ghcr.io/openclimatefix/ocf-dataservices:devsjc-rewrite \
  ceda-data \
  --product ukv_uk \
  --start 2024-01-01T00:00 \
  --end 2024-01-31T18:00 \
  --step-hours 6 \
  --store-path s3://my-example-bucket/ceda-ukv.icechunk
```

Add `--write-stats` to compute and update the monthly `stats` Zarr group during standalone CLI runs.

## Structure of a data module

Every data module follows a standard directory layout:

```
packages/<provider>-data/
  pyproject.toml            # Provider dependencies, `schemas`, `storage`, and `[project.scripts]`
  README.md
  src/<provider>_data/
    schema.py               # Pandera schema: dimensions, bounds, and chunk/shard layout
    download.py             # Raw file download functions
    processing.py           # `_read_raw_grib` -> `_transform` -> `@validates(Schema)` entrypoint
    client.py               # Optional API client wrapper
    cli.py                  # Standalone CLI entrypoint using `storage.cli.run_cli`
  tests/
    test_*.py
```

### Fetch, transform, and validate pattern

Each module separates data ingestion into three steps:

1. **Fetch** (`_read_raw_grib` or `_compute`): Performs all file or network I/O and returns raw
   `xarray` objects with native provider variable names.
2. **Transform** (`_transform`): Performs pure in-memory transformations. It extracts GRIB metadata
   attributes into `init_time` auxiliary coordinates, renames variables and coordinates, converts
   units, slices spatial and temporal bounds, and calls `schemas.dim_order.enforce_dim_order`.
3. **Validated entrypoint**: Composes the fetch and transform steps and applies the
   `@schemas.validates(Schema)` decorator. Callers must use this function to guarantee schema
   compliance.

## Adding a new data module

1. Create `packages/<provider>-data/` with `pyproject.toml`, `src/<provider>_data/__init__.py`, and
   `tests/`.
2. Add `<provider>-data` to the root `pyproject.toml` under `dependencies` and `[tool.uv.sources]`,
   then run `uv sync --all-packages`.
3. Define the product schema in `schema.py` by subclassing `schemas.base.NwpDatasetSchema`.
4. Implement the fetch, transform, and `@schemas.validates` functions in `download.py` and
   `processing.py`.
5. Implement `cli.py` using `storage.cli.ProductSpec` and `storage.cli.run_cli`, and register the
   script in `packages/<provider>-data/pyproject.toml` under `[project.scripts]`.
6. Add unit tests under `packages/<provider>-data/tests/` testing `_transform` with synthetic
   datasets and testing the validated entrypoint with mocked I/O.
7. If the module is orchestrated by Dagster, add L0 and L1 assets under
   `src/ocf_dataservices/defs/assets/` and register them in `src/ocf_dataservices/definitions.py`.

## Development

- `make run`: Start the local Dagster UI and daemon (`uv run dg dev`).
- `make lint`: Run Ruff (`uv run ruff check --fix .`).
- `make test`: Run the full unit test suite across `tests/` and `packages/*/tests`.
- `uv run ty check`: Run static type checking across the workspace.

## Deployment

`Containerfile` builds a self-contained image containing the virtual environment at `/app/.venv`.
By default, it starts the Dagster gRPC code server (`dagster api grpc -m ocf_dataservices.definitions`).

`docker-compose.yaml` runs the full Dagster service stack (`dagster-postgres`, `dagster-codeserver`,
`dagster-webserver`, and `dagster-daemon`). Runs execute in isolated containers via
`dagster_docker.DockerRunLauncher`.

To deploy on a host:

1. Copy `docker-compose.yaml` to `/etc/dagster/docker-compose.yaml`.
2. Create `/etc/dagster/.env` with required environment variables (`L0_ROOT_PATH`, `L1_ROOT_PATH`,
   `L2_ROOT_PATH`, `ECMWF_API_KEY`, `ECMWF_API_EMAIL`, `ECMWF_API_URL`, `CEDA_USER`, `CEDA_PASS`,
   `ECMWF_REALTIME_S3_BUCKET`, `ECMWF_REALTIME_S3_ACCESS_KEY`, `ECMWF_REALTIME_S3_ACCESS_SECRET`,
   `METOFFICE_API_KEY`).
3. Start the stack with `docker compose -f /etc/dagster/docker-compose.yaml up -d`.
