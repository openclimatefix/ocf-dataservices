# ocf-dataservices

Consolidates the external data sources [Open Climate Fix](https://openclimatefix.org)'s ML
pipelines depend on — currently various flavours of ECMWF NWP data — into a single, validated,
consistently-shaped store. Each source has its own access pattern, format, and quirks (a Zarr
archive, the ECMWF MARS API, real-time GRIB dissemination over S3); this repo standardizes
ingestion of all of them so downstream ML consumers read one queryable format and don't need to
know anything about where the data came from.

Pipelines are orchestrated with [Dagster](https://dagster.io).

## Architecture

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) split into two
kinds of project:

- **`src/ocf_dataservices`** (the root project) is the Dagster orchestration layer, and only that:
  asset/sensor definitions (`defs/`), IO managers and resources (`resources/`), wired together in
  `definitions.py`. It has no data-provider-specific parsing logic of its own.
- **`packages/*`** are independent "data modules", one per external data source, each an
  independent workspace member with its own dependencies (e.g. `cfgrib`, `ecmwf-api-client`,
  `dynamical-catalog`). Keeping them separate means a provider-specific dependency (or a breaking
  change to one provider's parsing logic) can't leak into the orchestration layer or into other
  providers.
- **`packages/schemas`** is a shared validation library used by every data module. It has no
  Dagster dependency and no provider-specific logic — just a common `pandera` schema base class and
  helpers (see [`packages/schemas/README.md`](packages/schemas/README.md) for what a schema does
  and doesn't validate).

Dependencies flow one way: `ocf_dataservices` (Dagster) → data modules (`packages/*-data`) →
`schemas`. `schemas` has no dependents inside itself.

Current data modules:

| Package | Source | Product |
|---|---|---|
| `dynamical-data` | [Dynamical.org](https://dynamical.org)'s public Zarr archive | ECMWF ENS |
| `ecmwfmars-data` | ECMWF's MARS API | ECMWF ENS |
| `ecmwflive-data` | ECMWF's real-time GRIB dissemination (S3) | ECMWF HRES/ENS live, west-europe/India |
| `metoffice-data` | MetOffice Weather DataHub order API | UM global (west-europe/India), UKV |

## Asset tiers: L0 / L1 / L2

Assets are organized into tiers, mirroring how far the data is from its raw, as-received form:

- **L0 — raw.** Data exactly as received from the source (e.g. a downloaded GRIB file), stored
  verbatim via `RawFileIOManager`, with no parsing or validation. This exists so that a failed or
  changed downstream transform can be replayed without re-fetching from a source that may be
  rate-limited, ephemeral, or paid for. Not every data module needs an L0 asset: `dynamical-data`
  reads lazily from an existing external Zarr archive and only fetches at L1 time, so there's
  nothing raw to persist first.
- **L1 — validated.** Parsed into an `xarray.Dataset`, renamed to canonical variable names,
  unit-converted, and validated against a `pandera` schema declaring shape, dtype, nullability, and
  physical unit bounds. Stored as chunked/sharded Zarr in an Icechunk repository
  (`XarrayIcechunkIOManager`), one dataset per asset, appended along `init_time` per Dagster
  partition. **This is the tier ML pipelines are expected to read from.**
- **L2 — derived.** Reserved for further-derived data (e.g. blending multiple providers together,
  feature engineering, resampling). The `l2_io_manager` resource is already wired up in
  `definitions.py`, but no L2 assets exist yet — add them the same way as L1 assets, described
  below, when needed.

## Structure of a data module

Every data module follows the same shape. Using `ecmwfmars-data` as an example:

```
packages/<provider>-data/
  pyproject.toml            # provider-specific deps + `schemas`
  README.md
  src/<provider>_data/
    <product>/
      schema.py              # pandera schema: shape, dtype, bounds, chunk/shard layout
      download.py            # fetch -> transform -> validate
      client.py               # optional: auth/API client wrapper, if the provider needs one
  tests/
    <product>/
      test_*.py
```

### `download.py` (or `processing.py`): fetch, transform, validate

Each module's public entrypoint is a thin function composed of three separable pieces:

1. **Fetch** (e.g. `_read_raw_grib`, `_compute`) — the only part that does I/O: a network request,
   an API call, a disk read. Returns data close to its native shape, with the provider's own
   variable/coordinate names. No renaming, no unit conversion.
2. **Transform** (`_transform`) — a pure function, no I/O: renames variables/coordinates to the
   schema's canonical names, does unit conversion, spatial/temporal slicing, and finishes by calling
   `schemas.dim_order.enforce_dim_order()` to produce the schema's declared dimension order. Because
   it's pure, it can be unit-tested with a small synthetic in-memory `xr.Dataset` — no network, no
   real GRIB/Zarr files required.
3. **The public entrypoint** — composes fetch + transform and is decorated with
   `@schemas.validates(YourSchema)`. This is the *only* function Dagster assets (or anything else)
   should call. The decorator guarantees every dataset that leaves the module has already passed
   schema validation — never validate at the Dagster asset call site.

Real examples of this split:

- `dynamical_data.ecmwf_ens.download`: `_compute` → `_to_dataset` → `download`
- `ecmwfmars_data.ens.download`: `_read_raw_grib` → `_transform` → `convert_to_dataset`
- `ecmwflive_data.processing`: `_read_raw_grib` → `_transform` → `process_ecmwf_live_validated`
- `metoffice_data.processing`: `_read_raw_grib` → `_transform` → `process_metoffice_westeurope` /
  `process_metoffice_india` / `process_metoffice_ukv`

### `schema.py`

Subclasses `schemas.base.NwpDatasetSchema` and declares:

- `_dims` — the dataset's dimension order.
- `_chunks` / `_shards` — the Zarr chunk/shard layout `XarrayIcechunkIOManager` writes with.
- One field per coordinate/variable, using the shared `schemas.nwp_coordinates` /
  `schemas.nwp_variables` builders (add a new builder there if the variable doesn't exist yet).

## Adding a new data module

1. Scaffold `packages/<provider>-data/`: a `pyproject.toml` (name, provider-specific deps, and
   `schemas` as a dependency), `src/<provider>_data/__init__.py`, and `tests/`.
2. Add `<provider>-data` to the root `pyproject.toml`'s `dependencies` and `[tool.uv.sources]`, then
   run `uv sync --all-packages`.
3. Write `schema.py`, subclassing `schemas.base.NwpDatasetSchema` as described above.
4. Write `download.py` with the fetch / transform / validate split: a fetch function, a pure
   `_transform`, and a public entrypoint decorated with `@schemas.validates(YourSchema)`.
5. Add tests under `tests/`, targeting `_transform` directly with a synthetic `xr.Dataset`, plus an
   end-to-end test of the public entrypoint with the fetch step mocked out.
6. Wire up Dagster assets in `src/ocf_dataservices/defs/`:
   - if the provider requires persisting a raw file before parsing, an L0 `@dg.asset` with
     `io_manager_key="l0_io_manager"` that returns the path to the fetched file;
   - an L1 `@dg.asset` with `io_manager_key="l1_io_manager"` that calls your module's validated
     entrypoint and returns `dg.Output(ds, metadata={"schema": YourSchema, ...})` — the `"schema"`
     metadata key is how `XarrayIcechunkIOManager` derives chunk/shard layout and the partition
     append dimension, so it must always be set;
   - if the provider needs an authenticated client, add a resource under
     `src/ocf_dataservices/resources/` (following `DagsterMarsClient`'s pattern) and wire it into
     `local_resources` in `definitions.py`.
7. Register the new assets (and any sensors) in `src/ocf_dataservices/definitions.py`.
8. Add any new environment variables the module needs to `.env` (see existing keys for the
   convention) and to wherever secrets are provisioned for deployment.
9. Run `make run` and materialize the new assets in the Dagster UI to check the whole path
   end-to-end.

## Development

- `make run` — starts the local Dagster UI/daemon (`uv run dg dev`), using `dagster_history/` for
  run storage.
- `make lint` — `ruff check --fix .`
- Tests live per-package under `packages/<name>/tests`, written as `unittest.TestCase`s;
  run with e.g. `uv run python -m unittest discover -s packages/<name>/tests`.
- `uv sync --all-packages` after adding or changing a workspace member's dependencies.

## Deployment

`Containerfile` builds a self-contained image (the installed venv only, no source tree) and runs
`dagster api grpc -m ocf_dataservices.definitions`, for use as a Dagster code location.

`docker-compose.yaml` runs the full stack — Postgres, the code location gRPC server
(`dagster-codeserver`), `dagster-webserver`, and `dagster-daemon` — with `dagster.yaml` and
`workspace.yaml` embedded as Docker `configs:` rather than separate host files. Runs are launched
via `dagster_docker.DockerRunLauncher`, so each run executes in its own short-lived container
rather than in-process in the daemon.

To deploy on a server:

1. Copy `docker-compose.yaml` to `/etc/dagster/docker-compose.yaml`.
2. Create `/etc/dagster/.env` with the variables the pipelines need (`L0_ROOT_PATH`,
   `L1_ROOT_PATH`, `L2_ROOT_PATH`, `ECMWF_API_KEY`, `ECMWF_API_EMAIL`, `ECMWF_API_URL`,
   `ECMWF_REALTIME_S3_BUCKET`, `ECMWF_REALTIME_S3_ACCESS_KEY`, `ECMWF_REALTIME_S3_ACCESS_SECRET`,
   `METOFFICE_API_KEY`, plus any added per the "Adding a new data module" checklist above). This file is loaded three
   ways: as `env_file:` for the three core services, for Postgres credential substitution in the
   compose file itself, and — mounted read-only into every launched run container — via the
   `secrets: EnvFileLoader` configured in the embedded `dagster.yaml`, so new variables only need
   adding here, not to any run-launcher config.
3. Install `deploy/dagster.service` as a systemd unit (e.g. `/etc/systemd/system/dagster.service`),
   then `systemctl daemon-reload && systemctl enable --now dagster.service`. It runs
   `docker compose -f /etc/dagster/docker-compose.yaml` with `WorkingDirectory=/etc/dagster`, so
   the compose file's relative `.env` reference resolves correctly.
