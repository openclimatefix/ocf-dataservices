# dynamical-data

Fetches ECMWF ENS forecast data from [Dynamical.org](https://dynamical.org)'s public Zarr
archive and validates it into an L1-ready `xr.Dataset`.

Dynamical.org re-publishes ECMWF's ensemble forecast as a continuously-updated Zarr store, so
unlike the other data modules in this workspace there's no raw file to fetch and persist first —
`ecmwf_ens/download.py` opens the archive lazily (`open_it`), computes just the requested
init-time/region slice into memory, and validates the result, all as part of the single L1 asset
(`l1_dynamical_ecmwf_ens_uk_v1` in `ocf_dataservices.defs.assets`).

## Layout

- `ecmwf_ens/schema.py` — `DynamicalEcmwfEnsSchema`, the pandera schema this product validates
  against.
- `ecmwf_ens/download.py` — `open_it` (lazy open + slice) and `download` (fetch + validate), split
  as described in the root README.

See the [repository README](../../README.md) for the general data module structure and asset
tiers, and the [`schemas` package](../schemas/README.md) for what validation does and doesn't
cover.
