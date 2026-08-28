# ecmwfmars-data

Downloads ECMWF ENS forecast data via [ECMWF's MARS archive
API](https://www.ecmwf.int/en/forecasts/datasets/archive-datasets) and converts the raw GRIB
output into a validated L1-ready `xr.Dataset`.

MARS access requires an authenticated ECMWF account; request parameters follow [MARS request
syntax](https://confluence.ecmwf.int/display/UDOC/MARS+request+syntax). Requests are built and
submitted by `ens/client.py`, and the resulting GRIB file is persisted as this product's L0 asset
(`l0_mars_ecmwf_ens_uk_v1`) before being parsed.

## Layout

- `ens/client.py` — `MarsClient`/`MarsRequest`, a thin wrapper around the MARS API.
- `ens/schema.py` — `MarsEcmwfEnsSchema`, the pandera schema this product validates against.
- `ens/download.py` — `download_raw` (L0 fetch), plus `_read_raw_grib` / `_transform` /
  `convert_to_dataset` (L1 fetch/transform/validate), split as described in the root README.

See the [repository README](../../README.md) for the general data module structure and asset
tiers, and the [`schemas` package](../schemas/README.md) for what validation does and doesn't
cover.
