# ecmwflive-data

Ingests ECMWF's real-time GRIB dissemination stream — delivered to an S3 bucket ahead of the
public archive — for [ECMWF's HRES/ENS live forecasts](https://www.ecmwf.int/en/forecasts), and
validates it into L1-ready `xr.Dataset`s for the west-europe and India regions.

A sensor (`ecmwf_live_s3_sensor`) watches the dissemination S3 bucket for newly-completed
init times and triggers materialization; the GRIB file for that partition is persisted as the L0
asset (`l0_ecmwf_live_local`) before being parsed. Both regions share a single schema
(`EcmwfLiveSchema`) — same 84h forecast horizon and variable set — differing only in their
bounding box.

## Layout

- `schema.py` — `EcmwfLiveSchema`, the pandera schema these products validate against.
- `download.py` — S3 listing (`get_completed_init_times`) and fetch
  (`download_live_ecmwf_partition`) for the L0 asset.
- `processing.py` — `_read_raw_grib` / `_transform` (region-agnostic parsing), and
  `process_ecmwf_live_validated` (the fetch/transform/validate entrypoint), split as described in
  the root README.

See the [repository README](../../README.md) for the general data module structure and asset
tiers, and the [`schemas` package](../schemas/README.md) for what validation does and doesn't
cover.
