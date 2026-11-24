# dynamical-data

Downloads and validates ECMWF ENS forecast data from [Dynamical.org](https://dynamical.org)'s
public Icechunk/Zarr archive.

## Layout

- `ecmwf_ens/schema.py`: `DynamicalEcmwfEnsSchema`, the Pandera schema for the 50-member ensemble
  dataset.
- `ecmwf_ens/download.py`: `open_it` (lazy open and spatial/temporal slice) and `download`
  (parallel variable fetch, forward rate alignment, and schema validation).
- `cli.py`: Standalone CLI entrypoint (`dynamical-data --product ens_uk`).
