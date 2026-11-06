# ceda-data

Ingests historical MetOffice Unified Model (UM) forecasts from the
[CEDA Archive](https://catalogue.ceda.ac.uk/uuid/f47bc62786394626b665e23b658d385f) (`ukv-grib` and
`global-grib`) and validates them into L1-ready `xr.Dataset`s.

CEDA's archive differs from the live MetOffice Weather DataHub API (`metoffice-data`) in both grid
projection and variable set:

| Product | CEDA path | Grid | Steps | Schema |
|---|---|---|---|---|
| `ukv_uk` | `ukv-grib` (`Wholesale1`/`Wholesale1T54`/`Wholesale2`/`Wholesale2T54`) | OSGB 2km (`y_osgb` 704 × `x_osgb` 548) | 0–54 | `CedaUkvSchema` |
| `global_west_europe` | `global-grib` (`WSGlobal17km`, `AreaA`–`AreaH`) | Regular lat/lon ~17km cropped to West Europe `[63, -12, 35, 26]` | 0–144 | `CedaGlobalWestEuropeSchema` |

## Layout

- `client.py` — `CedaClient`: exchanges `CEDA_USER` / `CEDA_PASS` for a Bearer token via
  `https://services.ceda.ac.uk/api/token/create/` and streams GRIB files with retry.
- `download.py` — `download_ukv_partition` and `download_global_partition`.
- `schema.py` — `CedaUkvSchema` and `CedaGlobalWestEuropeSchema`.
- `processing.py` — `_read_raw_grib`, `_transform_ukv`, `_transform_global`, and the validated
  entrypoints `process_ceda_ukv` and `process_ceda_global_west_europe`.
- `cli.py` — standalone container CLI entrypoint (`ceda-data`).
