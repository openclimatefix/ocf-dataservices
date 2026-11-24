# ceda-data

Downloads and validates historical MetOffice Unified Model (UM) forecasts from the
[CEDA Archive](https://catalogue.ceda.ac.uk/uuid/f47bc62786394626b665e23b658d385f) (`ukv-grib` and
`global-grib`).

## Products

| Product | CEDA path | Grid | Steps | Schema |
|---|---|---|---|---|
| `ukv_uk` | `ukv-grib` (`Wholesale1`, `Wholesale1T54`, `Wholesale2`, `Wholesale2T54`) | OSGB 2km (`y_osgb` 704 × `x_osgb` 548) | 0–54h | `CedaUkvSchema` |
| `global_west_europe` | `global-grib` (`WSGlobal17km`, `AreaA`) | Regular lat/lon ~17km cropped to `[63, -12, 35, 26]` | 0–144h | `CedaGlobalWestEuropeSchema` |

## Layout

- `client.py`: `CedaClient` exchanges `CEDA_USER` and `CEDA_PASS` for a Bearer token via
  `https://services.ceda.ac.uk/api/token/create/` and downloads GRIB files with retry.
- `download.py`: `download_ukv_partition` and `download_global_partition` fetch and concatenate
  per-run GRIB files into a single local file.
- `schema.py`: `CedaUkvSchema` and `CedaGlobalWestEuropeSchema`.
- `processing.py`: `_read_raw_grib`, `_transform`, `process_ceda_ukv`, and
  `process_ceda_global_west_europe`.
- `cli.py`: Standalone CLI entrypoint (`ceda-data`).
