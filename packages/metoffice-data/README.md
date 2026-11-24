# metoffice-data

Downloads and validates forecast data from the
[MetOffice Weather DataHub](https://datahub.metoffice.gov.uk) atmospheric-models order API.

## Products

| Product | Order ID | Grid | Steps | Schema |
|---|---|---|---|---|
| `global_west_europe` | `westeurope-12params-54steps` | Regular lat/lon ~10km (`[63, -12, 35, 26]`) | 0–54h | `MetOfficeGlobalWesteuropeSchema` |
| `global_india` | `india-11params-54steps` | Regular lat/lon ~10km (`[35, 67, 6, 97]`) | 0–54h | `MetOfficeGlobalIndiaSchema` |
| `ukv_uk` | `uk-12params-42steps` | LAEA 2km (`y_laea` 639 × `x_laea` 455) | 0–42h | `MetOfficeUkvSchema` |

## Layout

- `client.py`: `DatahubClient` for listing and downloading order GRIB files.
- `download.py`: `download_order_partition`, which concatenates per-parameter GRIB files into a
  single local file.
- `schema.py`: `MetOfficeGlobalWesteuropeSchema`, `MetOfficeGlobalIndiaSchema`, and
  `MetOfficeUkvSchema`.
- `processing.py`: `_read_raw_grib`, `_transform`, and the validated entrypoints
  `process_metoffice_westeurope`, `process_metoffice_india`, and `process_metoffice_ukv`.
- `cli.py`: Standalone CLI entrypoint (`metoffice-data`).
