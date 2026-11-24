# ecmwflive-data

Downloads and validates ECMWF real-time GRIB dissemination data from S3 for the West Europe and
India regions.

## Products

| Product | Bounding box (`[N, W, S, E]`) | Steps | Schema |
|---|---|---|---|
| `hres_west_europe` | `[63, -12, 35, 26]` | 0–84h | `EcmwfLiveSchema` |
| `hres_india` | `[35, 67, 6, 97]` | 0–84h | `EcmwfLiveSchema` |

## Layout

- `schema.py`: `EcmwfLiveSchema`.
- `download.py`: `get_completed_init_times` and `download_live_ecmwf_partition`.
- `processing.py`: `_read_raw_grib`, `_transform`, `process_ecmwf_live`, and
  `process_ecmwf_live_validated`.
- `cli.py`: Standalone CLI entrypoint (`ecmwflive-data`).
