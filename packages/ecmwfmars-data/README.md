# ecmwfmars-data

Downloads and validates ECMWF ENS and HRES forecast data from the
[ECMWF MARS archive API](https://www.ecmwf.int/en/forecasts/datasets/archive-datasets).

## Products

| Product | Stream / Type | Bounding box (`[N, W, S, E]`) | Steps | Schema |
|---|---|---|---|---|
| `ens_uk` | `enfo` / `pf` (50 members) | `[62, -12, 48, 3]` | 0–84h | `MarsEcmwfEnsSchema` |
| `hres_west_europe` | `oper` (00/12Z), `scda` (06/18Z) / `fc` | `[63, -12, 35, 26]` | 0–84h | `MarsEcmwfHresSchema` |

## Layout

- `ens/client.py`: `MarsClient` and `MarsRequest` for submitting, polling, and downloading MARS
  requests.
- `ens/schema.py` and `ens/download.py`: Schema, download, and conversion functions for ECMWF ENS.
- `hres/schema.py` and `hres/download.py`: Schema, download, and conversion functions for ECMWF
  HRES (matching the live dissemination dimensions and 18-variable schema).
- `cli.py`: Standalone CLI entrypoint (`ecmwfmars-data`).
