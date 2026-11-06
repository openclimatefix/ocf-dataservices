# metoffice-data

Ingests the [MetOffice Weather DataHub](https://datahub.metoffice.gov.uk) atmospheric-models order
API and validates it into L1-ready `xr.Dataset`s. Data is delivered per order, one GRIB file per
parameter per step; this module lists an order's files for an init time, downloads them, and parses
them into a single validated dataset.

Three orders are consumed, each with its own asset pair and schema:

| Order | Model | Region | Init times | Steps | Schema |
|---|---|---|---|---|---|
| `westeurope-12params-54steps` | UM global ~10km | N63 E26 S35 W-12 (lat/lon) | 00/06/12/18 | 0–54 | `MetOfficeGlobalWesteuropeSchema` |
| `india-11params-54steps` | UM global ~10km | N35 E97 S6 W67 (lat/lon) | 00/12 | 0–54 | `MetOfficeGlobalIndiaSchema` |
| `uk-12params-42steps` | UM UKV ~2km | GB/EIRE (LAEA `x_laea`/`y_laea`) | 00/03/…/21 | 0–42 | `MetOfficeUkvSchema` |

The two global orders share the same lat/lon grid and differ only in region and whether pressure at
MSL is included; the UKV order is kept on the model's native Lambert Azimuthal Equal Area grid (not
remapped to lat/lon), so its schema carries projected `x_laea`/`y_laea` coordinates (metres). cfgrib
doesn't read those projected coordinate values reliably, so `_assign_ukv_coords` reassigns them from
the known grid arrays (639 × 455 at 2km), matched by size.

Radiation fluxes are left in W m⁻² as delivered (downward shortwave is the mean over the following
hour; downward longwave is instantaneous), with no accumulation differencing.

## Layout

- `schema.py` — the three pandera schemas these products validate against.
- `client.py` — `DatahubClient`: `list_file_ids` (the order `/latest` endpoint) and `download_file`.
- `download.py` — `download_order_partition`, which concatenates an init time's GRIB files into one
  local file for the L0 asset.
- `processing.py` — `_read_raw_grib` / `_transform` (parsing shared across orders), and
  `process_metoffice_westeurope` / `process_metoffice_india` / `process_metoffice_ukv` (the
  per-order fetch/transform/validate entrypoints).

Surface-adjusted wind parameters arrive from cfgrib as `unknown` and are disambiguated by their
GRIB2 `parameterNumber` (see the mapping in `processing.py`).

See the [repository README](../../README.md) for the general data module structure and asset tiers,
and the [`schemas` package](../schemas/README.md) for what validation does and doesn't cover.
