import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import cfgrib
import xarray as xr
from schemas.ecmwf_hres import transform_ecmwf_hres
from schemas.validation import is_valid_grib_file, validates

from ..ens.client import MarsClient, MarsRequest
from .schema import MarsEcmwfHresSchema

# The 18 ECMWF MARS paramIds matching EcmwfLiveSchema:
# 165=10u, 166=10v, 228246=100u, 228247=100v, 167=2t, 228239=200u, 228240=200v,
# 47=dsrp, 57=uvb, 188=hcc, 186=lcc, 187=mcc, 260048=tprate, 141=sd, 175=strd,
# 169=ssrd, 164=tcc, 3020=vis
HRES_MARS_PARAMETERS: Final[list[str]] = [
    "165",
    "166",
    "228246",
    "228247",
    "167",
    "228239",
    "228240",
    "47",
    "57",
    "188",
    "186",
    "187",
    "260048",
    "141",
    "175",
    "169",
    "164",
    "3020",
]
HRES_WEST_EUROPE_BBOX_NWSE: Final[list[int]] = [63, -12, 35, 26]
HRES_STEPS: Final[list[int]] = list(range(86))


def download_raw_hres(
    client: MarsClient,
    init_time: dt.datetime,
    bbox_nwse: list[int],
    steps: list[int],
    target_path: Path | str,
) -> None:
    """Download raw ECMWF MARS HRES data to a local GRIB file."""
    if is_valid_grib_file(target_path):
        return

    request = MarsRequest.hres(
        params=HRES_MARS_PARAMETERS,
        init_time=init_time,
        steps=steps,
        bbox_nwse=bbox_nwse,
    )

    with open(target_path, "wb") as f:
        client.execute(request, f)

    if not is_valid_grib_file(target_path):
        Path(target_path).unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file {target_path} is not a valid GRIB file.")


def _read_raw_hres_grib(grib_path: Path | str) -> xr.Dataset:
    """Read a raw MARS HRES GRIB file into an xarray Dataset."""
    try:
        datasets = cfgrib.open_datasets(
            grib_path,
            backend_kwargs={
                "indexpath": "",
                "read_keys": ["generatingProcessIdentifier", "expver", "tablesVersion"],
                "ignore_keys": [
                    "valid_time",
                    "stepType",
                    "typeOfLevel",
                    "surface",
                    "heightAboveGround",
                    "meanSea",
                ],
            },
        )
    except Exception as e:
        raise ValueError(f"Failed to open datasets from {grib_path}: {e}") from e

    if not datasets:
        raise ValueError(f"No datasets found in {grib_path}")

    return xr.merge(datasets, compat="override", combine_attrs="override")


@validates(MarsEcmwfHresSchema)
def convert_hres_to_dataset(
    grib_path: Path | str,
    bbox_nwse: Sequence[float] = HRES_WEST_EUROPE_BBOX_NWSE,
    max_step_hours: int = 84,
) -> xr.Dataset:
    """Convert a raw ECMWF MARS HRES GRIB file to an xarray Dataset matching MarsEcmwfHresSchema."""
    return transform_ecmwf_hres(
        _read_raw_hres_grib(grib_path),
        bbox_nwse=bbox_nwse,
        max_step_hours=max_step_hours,
    )
