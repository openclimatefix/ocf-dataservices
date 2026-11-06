import datetime as dt
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import xarray as xr
from storage.cli import ProductSpec, run_cli

from .ens.client import MarsClient
from .ens.download import convert_to_dataset, download_raw
from .ens.schema import MarsEcmwfEnsSchema
from .hres.download import (
    HRES_STEPS,
    HRES_WEST_EUROPE_BBOX_NWSE,
    convert_hres_to_dataset,
    download_raw_hres,
)
from .hres.schema import MarsEcmwfHresSchema

UK_BBOX_NWSE: Final[list[int]] = [62, -12, 48, 3]
ENS_STEPS: Final[list[int]] = list(range(86))
ENS_NUMBERS: Final[list[int]] = list(range(1, 51))


def _get_mars_client() -> MarsClient:
    url = os.getenv("ECMWF_API_URL", "")
    key = os.getenv("ECMWF_API_KEY", "")
    email = os.getenv("ECMWF_API_EMAIL", "")
    if not (url and key and email):
        raise RuntimeError(
            "ECMWF_API_URL, ECMWF_API_KEY, and ECMWF_API_EMAIL environment variables are required."
        )
    return MarsClient(url=url, key=key, email=email)


def _fetch_mars_ens_uk(init_time: dt.datetime) -> xr.Dataset:
    client = _get_mars_client()
    with tempfile.TemporaryDirectory() as tmp_dir:
        grib_path = Path(tmp_dir) / f"mars_ens_{init_time:%Y%m%d%H%M}.grib"
        download_raw(
            client=client,
            init_time=init_time,
            bbox_nwse=UK_BBOX_NWSE,
            steps=ENS_STEPS,
            numbers=ENS_NUMBERS,
            target_path=grib_path,
        )
        return convert_to_dataset(grib_path)


def _fetch_mars_hres_west_europe(init_time: dt.datetime) -> xr.Dataset:
    client = _get_mars_client()
    with tempfile.TemporaryDirectory() as tmp_dir:
        grib_path = Path(tmp_dir) / f"mars_hres_west_europe_{init_time:%Y%m%d%H%M}.grib"
        download_raw_hres(
            client=client,
            init_time=init_time,
            bbox_nwse=HRES_WEST_EUROPE_BBOX_NWSE,
            steps=HRES_STEPS,
            target_path=grib_path,
        )
        return convert_hres_to_dataset(grib_path)


PRODUCTS: dict[str, ProductSpec] = {
    "ens_uk": ProductSpec(
        schema=MarsEcmwfEnsSchema,
        fetch=_fetch_mars_ens_uk,
        description="ECMWF MARS ENS UK subset",
    ),
    "hres_west_europe": ProductSpec(
        schema=MarsEcmwfHresSchema,
        fetch=_fetch_mars_hres_west_europe,
        description="ECMWF MARS HRES West Europe subset (matching live dimensions and variables)",
    ),
}


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the standalone `ecmwfmars-data` CLI."""
    run_cli("ecmwfmars-data", PRODUCTS, argv=argv)


if __name__ == "__main__":
    main()
