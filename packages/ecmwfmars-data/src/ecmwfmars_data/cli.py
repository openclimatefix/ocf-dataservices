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

UK_BBOX_NWSE: Final[list[int]] = [62, -12, 48, 3]
ENS_STEPS: Final[list[int]] = list(range(86))
ENS_NUMBERS: Final[list[int]] = list(range(1, 51))


def _fetch_mars_ens_uk(init_time: dt.datetime) -> xr.Dataset:
    url = os.getenv("ECMWF_API_URL", "")
    key = os.getenv("ECMWF_API_KEY", "")
    email = os.getenv("ECMWF_API_EMAIL", "")
    if not (url and key and email):
        raise RuntimeError(
            "ECMWF_API_URL, ECMWF_API_KEY, and ECMWF_API_EMAIL environment variables are required."
        )

    client = MarsClient(url=url, key=key, email=email)
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


PRODUCTS: dict[str, ProductSpec] = {
    "ens_uk": ProductSpec(
        schema=MarsEcmwfEnsSchema,
        fetch=_fetch_mars_ens_uk,
        description="ECMWF MARS ENS UK subset",
    ),
}


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the standalone `ecmwfmars-data` CLI."""
    run_cli("ecmwfmars-data", PRODUCTS, argv=argv)


if __name__ == "__main__":
    main()
