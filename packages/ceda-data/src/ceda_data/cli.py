import datetime as dt
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import xarray as xr
from storage.cli import ProductSpec, run_cli

from .client import CedaClient
from .download import download_global_partition, download_ukv_partition
from .processing import process_ceda_global_west_europe, process_ceda_ukv
from .schema import CedaGlobalWestEuropeSchema, CedaUkvSchema


def _get_client() -> CedaClient:
    username = os.getenv("CEDA_USER", "")
    password = os.getenv("CEDA_PASS", "")
    if not (username and password):
        raise RuntimeError("CEDA_USER and CEDA_PASS environment variables are required.")
    return CedaClient(username=username, password=password)


def _make_fetcher(
    name: str,
    download_fn: Callable[[CedaClient, dt.datetime, Path], int],
    process_fn: Callable[[Path | str], xr.Dataset],
) -> Callable[[dt.datetime], xr.Dataset]:
    def _fetch(init_time: dt.datetime) -> xr.Dataset:
        client = _get_client()
        with tempfile.TemporaryDirectory() as tmp_dir:
            grib_path = Path(tmp_dir) / f"ceda_{name}_{init_time:%Y%m%d%H%M}.grib"
            download_fn(client, init_time, grib_path)
            return process_fn(grib_path)

    return _fetch


PRODUCTS: dict[str, ProductSpec] = {
    "ukv_uk": ProductSpec(
        schema=CedaUkvSchema,
        fetch=_make_fetcher("ukv_uk", download_ukv_partition, process_ceda_ukv),
        description="CEDA MetOffice UKV 2km OSGB archive",
    ),
    "global_west_europe": ProductSpec(
        schema=CedaGlobalWestEuropeSchema,
        fetch=_make_fetcher(
            "global_west_europe",
            download_global_partition,
            process_ceda_global_west_europe,
        ),
        description="CEDA MetOffice Global 17km archive cropped to West Europe",
    ),
}


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the standalone `ceda-data` CLI."""
    run_cli("ceda-data", PRODUCTS, argv=argv)


if __name__ == "__main__":
    main()
