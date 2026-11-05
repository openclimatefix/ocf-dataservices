import datetime as dt
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import xarray as xr
from storage.cli import ProductSpec, run_cli

from .client import DatahubClient
from .download import download_order_partition
from .processing import (
    process_metoffice_india,
    process_metoffice_ukv,
    process_metoffice_westeurope,
)
from .schema import (
    MetOfficeGlobalIndiaSchema,
    MetOfficeGlobalWesteuropeSchema,
    MetOfficeUkvSchema,
)

WESTEUROPE_ORDER_ID: Final[str] = "westeurope-12params-54steps"
INDIA_ORDER_ID: Final[str] = "india-11params-54steps"
UKV_ORDER_ID: Final[str] = "uk-12params-42steps"


def _make_fetcher(
    order_id: str,
    process_fn: Callable[[Path | str], xr.Dataset],
) -> Callable[[dt.datetime], xr.Dataset]:
    def _fetch(init_time: dt.datetime) -> xr.Dataset:
        api_key = os.getenv("METOFFICE_API_KEY", "")
        if not api_key:
            raise RuntimeError("METOFFICE_API_KEY environment variable is required.")

        client = DatahubClient(api_key=api_key)
        with tempfile.TemporaryDirectory() as tmp_dir:
            grib_path = Path(tmp_dir) / f"metoffice_{order_id}_{init_time:%Y%m%d%H%M}.grib"
            num_files = download_order_partition(
                client=client,
                order_id=order_id,
                init_time=init_time,
                out_path=grib_path,
            )
            if num_files == 0:
                raise RuntimeError(
                    f"No MetOffice files available for order '{order_id}' at {init_time.isoformat()}."
                )
            return process_fn(grib_path)

    return _fetch


PRODUCTS: dict[str, ProductSpec] = {
    "global_west_europe": ProductSpec(
        schema=MetOfficeGlobalWesteuropeSchema,
        fetch=_make_fetcher(WESTEUROPE_ORDER_ID, process_metoffice_westeurope),
        description="MetOffice Global West Europe order",
    ),
    "global_india": ProductSpec(
        schema=MetOfficeGlobalIndiaSchema,
        fetch=_make_fetcher(INDIA_ORDER_ID, process_metoffice_india),
        description="MetOffice Global India order",
    ),
    "ukv_uk": ProductSpec(
        schema=MetOfficeUkvSchema,
        fetch=_make_fetcher(UKV_ORDER_ID, process_metoffice_ukv),
        description="MetOffice UKV UK order",
    ),
}


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the standalone `metoffice-data` CLI."""
    run_cli("metoffice-data", PRODUCTS, argv=argv)


if __name__ == "__main__":
    main()
