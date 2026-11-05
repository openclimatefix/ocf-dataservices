import datetime as dt
from collections.abc import Sequence
from typing import Final

import xarray as xr
from storage.cli import ProductSpec, run_cli

from .ecmwf_ens.download import download, open_it
from .ecmwf_ens.schema import DynamicalEcmwfEnsSchema

UK_BBOX_NWSE: Final[tuple[float, float, float, float]] = (62, -12, 48, 3)


def _fetch_ecmwf_ens_uk(init_time: dt.datetime) -> xr.Dataset:
    ds_lazy = open_it(nwp_init_time=init_time, bbox_nwse=UK_BBOX_NWSE)
    return download(ds_lazy)


PRODUCTS: dict[str, ProductSpec] = {
    "ens_uk": ProductSpec(
        schema=DynamicalEcmwfEnsSchema,
        fetch=_fetch_ecmwf_ens_uk,
        description="Dynamical.org ECMWF ENS UK subset",
    ),
}


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the standalone `dynamical-data` CLI."""
    run_cli("dynamical-data", PRODUCTS, argv=argv)


if __name__ == "__main__":
    main()
