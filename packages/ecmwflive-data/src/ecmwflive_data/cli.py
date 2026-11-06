import datetime as dt
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import xarray as xr
from storage.cli import ProductSpec, run_cli

from .download import download_live_ecmwf_partition, get_completed_init_times
from .processing import process_ecmwf_live_validated
from .schema import EcmwfLiveSchema


def _make_fetcher(
    bbox_nwse: list[float],
    max_step_hours: int,
) -> Callable[[dt.datetime], xr.Dataset]:
    def _fetch(init_time: dt.datetime) -> xr.Dataset:
        bucket = os.getenv("ECMWF_REALTIME_S3_BUCKET", "")
        if not bucket:
            raise RuntimeError("ECMWF_REALTIME_S3_BUCKET environment variable is required.")

        access_key = os.getenv("ECMWF_REALTIME_S3_ACCESS_KEY", "")
        secret = os.getenv("ECMWF_REALTIME_S3_ACCESS_SECRET", "")
        endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        region_name = os.getenv("ECMWF_REALTIME_S3_REGION", "eu-west-1")
        prefix = os.getenv("ECMWF_REALTIME_DISSEMINATION_FILE_PREFIX", "A2")

        completed = get_completed_init_times(
            bucket=bucket,
            access_key=access_key,
            secret=secret,
            endpoint_url=endpoint_url,
            region_name=region_name,
            prefix=prefix,
            last_processed=init_time - dt.timedelta(seconds=1),
        )
        if init_time not in completed:
            raise RuntimeError(
                f"ECMWF live run for {init_time.isoformat()} is not complete in S3 bucket '{bucket}'."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            grib_path = Path(tmp_dir) / f"ecmwf_live_{init_time:%Y%m%d%H%M}.grib"
            num_files = download_live_ecmwf_partition(
                partition_time=init_time,
                out_path=grib_path,
                bucket=bucket,
                access_key=access_key,
                secret=secret,
                endpoint_url=endpoint_url,
                region_name=region_name,
                prefix=prefix,
            )
            if num_files == 0:
                raise RuntimeError(
                    f"No ECMWF live files found in S3 bucket '{bucket}' for {init_time.isoformat()}."
                )
            return process_ecmwf_live_validated(
                grib_path=grib_path,
                bbox_nwse=bbox_nwse,
                max_step_hours=max_step_hours,
            )

    return _fetch


PRODUCTS: dict[str, ProductSpec] = {
    "hres_west_europe": ProductSpec(
        schema=EcmwfLiveSchema,
        fetch=_make_fetcher(bbox_nwse=[63, -12, 35, 26], max_step_hours=84),
        description="ECMWF HRES Live West Europe region",
    ),
    "hres_india": ProductSpec(
        schema=EcmwfLiveSchema,
        fetch=_make_fetcher(bbox_nwse=[35, 67, 6, 97], max_step_hours=84),
        description="ECMWF HRES Live India region",
    ),
}


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the standalone `ecmwflive-data` CLI."""
    run_cli("ecmwflive-data", PRODUCTS, argv=argv)


if __name__ == "__main__":
    main()
