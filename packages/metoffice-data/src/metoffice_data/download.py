import datetime as dt
from pathlib import Path

from .client import DatahubClient


def download_order_partition(
    client: DatahubClient,
    order_id: str,
    init_time: dt.datetime,
    out_path: Path,
) -> int:
    """Download all GRIB files for an order's init time, concatenated into a single local file.

    GRIB files are a sequence of self-describing messages, so appending the per-parameter files
    into one file yields a valid multi-message GRIB that cfgrib can open in one pass.

    Returns the number of files downloaded (0 if the init time isn't available yet).
    """
    file_ids = client.list_file_ids(order_id=order_id, init_time=init_time)
    if not file_ids:
        return 0

    with out_path.open("wb") as out_f:
        for file_id in file_ids:
            client.download_file(order_id=order_id, file_id=file_id, target=out_f)

    return len(file_ids)
