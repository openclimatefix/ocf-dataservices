import datetime as dt
from pathlib import Path

from schemas.validation import is_valid_grib_file

from .client import DatahubClient


def download_order_partition(
    client: DatahubClient,
    order_id: str,
    init_time: dt.datetime,
    out_path: Path,
) -> int:
    """Download all GRIB files for an order initialization time into a single local file.

    Concatenates per-parameter GRIB files in sequence so `cfgrib` can open the multi-message file in
    one pass.

    Returns:
        Number of files downloaded, or `0` if the initialization time is not yet available.
    """
    if is_valid_grib_file(out_path):
        return 1

    file_ids = client.list_file_ids(order_id=order_id, init_time=init_time)
    if not file_ids:
        return 0

    with out_path.open("wb") as out_f:
        for file_id in file_ids:
            client.download_file(order_id=order_id, file_id=file_id, target=out_f)

    if not is_valid_grib_file(out_path):
        out_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file {out_path} for order '{order_id}' is not a valid GRIB file."
        )

    return len(file_ids)
