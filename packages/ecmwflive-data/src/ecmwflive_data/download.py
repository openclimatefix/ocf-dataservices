import concurrent.futures
import datetime as dt
import re
from collections.abc import Sequence
from pathlib import Path

import s3fs
from schemas.validation import is_valid_grib_file


def get_completed_init_times(
    bucket: str,
    access_key: str,
    secret: str,
    endpoint_url: str | None,
    region_name: str,
    prefix: str,
    last_processed: dt.datetime,
    files_out: list[str] | None = None,
) -> dict[dt.datetime, dt.datetime]:
    """Check S3 for complete initialization times for ECMWF live data."""
    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret,
        client_kwargs={
            "endpoint_url": endpoint_url,
            "region_name": region_name,
        },
    )

    try:
        files = fs.ls(f"{bucket}/ecmwf")
    except Exception as e:
        raise RuntimeError(f"Failed to list S3 bucket: {e}") from e

    if files_out is not None:
        files_out.extend(files)

    pattern = re.compile(rf"^{prefix}[DS](\d{{8}})[D]?(\d{{8}})\d$")
    init_time_max_steps: dict[dt.datetime, dt.datetime] = {}
    now = dt.datetime.now(dt.UTC)

    for f in files:
        filename = f.split("/")[-1]
        match = pattern.search(filename)
        if match:
            it_str, tt_str = match.groups()
            try:
                month = int(it_str[:2])
                year = now.year
                if month == 12 and now.month == 1:
                    year -= 1
                elif month == 1 and now.month == 12:
                    year += 1

                it = dt.datetime.strptime(f"{year}{it_str}", "%Y%m%d%H%M").replace(tzinfo=dt.UTC)
                tt_year = year + 1 if int(tt_str[:2]) < month else year
                tt = dt.datetime.strptime(f"{tt_year}{tt_str}", "%Y%m%d%H%M").replace(tzinfo=dt.UTC)

                if it not in init_time_max_steps or tt > init_time_max_steps[it]:
                    init_time_max_steps[it] = tt
            except ValueError:
                continue

    completed = {}
    for it, max_tt in init_time_max_steps.items():
        if it <= last_processed:
            continue
        max_required_step = 168 if it.hour in (0, 12) else 144
        if (max_tt - it).total_seconds() / 3600 >= max_required_step:
            completed[it] = max_tt

    return completed


def download_live_ecmwf_partition(
    partition_time: dt.datetime,
    out_path: Path,
    bucket: str,
    access_key: str,
    secret: str,
    endpoint_url: str | None,
    region_name: str,
    prefix: str,
    files: Sequence[str] | None = None,
    max_workers: int = 8,
) -> int:
    """Download raw ECMWF live data files in parallel into a single local file."""
    if is_valid_grib_file(out_path):
        return 1

    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret,
        client_kwargs={
            "endpoint_url": endpoint_url,
            "region_name": region_name,
        },
    )
    pattern = re.compile(rf"^{prefix}[DS]{partition_time.strftime('%m%d%H%M')}[D]?\d{{8}}\d$")
    listing = files if files is not None else fs.ls(f"{bucket}/ecmwf")
    matched = [f for f in listing if pattern.search(f.split("/")[-1])]

    if not matched:
        return 0

    def _read_s3_bytes(path: str) -> bytes:
        with fs.open(path, "rb") as in_f:
            return in_f.read()

    try:
        with (
            out_path.open("wb") as out_f,
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor,
        ):
            for payload in executor.map(_read_s3_bytes, matched):
                out_f.write(payload)
    except Exception:
        out_path.unlink(missing_ok=True)
        raise

    if not is_valid_grib_file(out_path):
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file {out_path} is not a valid GRIB file.")

    return len(matched)
