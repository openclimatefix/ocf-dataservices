import datetime as dt
import re
from pathlib import Path

import s3fs


def get_completed_init_times(
    bucket: str,
    access_key: str,
    secret: str,
    endpoint_url: str | None,
    region_name: str,
    prefix: str,
    last_processed: dt.datetime,
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
                tt = dt.datetime.strptime(f"{year}{tt_str}", "%Y%m%d%H%M").replace(tzinfo=dt.UTC)

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
) -> int:
    """Download raw ECMWF live data files into a single local file."""
    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret,
        client_kwargs={
            "endpoint_url": endpoint_url,
            "region_name": region_name,
        },
    )
    pattern = re.compile(rf"^{prefix}[DS]{partition_time.strftime('%m%d%H%M')}[D]?\d{{8}}\d$")
    files = [f for f in fs.ls(f"{bucket}/ecmwf") if pattern.search(f.split("/")[-1])]

    if not files:
        return 0

    with out_path.open("wb") as out_f:
        for f in files:
            with fs.open(f, "rb") as in_f:
                out_f.write(in_f.read())

    return len(files)
