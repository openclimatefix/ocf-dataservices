import concurrent.futures
import datetime as dt
import itertools
from collections.abc import Sequence
from pathlib import Path

from schemas.validation import is_valid_grib_file

from .client import CedaClient


def _download_many_to_file(
    client: CedaClient,
    relative_paths: Sequence[str],
    out_path: Path,
    max_workers: int = 10,
) -> int:
    """Download `relative_paths` in parallel and stream them in order into `out_path`."""
    if is_valid_grib_file(out_path):
        return 1
    if not relative_paths:
        return 0

    count = 0
    try:
        with (
            out_path.open("wb") as out_f,
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor,
        ):
            for payload in executor.map(client.fetch_bytes, relative_paths):
                out_f.write(payload)
                count += 1
    except Exception:
        out_path.unlink(missing_ok=True)
        raise

    if not is_valid_grib_file(out_path):
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file {out_path} is not a valid GRIB file.")

    return count


def download_ukv_partition(
    client: CedaClient,
    init_time: dt.datetime,
    out_path: Path,
    max_workers: int = 4,
) -> int:
    """Download the 4 CEDA UKV wholesale GRIB files for `init_time` into `out_path`."""
    stubs = ("Wholesale1", "Wholesale1T54", "Wholesale2", "Wholesale2T54")
    relative_paths = [
        init_time.strftime(f"ukv-grib/%Y/%m/%d/%Y%m%d%H%M_u1096_ng_umqv_{stub}.grib")
        for stub in stubs
    ]
    return _download_many_to_file(
        client=client,
        relative_paths=relative_paths,
        out_path=out_path,
        max_workers=max_workers,
    )


def download_global_partition(
    client: CedaClient,
    init_time: dt.datetime,
    out_path: Path,
    areas: Sequence[str] = ("AreaA",),
    max_workers: int = 10,
) -> int:
    """Download the CEDA Global 17km GRIB files for `init_time` into `out_path`.

    Defaults to `AreaA` (Lat 0..89.856, Lon -45..45), which covers all of West Europe.
    """
    params = (
        "Total_Downward_Surface_SW_Flux",
        "high_cloud_amount",
        "low_cloud_amount",
        "medium_cloud_amount",
        "mean_sea_level_pressure",
        "relative_humidity_1_5m",
        "snow_depth",
        "temperature_1_5m",
        "total_cloud",
        "visibility_1_5m",
        "wind_u_10m",
        "wind_v_10m",
    )
    relative_paths = [
        init_time.strftime(
            f"global-grib/%Y/%m/%d/%Y%m%d%H_WSGlobal17km_{param}_{area}_000144.grib"
        )
        for param, area in itertools.product(params, areas)
    ]
    return _download_many_to_file(
        client=client,
        relative_paths=relative_paths,
        out_path=out_path,
        max_workers=max_workers,
    )
