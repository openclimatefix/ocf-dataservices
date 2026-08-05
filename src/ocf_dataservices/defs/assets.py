import datetime as dt
from typing import Final

import dagster as dg
import xarray as xr
from dynamical_data.ecmwf_ens.download import NwpRunNotYetAvailable, download, open_it
from dynamical_data.ecmwf_ens.schema import EcmwfEnsSchema

ecmwf_ens_partitions = dg.DailyPartitionsDefinition(
    start_date="2024-04-01", timezone="UTC", end_offset=1
)
"""One partition per day of ECMWF ENS 00Z runs. ``end_offset=1`` makes today's key exist before
its 00Z run has actually landed, matching Dynamical's publication lag; shared with
``ecmwf_ens_job``/``ecmwf_ens_schedule`` in ``defs/schedules.py``."""

_ECMWF_ENS_MAX_RETRIES: Final[int] = 8
"""Retries × ``_ECMWF_ENS_RETRY_DELAY_SECONDS`` ≈ 4h of coverage past the 08:30 UTC schedule
(``ecmwf_ens_schedule``), comfortably past Dynamical's typical publication time. Only applies to
``NwpRunNotYetAvailable``; a genuine bug fails immediately instead of retrying for hours."""

_ECMWF_ENS_RETRY_DELAY_SECONDS: Final[int] = 1800
"""How long to wait between retries of a not-yet-published ECMWF run."""

@dg.asset(
    partitions_def=ecmwf_ens_partitions,
    key=dg.AssetKey(["nwp", "ecmwf_ens_uk_dynamical_0p25deg"]),
    group_name="L1",
    # The `pool="ECMWF"` works in conjunction with the Dagster instance configuration
    # (e.g., in `dagster.yaml`) to limit the number of times this asset can be run
    # concurrently. This is crucial because downloading ECMWF data is memory-intensive.
    # See: https://docs.dagster.io/guides/operate/managing-concurrency/concurrency-pools
    pool="ECMWF",
    metadata={
        "append_dim": "init_time",
        "chunks": EcmwfEnsSchema._chunks,
        "shards": EcmwfEnsSchema._shards,
        "bbox_nwse": [62, -12, 48, 3],
    },
)
def ecmwf_ens_uk_dynamical_0p25deg(context: dg.AssetExecutionContext) -> xr.Dataset:
    """
    Downloads and processes ECMWF ensemble NWP data for a specific day.
    """
    partition_date_str = context.partition_key
    nwp_init_time = dt.datetime.strptime(partition_date_str, "%Y-%m-%d").replace(tzinfo=dt.UTC)
    metadata = context.assets_def.get_asset_spec().metadata

    io_manager = context.resources.io_manager

    existing_partition = io_manager.existing_partition(
        context.asset_key,
        metadata["append_dim"],
        partition_date_str,
    )
    if existing_partition is not None:
        context.log.info(f"Data for partition {partition_date_str} already exists in Icechunk. Skipping download.")
        return existing_partition

    try:
        ds_lazy = open_it(nwp_init_time=nwp_init_time, bbox_nwse=tuple(metadata["bbox_nwse"]))
    except NwpRunNotYetAvailable as exc:
        raise dg.RetryRequested(
            max_retries=_ECMWF_ENS_MAX_RETRIES, seconds_to_wait=_ECMWF_ENS_RETRY_DELAY_SECONDS
        ) from exc
    context.log.info("Lazily opened Icechunk store.")

    ds: xr.Dataset = download(ds_lazy)
    context.log.info("Downloaded and validated Icechunk data.")

    return ds


