import time

import dagster as dg
import xarray as xr
from dynamical_data.cli import UK_BBOX_NWSE
from dynamical_data.ecmwf_ens.download import NwpRunNotYetAvailable, download, open_it
from dynamical_data.ecmwf_ens.schema import DynamicalEcmwfEnsSchema

ecmwf_ens_partitions = dg.TimeWindowPartitionsDefinition(
    cron_schedule="0 0,6,12,18 * * *",
    start="2024-04-01T00:00",
    timezone="UTC",
    fmt="%Y-%m-%dT%H:%M",
    end_offset=2,  # Delay keys by 2 intervals (~12 hours) to match availability lag
)
"""One partition per 6-hourly ECMWF ENS run (00Z, 06Z, 12Z, 18Z). ``end_offset=2`` allows data time to land."""


@dg.asset(
    key_prefix="nwp",
    partitions_def=ecmwf_ens_partitions,
    group_name="L1",
    io_manager_key="l1_io_manager",
    pool="dynamical",
    metadata={
        "schema": DynamicalEcmwfEnsSchema,
        "bbox_nwse": list(UK_BBOX_NWSE),
    },
)
def l1_dynamical_ecmwf_ens_uk_v1(context: dg.AssetExecutionContext) -> dg.Output[xr.Dataset]:
    """
    Downloads and processes ECMWF ensemble NWP data for a specific day from Dynamical.
    """
    partition_key = context.partition_key
    nwp_init_time = context.partition_time_window.start
    metadata = context.assets_def.get_asset_spec().metadata

    io_manager = context.resources.l1_io_manager

    existing_partition = io_manager.existing_partition(
        context.asset_key,
        metadata["schema"].append_dim(),
        partition_key,
    )
    if existing_partition is not None:
        context.log.info(
            f"Data for partition {partition_key} already exists in Icechunk. Skipping download."
        )
        return dg.Output(existing_partition, metadata={"status": "skipped"})

    start_time = time.perf_counter()
    try:
        ds_lazy = open_it(nwp_init_time=nwp_init_time, bbox_nwse=tuple(metadata["bbox_nwse"]))
    except NwpRunNotYetAvailable as exc:
        # 8 retries * 1800s ≈ 4h of coverage past the 08:30 UTC schedule, comfortably past
        # Dynamical's typical publication time.
        raise dg.RetryRequested(max_retries=8, seconds_to_wait=1800) from exc
    context.log.info("Lazily opened Icechunk store.")

    ds: xr.Dataset = download(ds_lazy)
    elapsed_time = time.perf_counter() - start_time
    context.log.info("Downloaded and validated Icechunk data.")

    return dg.Output(ds, metadata={"processing_time_seconds": dg.MetadataValue.float(elapsed_time)})
