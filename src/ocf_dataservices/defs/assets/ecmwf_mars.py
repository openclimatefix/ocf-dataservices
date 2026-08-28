import os
import random
import tempfile
import time
from pathlib import Path

import dagster as dg
import xarray as xr
from ecmwfmars_data.ens.client import MarsQueueLimitError
from ecmwfmars_data.ens.download import convert_to_dataset, download_raw
from ecmwfmars_data.ens.schema import MarsEcmwfEnsSchema

from ocf_dataservices.defs.assets.dynamical import ecmwf_ens_partitions
from ocf_dataservices.resources.mars import DagsterMarsClient


@dg.asset(
    key_prefix="nwp",
    partitions_def=ecmwf_ens_partitions,
    group_name="L0",
    pool="ecmwf_mars",
    io_manager_key="l0_io_manager",
    metadata={
        "bbox_nwse": [62, -12, 48, 3],
        "steps": list(range(86)),
        "numbers": list(range(1, 51)),
    },
)
def l0_mars_ecmwf_ens_uk_v1(
    context: dg.AssetExecutionContext, mars_client: DagsterMarsClient
) -> dg.Output[Path]:
    """
    Downloads raw ECMWF MARS ensemble GRIB data for a specific 00Z init time.
    Returns the path to the downloaded GRIB file.
    """
    partition_key = context.partition_key
    nwp_init_time = context.partition_time_window.start
    metadata = context.assets_def.get_asset_spec().metadata

    client = mars_client.get_client()

    # Use mkstemp to hold the GRIB data.
    # The l0_io_manager will move this file to final storage when we return it.
    fd, temp_path = tempfile.mkstemp(
        suffix=".grib", prefix=f"mars_ens_{partition_key.replace(':', '')}_"
    )
    os.close(fd)

    target_path = Path(temp_path)

    context.log.info(f"Downloading MARS ENS data for {nwp_init_time} to {target_path}")

    start_time = time.perf_counter()
    try:
        download_raw(
            client=client,
            init_time=nwp_init_time,
            bbox_nwse=metadata["bbox_nwse"],
            steps=metadata["steps"],
            numbers=metadata["numbers"],
            target_path=target_path,
        )
    except MarsQueueLimitError as e:
        context.log.warning(f"MARS queue full. Retrying. Error: {e}")
        raise dg.RetryRequested(max_retries=100, seconds_to_wait=random.randint(600, 720)) from e

    elapsed_time = time.perf_counter() - start_time

    context.log.info(f"Successfully downloaded MARS ENS data to {target_path}")

    return dg.Output(
        target_path, metadata={"processing_time_seconds": dg.MetadataValue.float(elapsed_time)}
    )


@dg.asset(
    key_prefix="nwp",
    partitions_def=ecmwf_ens_partitions,
    group_name="L1",
    pool="ecmwf_mars",
    io_manager_key="l1_io_manager",
    metadata={
        "schema": MarsEcmwfEnsSchema,
    },
    ins={
        "l0_mars_ecmwf_ens_uk_v1": dg.AssetIn(key=dg.AssetKey(["nwp", "l0_mars_ecmwf_ens_uk_v1"]))
    },
)
def l1_mars_ecmwf_ens_uk_v1(
    context: dg.AssetExecutionContext, l0_mars_ecmwf_ens_uk_v1: Path
) -> dg.Output[xr.Dataset]:
    """
    Converts raw ECMWF MARS ensemble GRIB data to an Xarray Dataset matching the MarsEcmwfEnsSchema.
    """
    grib_path = l0_mars_ecmwf_ens_uk_v1

    context.log.info(f"Converting MARS ENS GRIB data from {grib_path}")

    start_time = time.perf_counter()
    ds = convert_to_dataset(grib_path)
    elapsed_time = time.perf_counter() - start_time
    context.log.info("Successfully converted and validated MARS ENS data.")
    return dg.Output(ds, metadata={"processing_time_seconds": dg.MetadataValue.float(elapsed_time)})
