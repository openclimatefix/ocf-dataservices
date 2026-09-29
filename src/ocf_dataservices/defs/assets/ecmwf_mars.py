import os
import random
import tempfile
import time
import urllib.error
from pathlib import Path

import dagster as dg
import xarray as xr
from ecmwfmars_data.ens.client import APIException, MarsQueueLimitError, MarsRequest
from ecmwfmars_data.ens.download import convert_to_dataset
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

    start_time = time.perf_counter()

    # Use a deterministic path for the .url file in the shared L0 storage so retries can find it
    # We fetch L0_ROOT_PATH directly from the environment variables configured by Dagster
    l0_base_path = Path(os.environ.get("L0_ROOT_PATH", "/tmp/dagster_storage"))
    url_dir = l0_base_path / "nwp" / "l0_mars_ecmwf_ens_uk_v1" / "_mars_urls"
    url_dir.mkdir(parents=True, exist_ok=True)
    url_file_path = url_dir / f"{partition_key.replace(':', '')}.url"
    
    poll_url = None

    if url_file_path.exists():
        poll_url = url_file_path.read_text().strip()
        context.log.info(f"Found existing poll URL: {poll_url}")

    if not poll_url:
        try:
            req = MarsRequest.ens(
                params=["167", "169", "186", "187", "188"],
                init_time=nwp_init_time,
                steps=metadata["steps"],
                bbox_nwse=metadata["bbox_nwse"],
                number=metadata["numbers"],
            )
            context.log.info(f"Submitting new MARS request for {nwp_init_time}")
            poll_url = client.submit(req)
            url_file_path.write_text(poll_url)
            context.log.info(f"Submitted new MARS request. Poll URL: {poll_url}")
            
            # Yield slot immediately. Poll after 5 minutes initially.
            raise dg.RetryRequested(max_retries=1000, seconds_to_wait=300)
        except MarsQueueLimitError as e:
            context.log.warning(f"MARS queue full on submit. Retrying later. Error: {e}")
            raise dg.RetryRequested(max_retries=1000, seconds_to_wait=random.randint(300, 600)) from e

    try:
        status, response = client.status(poll_url)
    except MarsQueueLimitError as e:
        context.log.warning(f"MARS queue full evaluated during status check: {e}. Deleting MARS job and retrying later.")
        url_file_path.unlink(missing_ok=True)
        # Start over on next retry with jitter
        raise dg.RetryRequested(max_retries=1000, seconds_to_wait=random.randint(300, 600)) from e
    except (APIException, urllib.error.URLError, ConnectionError) as e:
        context.log.error(f"Network or API error while checking status for {poll_url}: {e}. Resetting request.")
        url_file_path.unlink(missing_ok=True)
        raise dg.RetryRequested(max_retries=1000, seconds_to_wait=300)

    context.log.info(f"MARS request status: {status}")

    if status in ("queued", "active"):
        raise dg.RetryRequested(max_retries=1000, seconds_to_wait=300)

    if status != "complete":
        context.log.warning(f"MARS job not complete. Status is '{status}'. Response: {response}. Deleting MARS job and retrying later.")
        url_file_path.unlink(missing_ok=True)
        try:
            client.cleanup(poll_url)
        except (APIException, urllib.error.URLError, ConnectionError):
            pass
        
        # If rejected (often due to queue limits evaluated async), back off with jitter
        raise dg.RetryRequested(max_retries=1000, seconds_to_wait=random.randint(300, 600))

    context.log.info("MARS request complete. Downloading data.")
    result_obj = response if "href" in response else response.get("result", {})
    result_href = result_obj.get("href")

    if not result_href:
        url_file_path.unlink(missing_ok=True)
        raise RuntimeError(f"Status is 'complete' but no href in response: {response}")

    expected_size = result_obj.get("size", 0)

    try:
        with open(target_path, "wb") as f:
            client.download_result(result_href, f, expected_size)
    finally:
        client.cleanup(poll_url)
        url_file_path.unlink(missing_ok=True)

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
