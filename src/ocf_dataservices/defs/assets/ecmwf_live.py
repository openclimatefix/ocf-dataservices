import datetime as dt
import os
import tempfile
from pathlib import Path

import dagster as dg
import xarray as xr
from ecmwflive_data.download import download_live_ecmwf_partition, get_completed_init_times
from ecmwflive_data.processing import process_ecmwf_live_validated
from ecmwflive_data.schema import EcmwfLiveSchema

ecmwf_live_partitions = dg.TimeWindowPartitionsDefinition(
    cron_schedule="0 0,6,12,18 * * *",
    start="2024-04-01T00:00",
    timezone="UTC",
    fmt="%Y-%m-%dT%H:%M",
)

l0_ecmwf_live_s3_v1 = dg.AssetSpec(
    key=dg.AssetKey(["nwp", "l0_ecmwf_live_s3_v1"]),
    partitions_def=ecmwf_live_partitions,
    group_name="L0",
    description="External asset representing raw ECMWF live data in S3",
)


@dg.sensor(
    minimum_interval_seconds=60,
)
def ecmwf_live_s3_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult:
    """Sensor that checks S3 for new ECMWF live data."""
    bucket = os.getenv("ECMWF_REALTIME_S3_BUCKET")
    if not bucket:
        context.log.warning("ECMWF_REALTIME_S3_BUCKET not set. Skipping sensor.")
        return dg.SensorResult()

    last_processed_str = context.cursor or "2000-01-01T00:00:00Z"
    last_processed = dt.datetime.fromisoformat(last_processed_str).replace(tzinfo=dt.UTC)

    try:
        completed_times = get_completed_init_times(
            bucket=bucket,
            access_key=os.getenv("ECMWF_REALTIME_S3_ACCESS_KEY", ""),
            secret=os.getenv("ECMWF_REALTIME_S3_ACCESS_SECRET", ""),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL", None),
            region_name=os.getenv("ECMWF_REALTIME_S3_REGION", "eu-west-1"),
            prefix=os.getenv("ECMWF_REALTIME_DISSEMINATION_FILE_PREFIX", "A2"),
            last_processed=last_processed,
        )
    except Exception as e:  # noqa: BLE001
        context.log.error(str(e))
        return dg.SensorResult()

    asset_events = []
    new_cursor = last_processed

    for it, max_tt in completed_times.items():
        partition_key = it.strftime("%Y-%m-%dT%H:%M")
        if ecmwf_live_partitions.has_partition_key(partition_key):
            asset_events.append(
                dg.AssetMaterialization(
                    asset_key=l0_ecmwf_live_s3_v1.key,
                    partition=partition_key,
                    metadata={
                        "init_time": it.isoformat(),
                        "max_target_time": max_tt.isoformat(),
                    },
                )
            )
            new_cursor = max(new_cursor, it)

    return dg.SensorResult(
        asset_events=asset_events,
        cursor=new_cursor.isoformat().replace("+00:00", "Z"),
    )


@dg.asset(
    key_prefix="nwp",
    deps=[l0_ecmwf_live_s3_v1],
    partitions_def=ecmwf_live_partitions,
    group_name="L0",
    io_manager_key="l0_io_manager",
    pool="ECMWF",
    automation_condition=dg.AutomationCondition.eager(),
)
def l0_ecmwf_live_local_v1(context: dg.AssetExecutionContext) -> dg.Output[Path]:
    """Download ECMWF live files for a specific partition into a local temporary GRIB file."""
    partition_time = context.partition_time_window.start

    fd, temp_path = tempfile.mkstemp(
        suffix=".grib", prefix=f"ecmwf_live_{partition_time.strftime('%Y%m%d%H%M')}_"
    )
    os.close(fd)
    out_path = Path(temp_path)

    num_files = download_live_ecmwf_partition(
        partition_time=partition_time,
        out_path=out_path,
        bucket=os.getenv("ECMWF_REALTIME_S3_BUCKET", ""),
        access_key=os.getenv("ECMWF_REALTIME_S3_ACCESS_KEY", ""),
        secret=os.getenv("ECMWF_REALTIME_S3_ACCESS_SECRET", ""),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        region_name=os.getenv("ECMWF_REALTIME_S3_REGION", "eu-west-1"),
        prefix=os.getenv("ECMWF_REALTIME_DISSEMINATION_FILE_PREFIX", "A2"),
    )

    if num_files == 0:
        raise dg.RetryRequested(max_retries=3, seconds_to_wait=60)

    return dg.Output(out_path, metadata={"num_files": num_files})


@dg.asset(
    key_prefix="nwp",
    partitions_def=ecmwf_live_partitions,
    group_name="L1",
    io_manager_key="l1_io_manager",
    pool="ECMWF",
    metadata={
        "bbox_nwse": [63, -12, 35, 26],
        "max_step_hours": 84,
        "schema": EcmwfLiveSchema,
    },
    automation_condition=dg.AutomationCondition.eager(),
    ins={"l0_ecmwf_live_local_v1": dg.AssetIn(key=dg.AssetKey(["nwp", "l0_ecmwf_live_local_v1"]))},
)
def l1_ecmwf_live_west_europe_v1(
    context: dg.AssetExecutionContext, l0_ecmwf_live_local_v1: Path
) -> dg.Output[xr.Dataset]:
    """Process L0 local GRIB data into L1 xarray dataset for the west Europe region."""
    metadata = context.assets_def.get_asset_spec().metadata
    ds = process_ecmwf_live_validated(
        grib_path=l0_ecmwf_live_local_v1,
        bbox_nwse=metadata["bbox_nwse"],
        max_step_hours=metadata["max_step_hours"],
    )
    return dg.Output(ds)


@dg.asset(
    key_prefix="nwp",
    partitions_def=ecmwf_live_partitions,
    group_name="L1",
    io_manager_key="l1_io_manager",
    pool="ECMWF",
    metadata={
        "bbox_nwse": [35, 67, 6, 97],
        "max_step_hours": 84,
        "schema": EcmwfLiveSchema,
    },
    automation_condition=dg.AutomationCondition.eager(),
    ins={"l0_ecmwf_live_local_v1": dg.AssetIn(key=dg.AssetKey(["nwp", "l0_ecmwf_live_local_v1"]))},
)
def l1_ecmwf_live_india_v1(
    context: dg.AssetExecutionContext, l0_ecmwf_live_local_v1: Path
) -> dg.Output[xr.Dataset]:
    """Process L0 local GRIB data into L1 xarray dataset for the India region."""
    metadata = context.assets_def.get_asset_spec().metadata
    ds = process_ecmwf_live_validated(
        grib_path=l0_ecmwf_live_local_v1,
        bbox_nwse=metadata["bbox_nwse"],
        max_step_hours=metadata["max_step_hours"],
    )
    return dg.Output(ds)
