import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import dagster as dg
import xarray as xr
from metoffice_data.download import download_order_partition
from metoffice_data.processing import (
    process_metoffice_india,
    process_metoffice_ukv,
    process_metoffice_westeurope,
)
from metoffice_data.schema import (
    MetOfficeGlobalIndiaSchema,
    MetOfficeGlobalWesteuropeSchema,
    MetOfficeUkvSchema,
)

from ocf_dataservices.resources.metoffice import DagsterDatahubClient


def _build_order_assets(
    name: str,
    order_id: str,
    partitions_def: dg.TimeWindowPartitionsDefinition,
    l0_cron: str,
    process_fn: Callable[[Path], xr.Dataset],
    schema: type,
) -> tuple[dg.AssetsDefinition, dg.AssetsDefinition]:
    """Build the L0 download + L1 process asset pair for a single MetOffice DataHub order.

    The L0 asset polls the order's `/latest` endpoint on `l0_cron` (offset past the init time to
    allow for dissemination latency) and downloads the GRIB files; the eager L1 asset processes and
    validates them against `schema`.
    """
    l0_key = dg.AssetKey(["nwp", f"l0_metoffice_{name}_v1"])

    @dg.asset(
        name=f"l0_metoffice_{name}_v1",
        key_prefix="nwp",
        partitions_def=partitions_def,
        group_name="L0",
        io_manager_key="l0_io_manager",
        pool="metoffice",
        automation_condition=dg.AutomationCondition.on_cron(l0_cron),
        metadata={"order_id": order_id},
    )
    def l0_asset(
        context: dg.AssetExecutionContext, metoffice_client: DagsterDatahubClient
    ) -> dg.Output[Path]:
        """Download the MetOffice DataHub GRIB files for a partition into one local file."""
        partition_time = context.partition_time_window.start

        fd, temp_path = tempfile.mkstemp(
            suffix=".grib", prefix=f"metoffice_{name}_{partition_time.strftime('%Y%m%d%H%M')}_"
        )
        os.close(fd)
        out_path = Path(temp_path)

        num_files = download_order_partition(
            client=metoffice_client.get_client(),
            order_id=order_id,
            init_time=partition_time,
            out_path=out_path,
        )

        if num_files == 0:
            raise dg.RetryRequested(max_retries=5, seconds_to_wait=600)

        return dg.Output(out_path, metadata={"num_files": num_files})

    @dg.asset(
        name=f"l1_metoffice_{name}_v1",
        key_prefix="nwp",
        partitions_def=partitions_def,
        group_name="L1",
        io_manager_key="l1_io_manager",
        pool="metoffice",
        metadata={"schema": schema},
        automation_condition=dg.AutomationCondition.eager(),
        ins={"l0": dg.AssetIn(key=l0_key)},
    )
    def l1_asset(context: dg.AssetExecutionContext, l0: Path) -> dg.Output[xr.Dataset]:
        """Process L0 GRIB data into an L1 xarray dataset for this order's region."""
        return dg.Output(process_fn(l0))

    return l0_asset, l1_asset


metoffice_westeurope_partitions = dg.TimeWindowPartitionsDefinition(
    cron_schedule="0 0,6,12,18 * * *",
    start="2024-04-01T00:00",
    timezone="UTC",
    fmt="%Y-%m-%dT%H:%M",
)

metoffice_india_partitions = dg.TimeWindowPartitionsDefinition(
    cron_schedule="0 0,12 * * *",
    start="2024-04-01T00:00",
    timezone="UTC",
    fmt="%Y-%m-%dT%H:%M",
)

metoffice_ukv_partitions = dg.TimeWindowPartitionsDefinition(
    cron_schedule="0 0,3,6,9,12,15,18,21 * * *",
    start="2024-04-01T00:00",
    timezone="UTC",
    fmt="%Y-%m-%dT%H:%M",
)

l0_metoffice_westeurope_v1, l1_metoffice_westeurope_v1 = _build_order_assets(
    name="westeurope",
    order_id="westeurope-12params-54steps",
    partitions_def=metoffice_westeurope_partitions,
    l0_cron="0 4,10,16,22 * * *",
    process_fn=process_metoffice_westeurope,
    schema=MetOfficeGlobalWesteuropeSchema,
)

l0_metoffice_india_v1, l1_metoffice_india_v1 = _build_order_assets(
    name="india",
    order_id="india-11params-54steps",
    partitions_def=metoffice_india_partitions,
    l0_cron="0 4,16 * * *",
    process_fn=process_metoffice_india,
    schema=MetOfficeGlobalIndiaSchema,
)

l0_metoffice_ukv_v1, l1_metoffice_ukv_v1 = _build_order_assets(
    name="ukv",
    order_id="uk-12params-42steps",
    partitions_def=metoffice_ukv_partitions,
    l0_cron="0 4,7,10,13,16,19,22,1 * * *",
    process_fn=process_metoffice_ukv,
    schema=MetOfficeUkvSchema,
)
