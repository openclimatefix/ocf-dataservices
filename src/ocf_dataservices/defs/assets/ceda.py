import datetime as dt
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import dagster as dg
import xarray as xr
from ceda_data.client import CedaClient
from ceda_data.download import download_global_partition, download_ukv_partition
from ceda_data.processing import process_ceda_global_west_europe, process_ceda_ukv
from ceda_data.schema import CedaGlobalWestEuropeSchema, CedaUkvSchema

from ocf_dataservices.resources.ceda import DagsterCedaClient


def _build_ceda_assets(
    name: str,
    partitions_def: dg.TimeWindowPartitionsDefinition,
    download_fn: Callable[[CedaClient, dt.datetime, Path], int],
    process_fn: Callable[[Path | str], xr.Dataset],
    schema: type,
) -> tuple[dg.AssetsDefinition, dg.AssetsDefinition]:
    """Build the L0 download + L1 process asset pair for a CEDA NWP product."""
    l0_key = dg.AssetKey(["nwp", f"l0_ceda_{name}_v1"])

    @dg.asset(
        name=f"l0_ceda_{name}_v1",
        key_prefix="nwp",
        partitions_def=partitions_def,
        group_name="L0",
        io_manager_key="l0_io_manager",
        pool="ceda",
    )
    def l0_asset(
        context: dg.AssetExecutionContext, ceda_client: DagsterCedaClient
    ) -> dg.Output[Path]:
        """Download CEDA GRIB files for a partition into one local file."""
        existing_path = context.resources.l0_io_manager.existing_partition(
            context.asset_key, context.partition_key
        )
        if existing_path is not None:
            context.log.info(
                f"Valid L0 GRIB for partition {context.partition_key} already exists at {existing_path}. Skipping download."
            )
            return dg.Output(existing_path, metadata={"status": "skipped"})

        partition_time = context.partition_time_window.start

        fd, temp_path = tempfile.mkstemp(
            suffix=".grib", prefix=f"ceda_{name}_{partition_time.strftime('%Y%m%d%H%M')}_"
        )
        os.close(fd)
        out_path = Path(temp_path)

        try:
            num_files = download_fn(ceda_client.get_client(), partition_time, out_path)
        except Exception:
            out_path.unlink(missing_ok=True)
            raise

        return dg.Output(out_path, metadata={"num_files": num_files})

    @dg.asset(
        name=f"l1_ceda_{name}_v1",
        key_prefix="nwp",
        partitions_def=partitions_def,
        group_name="L1",
        io_manager_key="l1_io_manager",
        pool=f"l1_ceda_{name}",
        metadata={"schema": schema},
        automation_condition=dg.AutomationCondition.eager(),
        ins={"l0": dg.AssetIn(key=l0_key)},
    )
    def l1_asset(context: dg.AssetExecutionContext, l0: Path) -> dg.Output[xr.Dataset]:
        """Process L0 CEDA GRIB data into a validated L1 xarray Dataset."""
        return dg.Output(process_fn(l0))

    return l0_asset, l1_asset


ceda_ukv_partitions = dg.TimeWindowPartitionsDefinition(
    cron_schedule="0 0,6,12,18 * * *",
    start="2017-06-01T00:00",
    timezone="UTC",
    fmt="%Y-%m-%dT%H:%M",
    end_offset=-20,  # 20 * 6h = 5-day lag for CEDA archive availability
)

ceda_global_partitions = dg.TimeWindowPartitionsDefinition(
    cron_schedule="0 0,12 * * *",
    start="2017-06-01T00:00",
    timezone="UTC",
    fmt="%Y-%m-%dT%H:%M",
    end_offset=-10,  # 10 * 12h = 5-day lag for CEDA archive availability
)

l0_ceda_ukv_uk_v1, l1_ceda_ukv_uk_v1 = _build_ceda_assets(
    name="ukv_uk",
    partitions_def=ceda_ukv_partitions,
    download_fn=download_ukv_partition,
    process_fn=process_ceda_ukv,
    schema=CedaUkvSchema,
)

l0_ceda_global_west_europe_v1, l1_ceda_global_west_europe_v1 = _build_ceda_assets(
    name="global_west_europe",
    partitions_def=ceda_global_partitions,
    download_fn=download_global_partition,
    process_fn=process_ceda_global_west_europe,
    schema=CedaGlobalWestEuropeSchema,
)
