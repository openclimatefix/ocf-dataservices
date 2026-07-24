import dagster as dg

from ocf_dataservices.resources.iomanager_xarray_icechunk import (
    XarrayIcechunkIOManager,
)
from ocf_dataservices.defs.assets import (
    ecmwf_ens_uk,
)


local_resources = {
    "io_manager": XarrayIcechunkIOManager(
        path="/tmp/dagster_storage",
    )
}

defs = dg.Definitions(
    assets=[ecmwf_ens_uk],
    resources=local_resources,
)
