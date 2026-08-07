import dagster as dg
from ocf_dataservices.defs.ecmwf_live import (
        ecmwf_live_s3_sensor,
        l0_ecmwf_live_local,
        l0_ecmwf_live_s3,
        l1_ecmwf_live_india,
        l1_ecmwf_live_nl,
        l1_ecmwf_live_uk,
)

from ocf_dataservices.defs.assets import (
        l0_mars_ecmwf_ens_uk_v1,
        l1_dynamical_ecmwf_ens_uk_v1,
        l1_mars_ecmwf_ens_uk_v1,
)
from ocf_dataservices.resources.iomanager_raw_file import (
        RawFileIOManager,
)
from ocf_dataservices.resources.iomanager_xarray_icechunk import (
        XarrayIcechunkIOManager,
)
from ocf_dataservices.resources.mars import (
        DagsterMarsClient,
)

# Using dg.EnvVar is Dagster best practice. It defers the resolution of the
# environment variable until execution time, and allows the Dagster UI to 
# explicitly display and validate the required configuration.
local_resources = {
    "l0_io_manager": RawFileIOManager(
        base_path=dg.EnvVar("L0_ROOT_PATH"),
    ),
    "l1_io_manager": XarrayIcechunkIOManager(
        path=dg.EnvVar("L1_ROOT_PATH"),
    ),
    "l2_io_manager": XarrayIcechunkIOManager(
        path=dg.EnvVar("L2_ROOT_PATH"),
    ),
    "mars_client": DagsterMarsClient(
        url=dg.EnvVar("ECMWF_API_URL"),
        key=dg.EnvVar("ECMWF_API_KEY"),
        email=dg.EnvVar("ECMWF_API_EMAIL"),
    ),
}

defs = dg.Definitions(
    assets=[
        l0_mars_ecmwf_ens_uk_v1,
        l1_mars_ecmwf_ens_uk_v1,
        l1_dynamical_ecmwf_ens_uk_v1,
        l0_ecmwf_live_s3,
        l0_ecmwf_live_local,
        l1_ecmwf_live_uk,
        l1_ecmwf_live_india,
        l1_ecmwf_live_nl,
    ],
    sensors=[
        ecmwf_live_s3_sensor,
    ],
    resources=local_resources,
)
