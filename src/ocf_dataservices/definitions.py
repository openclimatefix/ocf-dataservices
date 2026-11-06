import dagster as dg

from ocf_dataservices.defs.assets import (
    l0_ceda_global_west_europe_v1,
    l0_ceda_ukv_uk_v1,
    l0_mars_ecmwf_ens_uk_v1,
    l0_mars_ecmwf_hres_west_europe_v1,
    l1_ceda_global_west_europe_v1,
    l1_ceda_ukv_uk_v1,
    l1_mars_ecmwf_ens_uk_v1,
    l1_mars_ecmwf_hres_west_europe_v1,
)
from ocf_dataservices.resources.ceda import (
    DagsterCedaClient,
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
    "ceda_client": DagsterCedaClient(
        username=dg.EnvVar("CEDA_USER"),
        password=dg.EnvVar("CEDA_PASS"),
    ),
}

defs = dg.Definitions(
    assets=[
        l0_mars_ecmwf_ens_uk_v1,
        l1_mars_ecmwf_ens_uk_v1,
        l0_mars_ecmwf_hres_west_europe_v1,
        l1_mars_ecmwf_hres_west_europe_v1,
        l0_ceda_ukv_uk_v1,
        l1_ceda_ukv_uk_v1,
        l0_ceda_global_west_europe_v1,
        l1_ceda_global_west_europe_v1,
    ],
    resources=local_resources,
)
