import dagster as dg
from ecmwfmars_data.ens.client import MarsClient


class DagsterMarsClient(dg.ConfigurableResource):
    """Dagster resource wrapper for the MarsClient."""

    url: str = "https://api.ecmwf.int/v1"
    key: str
    email: str

    def get_client(self) -> MarsClient:
        return MarsClient(
            url=self.url,
            key=self.key,
            email=self.email,
        )
