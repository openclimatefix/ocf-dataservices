import dagster as dg
from ceda_data.client import CedaClient


class DagsterCedaClient(dg.ConfigurableResource):
    """Dagster resource wrapper for the CEDA DAP client."""

    username: str
    password: str
    token_url: str = "https://services.ceda.ac.uk/api/token/create/"
    base_url: str = "https://dap.ceda.ac.uk/badc/ukmo-nwp/data"

    def get_client(self) -> CedaClient:
        return CedaClient(
            username=self.username,
            password=self.password,
            token_url=self.token_url,
            base_url=self.base_url,
        )
