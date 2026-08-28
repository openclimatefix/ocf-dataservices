import dagster as dg
from metoffice_data.client import DatahubClient


class DagsterDatahubClient(dg.ConfigurableResource):
    """Dagster resource wrapper for the MetOffice DataHub client."""

    api_key: str
    base_url: str = "https://data.hub.api.metoffice.gov.uk/atmospheric-models/1.0.0/orders"
    dataspec: str = "1.1.0"

    def get_client(self) -> DatahubClient:
        return DatahubClient(
            api_key=self.api_key,
            base_url=self.base_url,
            dataspec=self.dataspec,
        )
