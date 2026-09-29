import datetime as dt
import json
import urllib.request
from typing import BinaryIO


class DatahubClient:
    """Client for the MetOffice Weather DataHub atmospheric-models order API.

    See https://datahub.metoffice.gov.uk/docs/f/category/atmospheric/type/atmospheric/api-documentation.
    Data is delivered per order, one GRIB file per parameter per step.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://data.hub.api.metoffice.gov.uk/atmospheric-models/1.0.0/orders",
        dataspec: str = "1.1.0",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.dataspec = dataspec

    def _headers(self, accept: str) -> dict[str, str]:
        return {"Accept": accept, "apikey": self.api_key}

    def list_file_ids(self, order_id: str, init_time: dt.datetime) -> list[str]:
        """List GRIB file IDs available for an order at a given init time.

        File IDs containing '+' are aggregate/summary entries and are skipped.
        """
        url = (
            f"{self.base_url}/{order_id}/latest"
            f"?detail=MINIMAL&runfilter={init_time:%Y%m%d%H}&dataSpec={self.dataspec}"
        )
        req = urllib.request.Request(url, headers=self._headers("application/json"), method="GET")
        with urllib.request.urlopen(req, timeout=30) as response:
            charset = response.info().get_param("charset") or "utf-8"
            data = json.loads(response.read().decode(charset))

        files = data.get("orderDetails", {}).get("files", [])
        return [f["fileId"] for f in files if "fileId" in f and "+" not in f["fileId"]]

    def download_file(self, order_id: str, file_id: str, target: BinaryIO) -> None:
        """Download a single GRIB file, streaming it into `target`."""
        url = f"{self.base_url}/{order_id}/latest/{file_id}/data?dataSpec={self.dataspec}"
        req = urllib.request.Request(url, headers=self._headers("application/x-grib"), method="GET")
        with urllib.request.urlopen(req, timeout=120) as response:
            target.writelines(iter(lambda: response.read(16 * 1024), b""))
