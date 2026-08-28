"""
ECMWF MARS API Client, adapted from the original.
"""

import dataclasses
import datetime as dt
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from contextlib import closing
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)


class APIException(Exception):
    pass


class MarsQueueLimitError(APIException):
    pass


@dataclasses.dataclass
class MarsRequest:
    """A request for data in the MARS format.

    See Also:
        - https://confluence.ecmwf.int/display/UDOC/MARS+request+syntax
    """

    params: list[str]
    init_time: dt.datetime
    steps: list[int]
    bbox_nwse: list[int]
    field_type: str
    classification: str = "od"
    expver: int = 1
    levtype: str = "sfc"
    stream: str = "oper"
    grid: str = "0.1/0.1"
    number: list[int] | None = None

    @classmethod
    def hres(
        cls,
        params: list[str],
        init_time: dt.datetime,
        steps: list[int],
        bbox_nwse: list[int],
        classification: str = "od",
        expver: int = 1,
        levtype: str = "sfc",
        grid: str = "0.1/0.1",
    ) -> MarsRequest:
        """Create a new request for High Resolution (HRES) operational data."""
        return cls(
            params=params,
            init_time=init_time,
            steps=steps,
            bbox_nwse=bbox_nwse,
            field_type="fc",
            classification=classification,
            expver=expver,
            levtype=levtype,
            stream="oper",
            grid=grid,
            number=None,
        )

    @classmethod
    def ens(
        cls,
        params: list[str],
        init_time: dt.datetime,
        steps: list[int],
        bbox_nwse: list[int],
        number: list[int],
        classification: str = "od",
        expver: int = 1,
        levtype: str = "sfc",
        grid: str = "0.1/0.1",
    ) -> MarsRequest:
        """Create a new request for Ensemble (ENS) perturbed forecast data."""
        return cls(
            params=params,
            init_time=init_time,
            steps=steps,
            bbox_nwse=bbox_nwse,
            field_type="pf",
            classification=classification,
            expver=expver,
            levtype=levtype,
            stream="enfo",
            grid=grid,
            number=number,
        )

    def gen_filename(self) -> str:
        """Generate a predictable filename for the requested data."""
        return f"ecmwf_{self.stream}-{self.field_type}_{self.init_time:%Y%m%dT%H}.grib"

    def to_dict(self) -> dict[str, str]:
        mars_req: dict[str, str] = {
            "class": self.classification,
            "date": f"{self.init_time:%Y%m%d}",
            "expver": str(self.expver),
            "levtype": self.levtype,
            "stream": self.stream,
            "param": "/".join(map(str, self.params)),
            "step": "/".join(map(str, self.steps)),
            "time": f"{self.init_time:%H}",
            "type": self.field_type,
            "area": "/".join(map(str, self.bbox_nwse)),
            "grid": self.grid,
        }
        if self.number is not None:
            mars_req["number"] = "/".join(map(str, self.number))

        return mars_req


class Ignore303(urllib.request.HTTPRedirectHandler):
    """Handler to automatically follow 301 and 302 redirects but ignore 303."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code in (301, 302):
            return urllib.request.Request(
                newurl,
                data=req.data,
                headers=req.headers,
                origin_req_host=req.origin_req_host,
                unverifiable=True,
            )
        return None

    def http_error_303(self, req, fp, code, msg, headers):
        return urllib.response.addinfourl(fp, headers, req.full_url, code)


class MarsClient:
    """Client for interacting with the ECMWF MARS API."""

    def __init__(
        self,
        url: str,
        key: str,
        email: str,
    ):
        self.url = url if url.endswith("/") else f"{url}/"
        self.key = key
        self.email = email
        self.opener = urllib.request.build_opener(Ignore303)

    def _call(
        self, method: str, url: str, payload: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any], str | None, int]:
        """
        Executes an HTTP request with exponential backoff on server errors.
        Returns: (HTTP status code, JSON response dictionary, Location header, Retry-After value)
        """
        max_tries = 10
        delay = 60

        for attempt in range(max_tries):
            try:
                return self._do_call(method, url, payload)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code == 400 and ("too many" in body.lower() or "queued" in body.lower()):
                    raise MarsQueueLimitError(f"MARS queue full: {body}")

                if e.code == 429 or e.code >= 500:
                    logger.warning(
                        f"HTTP {e.code} received (body: {body}). Retrying in {delay} seconds..."
                    )
                else:
                    raise APIException(f"HTTP Error {e.code}: {body}")
            except (urllib.error.URLError, ConnectionError) as e:
                logger.warning(f"Network error: {e}. Retrying in {delay} seconds...")

            if attempt < max_tries - 1:
                time.sleep(delay)

        raise APIException(f"Failed to call API after {max_tries} attempts.")

    def _do_call(
        self, method: str, url: str, payload: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any], str | None, int]:
        headers = {
            "Accept": "application/json",
            "From": self.email,
            "X-ECMWF-KEY": self.key,
        }
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            res = self.opener.open(req)
        except urllib.error.HTTPError as e:
            if e.code <= 299:
                res = e
            else:
                raise

        with closing(res):
            code = res.code
            body = res.read().decode("utf-8", errors="replace")
            location = res.headers.get("Location")
            retry_after = int(res.headers.get("Retry-After", 5))

        if code == 204:
            return code, {}, location, retry_after

        try:
            response_json = json.loads(body)
        except json.JSONDecodeError:
            response_json = {"error": f"Invalid JSON response: {body}"}

        if "error" in response_json:
            raise APIException(f"API Error: {response_json['error']}")

        return code, response_json, location, retry_after

    def execute(self, request: dict[str, Any] | MarsRequest, target: BinaryIO) -> dict[str, Any]:
        """
        Executes a MARS request and writes the downloaded data to `target`.

        Args:
            request: A MarsRequest helper object or dictionary outlining the parameters for MARS.
            target: A binary IO stream (like io.BytesIO() or an open file object).

        Returns:
            The final API response dictionary.
        """
        if isinstance(request, MarsRequest):
            request = request.to_dict()

        endpoint = urllib.parse.urljoin(self.url, "services/mars/requests")

        code, response, location, retry_after = self._call("POST", endpoint, request)

        if location:
            poll_url = urllib.parse.urljoin(endpoint, location)
        else:
            poll_url = endpoint

        try:
            status = response.get("status")

            # Poll the API until completion
            while status != "complete":
                time.sleep(retry_after)

                code, response, new_location, retry_after = self._call("GET", poll_url)

                if new_location:
                    poll_url = urllib.parse.urljoin(poll_url, new_location)

                status = response.get("status", status)

            # In cases of an HTTP 303, the response dictionary is the result
            result_obj = response if code == 303 else response.get("result", {})
            result_href = result_obj.get("href")

            if not result_href:
                raise APIException(
                    f"Status is 'complete', but no result href was returned. Response: {response}"
                )

            download_url = urllib.parse.urljoin(self.url, result_href)
            expected_size = result_obj.get("size", 0)

            self._download(download_url, target, expected_size)
        finally:
            # Clean up by telling the API to delete the request
            try:
                self._call("DELETE", poll_url)
            except APIException as e:
                logger.debug(f"Failed to cleanup request: {e}")

        return response

    def _download(self, url: str, target: BinaryIO, expected_size: int) -> None:
        """
        Downloads data from `url` in 1MB chunks and writes to `target`.
        """
        req = urllib.request.Request(url)
        bytes_transferred = 0

        with closing(urllib.request.urlopen(req)) as http:
            while True:
                chunk = http.read(1048576)  # Read 1MB at a time
                if not chunk:
                    break
                target.write(chunk)
                bytes_transferred += len(chunk)

        if expected_size and bytes_transferred != expected_size:
            logger.warning(
                f"WARNING: Expected {expected_size} bytes, but transferred {bytes_transferred} bytes."
            )
