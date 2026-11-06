import base64
import datetime as dt
import json
import logging
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class CedaClient:
    """Client for the CEDA DAP server (`dap.ceda.ac.uk`)."""

    def __init__(
        self,
        username: str,
        password: str,
        token_url: str = "https://services.ceda.ac.uk/api/token/create/",
        base_url: str = "https://dap.ceda.ac.uk/badc/ukmo-nwp/data",
        max_retries: int = 5,
    ) -> None:
        if not username or not password:
            raise ValueError("CEDA username and password must be non-empty.")
        self.username = username
        self.password = password
        self.token_url = token_url
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._token: str | None = None
        self._token_expires_at: dt.datetime = dt.datetime.min.replace(tzinfo=dt.UTC)

    def get_token(self) -> str:
        """Return a valid CEDA Bearer token, fetching a new one when expired or missing."""
        now = dt.datetime.now(tz=dt.UTC)
        with self._lock:
            if self._token is not None and now + dt.timedelta(seconds=60) < self._token_expires_at:
                return self._token

            credentials = base64.b64encode(
                f"{self.username}:{self.password}".encode()
            ).decode("ascii")
            req = urllib.request.Request(
                method="POST",
                url=self.token_url,
                headers={"Authorization": f"Basic {credentials}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as response:
                    data = json.load(response)
                token = str(data["access_token"])
                expires_raw = data.get("expires")
                if expires_raw:
                    expires_at = dt.datetime.fromisoformat(str(expires_raw)).astimezone(dt.UTC)
                else:
                    expires_at = now + dt.timedelta(hours=1)
            except Exception as e:
                raise OSError(f"Error authenticating with CEDA: {e}") from e

            self._token = token
            self._token_expires_at = expires_at
            return token

    def fetch_bytes(self, relative_path: str) -> bytes:
        """Download a single relative path under `base_url` and return its GRIB bytes."""
        url = f"{self.base_url}/{relative_path.lstrip('/')}"
        for attempt in range(self.max_retries):
            token = self.get_token()
            req = urllib.request.Request(
                method="GET",
                url=url,
                headers={"Authorization": f"Bearer {token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    payload = response.read()
                if len(payload) < 4 or payload[:4] != b"GRIB" or payload[-4:] != b"7777":
                    raise OSError(f"Invalid or truncated GRIB payload from '{url}'")
                return payload
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise FileNotFoundError(f"CEDA file not found (HTTP 404): {url}") from e
                if e.code == 401:
                    with self._lock:
                        self._token = None
                logger.debug(
                    "HTTP %d fetching '%s' (attempt %d/%d)",
                    e.code,
                    url,
                    attempt + 1,
                    self.max_retries,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Error fetching '%s' (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
            if attempt < self.max_retries - 1:
                time.sleep(2)

        raise OSError(f"Could not fetch {url} after {self.max_retries} attempts.")
