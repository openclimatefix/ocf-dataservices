import io
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr
from ceda_data.client import CedaClient
from ceda_data.download import download_global_partition
from ceda_data.processing import (
    _UKV_X_OSGB,
    _UKV_Y_OSGB,
    _assign_ukv_osgb_coords,
    process_ceda_global_west_europe,
    process_ceda_ukv,
)
from ceda_data.schema import CedaGlobalWestEuropeSchema, CedaUkvSchema


def _make_global_raw_tiles() -> list[xr.Dataset]:
    """Build synthetic CEDA Global hypercubes with 0-360 longitudes and CEDA shortNames."""
    shape = ("step", "latitude", "longitude")
    # Include 350.0 (-10.0) and 10.0 inside West Europe [63, -12, 35, 26], plus 60.0 outside
    coords = {
        "time": np.datetime64("2024-01-01T00:00"),
        "step": [np.timedelta64(0, "h"), np.timedelta64(1, "h")],
        "latitude": [45.0, 55.0],
        "longitude": [350.0, 10.0, 60.0],
    }
    raw_vars = {
        "tcc": 0.5,
        "hcc": 0.2,
        "mcc": 0.3,
        "lcc": 0.4,
        "vis": 10000.0,
        "r": 0.8,
        "msl": 101325.0,
        "sd": 0.0,
        "swavr": 200.0,
        "t": 288.15,
        "u": 4.0,
        "v": -3.0,
    }
    return [
        xr.Dataset(
            data_vars={
                k: (
                    shape,
                    np.full((2, 2, 3), val, dtype=np.float32),
                    {"GRIB_generatingProcessIdentifier": 45, "GRIB_tablesVersion": 19},
                )
            },
            coords=coords,
        )
        for k, val in raw_vars.items()
    ]


class TestCedaProcessing(unittest.TestCase):
    def test_process_global_west_europe_validates(self) -> None:
        with patch("ceda_data.processing._read_raw_grib", return_value=_make_global_raw_tiles()):
            out = process_ceda_global_west_europe("dummy.grib")

        CedaGlobalWestEuropeSchema.validate(out)
        self.assertEqual(list(out.dims), ["init_time", "step", "latitude", "longitude"])
        # Longitude 60.0 cropped out; 350.0 wrapped to -10.0; latitude sorted descending [55, 45]
        np.testing.assert_allclose(out.longitude.values, [-10.0, 10.0])
        np.testing.assert_allclose(out.latitude.values, [55.0, 45.0])
        np.testing.assert_allclose(out["temperature_2m"].values, 15.0, atol=1e-4)
        np.testing.assert_allclose(out["total_cloud_cover_atmosphere"].values, 50.0, atol=1e-4)
        np.testing.assert_allclose(out["relative_humidity_2m"].values, 80.0, atol=1e-4)
        self.assertEqual(list(out.coords["generating_process_id"].values), [45])
        self.assertEqual(list(out.coords["grib_tables_version"].values), [19])

    def test_assign_ukv_1d_values_and_validate(self) -> None:
        ny, nx = len(_UKV_Y_OSGB), len(_UKV_X_OSGB)
        n_vals = ny * nx
        # Build 1D "values" dataset with heightAboveGround=[0, 1] on temperature "t"
        t_arr = np.full((2, 2, n_vals), 283.15, dtype=np.float32)
        t_arr[:, 0, :] = 270.0  # 0m height should be dropped in favour of heightAboveGround=1
        ds_t = xr.Dataset(
            data_vars={
                "t": (
                    ("step", "heightAboveGround", "values"),
                    t_arr,
                    {
                        "GRIB_generatingProcessIdentifier": 45,
                        "GRIB_tablesVersion": 19,
                        "GRIB_latitudeOfFirstGridPointInDegrees": 48.0,
                        "GRIB_longitudeOfFirstGridPointInDegrees": -12.0,
                    },
                )
            },
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": [np.timedelta64(0, "h"), np.timedelta64(1, "h")],
                "heightAboveGround": [0, 1],
            },
        )
        other_vars = {
            "tcc": 0.5,
            "hcc": 0.2,
            "mcc": 0.3,
            "lcc": 0.4,
            "vis": 8000.0,
            "r": 75.0,
            "prate": 0.001,
            "sd": 0.0,
            "dlwrf": 300.0,
            "dswrf": 150.0,
            "u10": 2.0,
            "v10": 1.0,
        }
        ds_rest = xr.Dataset(
            data_vars={
                k: (("step", "values"), np.full((2, n_vals), v, dtype=np.float32))
                for k, v in other_vars.items()
            },
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": [np.timedelta64(0, "h"), np.timedelta64(1, "h")],
            },
        )

        with patch("ceda_data.processing._read_raw_grib", return_value=[ds_t, ds_rest]):
            out = process_ceda_ukv("dummy_ukv.grib")

        CedaUkvSchema.validate(out)
        self.assertEqual(
            dict(out.sizes),
            {"init_time": 1, "step": 2, "y_osgb": ny, "x_osgb": nx},
        )
        np.testing.assert_allclose(out["temperature_2m"].values, 10.0, atol=1e-4)
        self.assertEqual(float(out["y_osgb"][0]), 1223000.0)
        self.assertEqual(float(out["x_osgb"][0]), -239000.0)

    def test_assign_ukv_osgb_rejects_wrong_grid_size(self) -> None:
        bad_ds = xr.Dataset(
            data_vars={"t": (("values",), np.zeros(10, dtype=np.float32))}
        )
        with self.assertRaises(ValueError):
            _assign_ukv_osgb_coords(bad_ds)

    def test_ceda_client_token_and_fetch_bytes(self) -> None:
        client = CedaClient(username="u", password="p")
        token_payload = json.dumps(
            {"access_token": "tok123", "expires": "2099-01-01T00:00:00.000000+00:00"}
        ).encode()
        grib_payload = b"GRIB" + b"\x00" * 100 + b"7777"

        def fake_urlopen(req, timeout=None):
            resp = MagicMock()
            resp.__enter__.return_value = resp
            if "token/create" in req.full_url:
                resp.read.return_value = token_payload
                return io.BytesIO(token_payload)
            resp.read.return_value = grib_payload
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(client.get_token(), "tok123")
            self.assertEqual(client.fetch_bytes("ukv-grib/test.grib"), grib_payload)

    def test_ukv_overlapping_steps_between_wholesale_and_t54_deduplicated(self) -> None:
        ny, nx = len(_UKV_Y_OSGB), len(_UKV_X_OSGB)
        n_vals = ny * nx
        other_vars = {
            "tcc": 0.5,
            "hcc": 0.2,
            "mcc": 0.3,
            "lcc": 0.4,
            "vis": 8000.0,
            "r": 75.0,
            "prate": 0.001,
            "sd": 0.0,
            "dlwrf": 300.0,
            "dswrf": 150.0,
            "t": 283.15,
            "u10": 2.0,
            "v10": 1.0,
        }
        # Wholesale1 has steps [0h, 1h], Wholesale1T54 has overlapping steps [1h, 2h]
        ds_w1 = xr.Dataset(
            data_vars={
                k: (("step", "values"), np.full((2, n_vals), v, dtype=np.float32))
                for k, v in other_vars.items()
            },
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": [np.timedelta64(0, "h"), np.timedelta64(1, "h")],
            },
        )
        ds_w1_t54 = xr.Dataset(
            data_vars={
                k: (("step", "values"), np.full((2, n_vals), v, dtype=np.float32))
                for k, v in other_vars.items()
            },
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": [np.timedelta64(1, "h"), np.timedelta64(2, "h")],
            },
        )
        with patch("ceda_data.processing._read_raw_grib", return_value=[ds_w1, ds_w1_t54]):
            out = process_ceda_ukv("dummy_ukv.grib")

        self.assertEqual(
            list(out.step.values),
            [np.timedelta64(0, "h"), np.timedelta64(1, "h"), np.timedelta64(2, "h")],
        )

    def test_download_global_partition_defaults_to_area_a(self) -> None:
        import datetime as dt
        import tempfile
        from pathlib import Path

        client = MagicMock(spec=CedaClient)
        client.fetch_bytes.return_value = b"GRIB" + b"\x00" * 10_000 + b"7777"
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "global.grib"
            count = download_global_partition(
                client=client,
                init_time=dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.UTC),
                out_path=out_path,
            )
            self.assertEqual(count, 12)
            self.assertEqual(client.fetch_bytes.call_count, 12)
            for call in client.fetch_bytes.call_args_list:
                self.assertIn("_AreaA_", call.args[0])


if __name__ == "__main__":
    unittest.main()
