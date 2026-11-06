import datetime as dt
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr
from ecmwfmars_data.ens.client import MarsClient, MarsRequest
from ecmwfmars_data.ens.download import convert_to_dataset, download_raw
from ecmwfmars_data.hres.download import (
    HRES_MARS_PARAMETERS,
    HRES_WEST_EUROPE_BBOX_NWSE,
    convert_hres_to_dataset,
    download_raw_hres,
)


class TestDownload(unittest.TestCase):
    def test_download_raw(self):
        client = MagicMock(spec=MarsClient)
        client.execute.side_effect = lambda _req, f: f.write(
            b"GRIB" + b"\x00" * 100_000 + b"7777"
        )
        init_time = dt.datetime(2023, 1, 1, 0, 0, tzinfo=dt.UTC)
        bbox_nwse = [60, -10, 50, 5]
        steps = [0, 3, 6]
        numbers = [1, 2, 3]

        with TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "test.grib"

            download_raw(
                client=client,
                init_time=init_time,
                bbox_nwse=bbox_nwse,
                steps=steps,
                numbers=numbers,
                target_path=target_path,
            )

            # Ensure client.execute was called
            client.execute.assert_called_once()

            # The first argument should be a MarsRequest dictionary or object
            args, _ = client.execute.call_args
            request = args[0]

            if isinstance(request, MarsRequest):
                request_dict = request.to_dict()
            else:
                request_dict = request

            self.assertEqual(request_dict["param"], "167/169/186/187/188")
            self.assertEqual(request_dict["date"], "20230101")
            self.assertEqual(request_dict["time"], "00")
            self.assertEqual(request_dict["step"], "0/3/6")
            self.assertEqual(request_dict["area"], "60/-10/50/5")
            self.assertEqual(request_dict["number"], "1/2/3")
            self.assertEqual(request_dict["type"], "pf")

    def test_convert_to_dataset(self):
        # We mock xr.open_dataset to return a simple dataset
        # since generating a real GRIB file in tests is complex

        def mock_open_dataset(path, engine, backend_kwargs=None, **kwargs):
            backend_kwargs = backend_kwargs or {}
            step_type = backend_kwargs.get("filter_by_keys", {}).get("stepType")

            # Create dummy coordinates
            # time scalar (default cfgrib behavior for single init time)
            time_coord = np.datetime64("2023-01-01T00:00:00")
            step = np.array(
                [np.timedelta64(0, "h"), np.timedelta64(3, "h"), np.timedelta64(6, "h")]
            )
            number = np.array([1, 2], dtype=np.int64)
            latitude = np.array([55.0, 54.0])
            longitude = np.array([-5.0, -4.0])

            coords = {
                "time": time_coord,
                "step": step,
                "number": number,
                "latitude": latitude,
                "longitude": longitude,
            }

            if step_type == "instant":
                # instantaneous variables
                # t2m in Kelvin, hcc/mcc/lcc in 0-1
                t2m = np.full((2, 3, 2, 2), 273.15, dtype=np.float32)
                hcc = np.full((2, 3, 2, 2), 0.5, dtype=np.float32)
                mcc = np.full((2, 3, 2, 2), 0.25, dtype=np.float32)
                lcc = np.full((2, 3, 2, 2), 1.0, dtype=np.float32)

                return xr.Dataset(
                    data_vars={
                        "t2m": (("number", "step", "latitude", "longitude"), t2m),
                        "hcc": (("number", "step", "latitude", "longitude"), hcc),
                        "mcc": (("number", "step", "latitude", "longitude"), mcc),
                        "lcc": (("number", "step", "latitude", "longitude"), lcc),
                        # Dummy variable that should be ignored by the schema
                        "surface": (
                            ("number", "step", "latitude", "longitude"),
                            np.zeros((2, 3, 2, 2)),
                        ),
                    },
                    coords=coords,
                    attrs={
                        "GRIB_generatingProcessIdentifier": 154,
                        "GRIB_expver": "0001",
                        "GRIB_tablesVersion": 31,
                    },
                )
            elif step_type == "accum":
                # accumulated variables
                # ssrd in J m-2, increasing over time: 0, 10800, 21600 (yields 1.0 W m-2 diff)
                ssrd = np.zeros((2, 3, 2, 2), dtype=np.float32)
                ssrd[:, 1, :, :] = 10800.0
                ssrd[:, 2, :, :] = 21600.0
                return xr.Dataset(
                    data_vars={
                        "ssrd": (("number", "step", "latitude", "longitude"), ssrd),
                    },
                    coords=coords,
                )
            else:
                raise ValueError("Unexpected stepType")

        # Patch xr.open_dataset
        with patch("xarray.open_dataset", side_effect=mock_open_dataset):
            ds = convert_to_dataset("dummy_path.grib")

            # Check dimensions order
            expected_dims = ("init_time", "step", "ensemble_member", "latitude", "longitude")
            self.assertEqual(tuple(ds.dims), expected_dims)

            # Check variables
            self.assertIn("temperature_2m", ds.data_vars)
            self.assertIn("downward_short_wave_radiation_flux_surface", ds.data_vars)
            self.assertIn("high_cloud_cover", ds.data_vars)
            self.assertIn("medium_cloud_cover", ds.data_vars)
            self.assertIn("low_cloud_cover", ds.data_vars)
            self.assertNotIn("surface", ds.data_vars)  # Should be dropped/ignored

            # Check conversions using subtests
            expected_conversions = [
                ("temperature_2m", 0.0),
                ("downward_short_wave_radiation_flux_surface", 1.0),
                ("high_cloud_cover", 50.0),
                ("medium_cloud_cover", 25.0),
                ("low_cloud_cover", 100.0),
            ]
            for var_name, expected_value in expected_conversions:
                with self.subTest(var=var_name):
                    np.testing.assert_allclose(ds[var_name].values, expected_value, atol=1e-5)

            # Check coords renaming and metadata coordinates
            self.assertIn("init_time", ds.coords)
            self.assertIn("ensemble_member", ds.coords)
            self.assertEqual(list(ds.coords["generating_process_id"].values), [154])
            self.assertEqual(list(ds.coords["expver"].values), [1])
            self.assertEqual(list(ds.coords["grib_tables_version"].values), [31])
            self.assertIn("ingested_at", ds.coords)

    def test_download_raw_hres_and_convert_hres_to_dataset(self):
        client = MagicMock(spec=MarsClient)
        client.execute.side_effect = lambda _req, f: f.write(
            b"GRIB" + b"\x00" * 100_000 + b"7777"
        )
        init_time = dt.datetime(2024, 1, 10, 12, 0, tzinfo=dt.UTC)

        with TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "hres.grib"
            download_raw_hres(
                client=client,
                init_time=init_time,
                bbox_nwse=HRES_WEST_EUROPE_BBOX_NWSE,
                steps=[0, 1, 2],
                target_path=target_path,
            )
            client.execute.assert_called_once()
            args, _ = client.execute.call_args
            req_dict = args[0].to_dict() if isinstance(args[0], MarsRequest) else args[0]
            self.assertEqual(req_dict["stream"], "oper")
            self.assertEqual(req_dict["type"], "fc")
            self.assertEqual(req_dict["area"], "63/-12/35/26")
            self.assertEqual(req_dict["param"], "/".join(HRES_MARS_PARAMETERS))

            # 06Z and 18Z HRES runs use MARS stream "scda"
            req_06z = MarsRequest.hres(
                params=HRES_MARS_PARAMETERS,
                init_time=dt.datetime(2024, 1, 10, 6, 0, tzinfo=dt.UTC),
                steps=[0, 1, 2],
                bbox_nwse=HRES_WEST_EUROPE_BBOX_NWSE,
            ).to_dict()
            self.assertEqual(req_06z["stream"], "scda")

        shape = ("step", "latitude", "longitude")
        acc = np.zeros((3, 2, 2), dtype=np.float32)
        acc[1] = 3600.0
        acc[2] = 7200.0
        raw_ds = xr.Dataset(
            data_vars={
                "10u": (shape, np.ones((3, 2, 2), dtype=np.float32)),
                "10v": (shape, np.ones((3, 2, 2), dtype=np.float32)),
                "100u": (shape, np.ones((3, 2, 2), dtype=np.float32)),
                "100v": (shape, np.ones((3, 2, 2), dtype=np.float32)),
                "t2m": (shape, np.full((3, 2, 2), 283.15, dtype=np.float32)),
                "200u": (shape, np.ones((3, 2, 2), dtype=np.float32)),
                "200v": (shape, np.ones((3, 2, 2), dtype=np.float32)),
                "dsrp": (shape, acc.copy()),
                "uvb": (shape, acc.copy()),
                "hcc": (shape, np.full((3, 2, 2), 0.5, dtype=np.float32)),
                "lcc": (shape, np.full((3, 2, 2), 0.5, dtype=np.float32)),
                "mcc": (shape, np.full((3, 2, 2), 0.5, dtype=np.float32)),
                "tprate": (shape, np.full((3, 2, 2), 0.001, dtype=np.float32)),
                "sd": (shape, np.zeros((3, 2, 2), dtype=np.float32)),
                "strd": (shape, acc.copy()),
                "ssrd": (shape, acc.copy()),
                "tcc": (shape, np.full((3, 2, 2), 0.5, dtype=np.float32)),
                "vis": (shape, np.full((3, 2, 2), 10000.0, dtype=np.float32)),
            },
            coords={
                "time": np.datetime64("2024-01-10T12:00:00"),
                "step": [
                    np.timedelta64(0, "h"),
                    np.timedelta64(1, "h"),
                    np.timedelta64(2, "h"),
                ],
                "latitude": [55.0, 50.0],
                "longitude": [-5.0, 5.0],
            },
            attrs={
                "GRIB_generatingProcessIdentifier": 155,
                "GRIB_expver": "0001",
                "GRIB_tablesVersion": 31,
            },
        )
        with patch("cfgrib.open_datasets", return_value=[raw_ds]):
            ds = convert_hres_to_dataset("dummy_hres.grib")

        self.assertEqual(tuple(ds.dims), ("init_time", "step", "latitude", "longitude"))
        self.assertEqual(len(ds.data_vars), 18)
        self.assertEqual(list(ds.step.values), [np.timedelta64(0, "h"), np.timedelta64(1, "h")])
        np.testing.assert_allclose(ds["temperature_2m"].values, 10.0, atol=1e-4)
        np.testing.assert_allclose(
            ds["downward_short_wave_radiation_flux_surface"].values, 1.0, atol=1e-4
        )
        self.assertEqual(list(ds.coords["generating_process_id"].values), [155])


if __name__ == "__main__":
    unittest.main()
