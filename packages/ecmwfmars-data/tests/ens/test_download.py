import datetime as dt
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr
from ecmwfmars_data.ens.client import MarsClient, MarsRequest
from ecmwfmars_data.ens.download import convert_to_dataset, download_raw


class TestDownload(unittest.TestCase):
    def test_download_raw(self):
        client = MagicMock(spec=MarsClient)
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
            step = np.array([np.timedelta64(0, 'h'), np.timedelta64(3, 'h'), np.timedelta64(6, 'h')])
            number = np.array([1, 2], dtype=np.int16)
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
                        "surface": (("number", "step", "latitude", "longitude"), np.zeros((2, 3, 2, 2)))
                    },
                    coords=coords
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
                    coords=coords
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
            self.assertNotIn("surface", ds.data_vars) # Should be dropped/ignored
            
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
            
            # Check coords renaming
            self.assertIn("init_time", ds.coords)
            self.assertIn("ensemble_member", ds.coords)


if __name__ == "__main__":
    unittest.main()
