import datetime as dt
import os
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from ecmwflive_data.cli import PRODUCTS
from ecmwflive_data.processing import process_ecmwf_live


class TestProcessing(unittest.TestCase):
    def test_process_ecmwf_live_truncates_step(self):
        """Test that process_ecmwf_live correctly slices step coordinates."""
        # Create mock dataset
        ds = xr.Dataset(
            data_vars={
                "t2m": (("step", "latitude", "longitude"), np.ones((3, 2, 2), dtype=np.float32)),
            },
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": [np.timedelta64(0, "h"), np.timedelta64(84, "h"), np.timedelta64(96, "h")],
                "latitude": [52.0, 51.0],
                "longitude": [4.0, 5.0],
            },
        )

        with patch("cfgrib.open_datasets") as mock_open:
            # Mock open return
            mock_open.return_value = [ds]

            # Max step is 84
            out_ds = process_ecmwf_live(
                grib_path="dummy.grib", bbox_nwse=[53, 3, 50, 6], max_step_hours=84
            )

            self.assertIn("step", out_ds.coords)
            # 96h should be dropped
            self.assertNotIn(np.timedelta64(96, "h"), out_ds.step.values)
            self.assertIn(np.timedelta64(84, "h"), out_ds.step.values)

    def test_radiation_diff_and_cloud_overshoot_conversion(self):
        """Test radiation de-accumulation keeps step 84h and cloud overshoot 1.0001 scales to 100%."""
        ssrd = np.zeros((3, 2, 2), dtype=np.float32)
        ssrd[1, :, :] = 3600.0
        ssrd[2, :, :] = 7200.0
        ds = xr.Dataset(
            data_vars={
                "t2m": (("step", "latitude", "longitude"), np.full((3, 2, 2), 283.15, dtype=np.float32)),
                "hcc": (("step", "latitude", "longitude"), np.full((3, 2, 2), 1.0001, dtype=np.float32)),
                "ssrd": (("step", "latitude", "longitude"), ssrd),
            },
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": [np.timedelta64(83, "h"), np.timedelta64(84, "h"), np.timedelta64(85, "h")],
                "latitude": [52.0, 51.0],
                "longitude": [4.0, 5.0],
            },
        )

        with patch("cfgrib.open_datasets", return_value=[ds]):
            out_ds = process_ecmwf_live(
                grib_path="dummy.grib", bbox_nwse=[53, 3, 50, 6], max_step_hours=84
            )

        self.assertIn(np.timedelta64(84, "h"), out_ds.step.values)
        self.assertNotIn(np.timedelta64(85, "h"), out_ds.step.values)
        np.testing.assert_allclose(out_ds["high_cloud_cover"].values, 100.0, atol=1e-4)
        np.testing.assert_allclose(
            out_ds["downward_short_wave_radiation_flux_surface"].values, 1.0, atol=1e-4
        )

    def test_cli_fetch_rejects_incomplete_s3_run(self):
        """Test that standalone CLI refuses to download when S3 dissemination is incomplete."""
        t = dt.datetime(2026, 4, 8, 0, 0, tzinfo=dt.UTC)
        with (
            patch.dict(os.environ, {"ECMWF_REALTIME_S3_BUCKET": "test-bucket"}, clear=False),
            patch("ecmwflive_data.cli.get_completed_init_times", return_value={}),
            patch("ecmwflive_data.cli.download_live_ecmwf_partition") as mock_download,
        ):
            with self.assertRaises(RuntimeError):
                PRODUCTS["hres_west_europe"].fetch(t)
            mock_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
