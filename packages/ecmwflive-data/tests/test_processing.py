import datetime as dt
import os
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from ecmwflive_data.cli import PRODUCTS
from ecmwflive_data.download import download_live_ecmwf_partition, get_completed_init_times
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
            self.assertIn("generating_process_id", out_ds.coords)
            self.assertIn("expver", out_ds.coords)
            self.assertIn("grib_tables_version", out_ds.coords)
            self.assertIn("ingested_at", out_ds.coords)

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

        ds["hcc"].attrs = {
            "GRIB_generatingProcessIdentifier": 155,
            "GRIB_expver": "h9be",
            "GRIB_tablesVersion": 32,
        }

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
        self.assertEqual(list(out_ds.coords["generating_process_id"].values), [155])
        self.assertEqual(list(out_ds.coords["expver"].values), [-1])
        self.assertEqual(list(out_ds.coords["grib_tables_version"].values), [32])

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

    def test_get_completed_init_times_december_to_january_rollover(self):
        """A Dec 29 00Z run whose +168h target time lands on Jan 05 00Z of the next year completes."""
        with patch("s3fs.S3FileSystem") as mock_fs_cls:
            mock_fs = mock_fs_cls.return_value
            mock_fs.ls.return_value = [
                "bucket/ecmwf/A2D12290000122900001",
                "bucket/ecmwf/A2D12290000010500001",
            ]
            now = dt.datetime.now(dt.UTC)
            completed = get_completed_init_times(
                bucket="bucket",
                access_key="k",
                secret="s",
                endpoint_url=None,
                region_name="eu-west-1",
                prefix="A2",
                last_processed=dt.datetime(2000, 1, 1, tzinfo=dt.UTC),
            )
            expected_year = now.year - 1 if now.month == 1 else now.year
            expected_it = dt.datetime(expected_year, 12, 29, 0, 0, tzinfo=dt.UTC)
            expected_tt = dt.datetime(expected_year + 1, 1, 5, 0, 0, tzinfo=dt.UTC)
            self.assertIn(expected_it, completed)
            self.assertEqual(completed[expected_it], expected_tt)

    def test_mixed_0_360_global_and_regional_and_coarse_grids_align(self):
        """Global 0..360 1-deg and regional -10..10 1-deg merge cleanly; coarse 2-deg grid is ignored."""
        steps = [np.timedelta64(0, "h"), np.timedelta64(1, "h")]
        # Global-style 0..360 longitudes covering [-2..2] via [358, 359, 0, 1, 2]
        ds_global_fine = xr.Dataset(
            data_vars={"t2m": (("step", "latitude", "longitude"), np.full((2, 3, 5), 283.15, dtype=np.float32))},
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": steps,
                "latitude": [50.0, 51.0, 52.0],
                "longitude": [0.0, 1.0, 2.0, 358.0, 359.0],
            },
        )
        # Regional subset already in -180..180 on the same 1-deg spacing
        ds_reg_fine = xr.Dataset(
            data_vars={"10u": (("step", "latitude", "longitude"), np.full((2, 3, 5), 4.0, dtype=np.float32))},
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": steps,
                "latitude": [52.0, 51.0, 50.0],
                "longitude": [-2.0, -1.0, 0.0, 1.0, 2.0],
            },
        )
        # Coarser 2-deg grid that also overlaps bbox
        ds_coarse = xr.Dataset(
            data_vars={"vis": (("step", "latitude", "longitude"), np.full((2, 2, 3), 9000.0, dtype=np.float32))},
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": steps,
                "latitude": [52.0, 50.0],
                "longitude": [-2.0, 0.0, 2.0],
            },
        )

        with patch("cfgrib.open_datasets", return_value=[ds_global_fine, ds_reg_fine, ds_coarse]):
            out = process_ecmwf_live("dummy.grib", bbox_nwse=[52, -2, 50, 2], max_step_hours=84)

        np.testing.assert_allclose(out.longitude.values, [-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(out.latitude.values, [52.0, 51.0, 50.0])
        self.assertFalse(bool(out["temperature_2m"].isnull().any()))
        self.assertFalse(bool(out["wind_u_10m"].isnull().any()))
        self.assertNotIn("visibility", out.data_vars)

    def test_download_live_ecmwf_partition_unlinks_partial_file_on_error(self):
        import io
        import tempfile
        from pathlib import Path

        valid_chunk = b"GRIB" + b"\x00" * 110_000 + b"7777"
        with tempfile.TemporaryDirectory() as tmp, patch("s3fs.S3FileSystem") as mock_fs_cls:
            mock_fs = mock_fs_cls.return_value
            mock_fs.ls.return_value = [
                "bucket/ecmwf/A2S04080000040800001",
                "bucket/ecmwf/A2S04080000040801001",
            ]
            mock_fs.open.side_effect = [io.BytesIO(valid_chunk), OSError("S3 stream dropped")]
            out_path = Path(tmp) / "live.grib"
            with self.assertRaises(OSError):
                download_live_ecmwf_partition(
                    partition_time=dt.datetime(2026, 4, 8, 0, 0, tzinfo=dt.UTC),
                    out_path=out_path,
                    bucket="bucket",
                    access_key="k",
                    secret="s",
                    endpoint_url=None,
                    region_name="eu-west-1",
                    prefix="A2",
                )
            self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
