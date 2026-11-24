import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from metoffice_data.client import DatahubClient
from metoffice_data.download import download_order_partition
from metoffice_data.processing import (
    _UKV_X_LAEA,
    _UKV_Y_LAEA,
    _assign_ukv_coords,
    _read_raw_grib,
    _transform,
    process_metoffice_india,
)
from metoffice_data.schema import MetOfficeGlobalIndiaSchema


def _global_raw() -> xr.Dataset:
    """A small synthetic raw-ish global dataset with MetOffice/cfgrib-style names and units."""
    shape = ("step", "latitude", "longitude")
    return xr.Dataset(
        data_vars={
            "tcc": (shape, np.full((2, 2, 2), 50.0, dtype=np.float32)),
            "hcc": (shape, np.full((2, 2, 2), 20.0, dtype=np.float32)),
            "mcc": (shape, np.full((2, 2, 2), 30.0, dtype=np.float32)),
            "lcc": (shape, np.full((2, 2, 2), 40.0, dtype=np.float32)),
            "vis": (shape, np.full((2, 2, 2), 10000.0, dtype=np.float32)),
            "r": (shape, np.full((2, 2, 2), 80.0, dtype=np.float32)),
            "sd": (shape, np.zeros((2, 2, 2), dtype=np.float32)),
            "dswrf": (shape, np.full((2, 2, 2), 100.0, dtype=np.float32)),
            "t2m": (shape, np.full((2, 2, 2), 290.0, dtype=np.float32)),
            "u10": (shape, np.full((2, 2, 2), 3.0, dtype=np.float32)),
            "v10": (shape, np.full((2, 2, 2), -2.0, dtype=np.float32)),
        },
        coords={
            "time": np.datetime64("2024-01-01T00:00"),
            "step": [np.timedelta64(0, "h"), np.timedelta64(1, "h")],
            "latitude": [20.0, 21.0],
            "longitude": [70.0, 71.0],
        },
        attrs={
            "GRIB_generatingProcessIdentifier": 128,
            "GRIB_localTablesVersion": 1,
        },
    )


class TestProcessing(unittest.TestCase):
    def test_transform_units_and_dims(self):
        out = _transform(_global_raw(), dims=MetOfficeGlobalIndiaSchema.dims())

        self.assertEqual(list(out.dims), ["init_time", "step", "latitude", "longitude"])
        self.assertAlmostEqual(float(out["total_cloud_cover_atmosphere"].max()), 50.0, places=3)
        self.assertAlmostEqual(float(out["temperature_2m"].max()), 290.0 - 273.15, places=2)
        # Renamed to canonical names
        self.assertIn("wind_u_10m", out.data_vars)
        self.assertIn("relative_humidity_2m", out.data_vars)
        self.assertEqual(list(out.coords["generating_process_id"].values), [128])
        self.assertEqual(list(out.coords["local_tables_version"].values), [1])
        self.assertIn("ingested_at", out.coords)

    def test_validated_entrypoint(self):
        with patch("metoffice_data.processing._read_raw_grib", return_value=_global_raw()):
            out = process_metoffice_india("dummy.grib")
        MetOfficeGlobalIndiaSchema.validate(out)

    def test_assign_ukv_coords(self):
        ny, nx = len(_UKV_Y_LAEA), len(_UKV_X_LAEA)
        ds = xr.Dataset(
            data_vars={"t2m": (("y", "x"), np.zeros((ny, nx), dtype=np.float32))},
            coords={"latitude": (("y", "x"), np.zeros((ny, nx)))},
        )
        out = _assign_ukv_coords(ds)

        self.assertEqual(dict(out.sizes), {"y_laea": ny, "x_laea": nx})
        self.assertEqual(float(out["y_laea"][0]), 700000.0)
        self.assertEqual(float(out["y_laea"][-1]), -576000.0)
        self.assertEqual(float(out["x_laea"][0]), -576000.0)
        self.assertEqual(float(out["x_laea"][-1]), 332000.0)
        self.assertNotIn("latitude", out.coords)

    def test_wind_speed_direction_converted_to_uv(self):
        shape = ("step", "latitude", "longitude")
        ds = xr.Dataset(
            data_vars={
                "si10": (shape, np.full((2, 2, 2), 10.0, dtype=np.float32)),
                # 270 deg = wind from West = blowing East (+u, 0 v)
                "wdir": (shape, np.full((2, 2, 2), 270.0, dtype=np.float32)),
            },
            coords={
                "time": np.datetime64("2024-01-01T00:00"),
                "step": [np.timedelta64(0, "h"), np.timedelta64(1, "h")],
                "latitude": [20.0, 21.0],
                "longitude": [70.0, 71.0],
            },
        )
        out = _transform(ds, dims=MetOfficeGlobalIndiaSchema.dims())
        self.assertIn("wind_u_10m", out.data_vars)
        self.assertIn("wind_v_10m", out.data_vars)
        self.assertNotIn("wind_speed_10m", out.data_vars)
        self.assertNotIn("wind_direction_10m", out.data_vars)
        np.testing.assert_allclose(out["wind_u_10m"].values, 10.0, atol=1e-5)
        np.testing.assert_allclose(out["wind_v_10m"].values, 0.0, atol=1e-5)

    def test_read_raw_grib_keeps_tcc_and_rejects_mismatched_steps(self):
        import tempfile

        import eccodes

        with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
            gid_rh = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            eccodes.codes_set(gid_rh, "discipline", 0)
            eccodes.codes_set(gid_rh, "parameterCategory", 1)
            eccodes.codes_set(gid_rh, "parameterNumber", 1)
            eccodes.codes_set(gid_rh, "typeOfFirstFixedSurface", 103)
            eccodes.codes_set(gid_rh, "scaledValueOfFirstFixedSurface", 2)
            eccodes.codes_set(gid_rh, "scaleFactorOfFirstFixedSurface", 0)
            eccodes.codes_write(gid_rh, f)
            eccodes.codes_release(gid_rh)

            gid_tcc = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            eccodes.codes_set(gid_tcc, "centre", 74)
            eccodes.codes_set(gid_tcc, "discipline", 0)
            eccodes.codes_set(gid_tcc, "parameterCategory", 6)
            eccodes.codes_set(gid_tcc, "parameterNumber", 1)
            eccodes.codes_set(gid_tcc, "typeOfFirstFixedSurface", 1)
            eccodes.codes_write(gid_tcc, f)
            eccodes.codes_release(gid_tcc)
            f.flush()

            ds = _read_raw_grib(f.name)
            self.assertIn("tcc", ds.data_vars)
            self.assertTrue({"r", "r2"} & set(ds.data_vars))

        # Add a second step only for rh so step coordinates disagree -> join='exact' rejects
        with tempfile.NamedTemporaryFile(suffix=".grib2") as f_bad:
            for step_hr in (0, 1):
                gid_rh = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
                eccodes.codes_set(gid_rh, "discipline", 0)
                eccodes.codes_set(gid_rh, "parameterCategory", 1)
                eccodes.codes_set(gid_rh, "parameterNumber", 1)
                eccodes.codes_set(gid_rh, "stepRange", str(step_hr))
                eccodes.codes_write(gid_rh, f_bad)
                eccodes.codes_release(gid_rh)

            gid_tcc = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            eccodes.codes_set(gid_tcc, "centre", 74)
            eccodes.codes_set(gid_tcc, "discipline", 0)
            eccodes.codes_set(gid_tcc, "parameterCategory", 6)
            eccodes.codes_set(gid_tcc, "parameterNumber", 1)
            eccodes.codes_set(gid_tcc, "stepRange", "0")
            eccodes.codes_write(gid_tcc, f_bad)
            eccodes.codes_release(gid_tcc)
            f_bad.flush()

            with self.assertRaises(ValueError):
                _read_raw_grib(f_bad.name)

    def test_download_order_partition_skips_incomplete_and_keeps_percent_wisp(self):
        import datetime as dt
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock

        client = MagicMock(spec=DatahubClient)
        client.list_file_ids.return_value = ["f1", "f2"]
        with tempfile.TemporaryDirectory() as tmp:
            count = download_order_partition(
                client=client,
                order_id="india-11params-54steps",
                init_time=dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.UTC),
                out_path=Path(tmp) / "partial.grib",
                min_files=11 * 54,
            )
            self.assertEqual(count, 0)
            client.download_file.assert_not_called()

        raw = _global_raw()
        raw["hcc"] = xr.DataArray(
            np.full((2, 2, 2), 0.8, dtype=np.float32),
            dims=("step", "latitude", "longitude"),
            attrs={"GRIB_units": "%"},
        )
        out = _transform(raw, dims=MetOfficeGlobalIndiaSchema.dims())
        np.testing.assert_allclose(out["high_cloud_cover"].values, 0.8, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
