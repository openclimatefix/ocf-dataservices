import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from metoffice_data.processing import (
    _UKV_X_LAEA,
    _UKV_Y_LAEA,
    _assign_ukv_coords,
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


if __name__ == "__main__":
    unittest.main()
