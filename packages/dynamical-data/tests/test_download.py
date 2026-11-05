import datetime as dt
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from dynamical_data.ecmwf_ens.download import download, open_it
from dynamical_data.ecmwf_ens.schema import DynamicalEcmwfEnsSchema


def _mock_catalog_dataset() -> xr.Dataset:
    dims = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
    shape = (1, 4, 3, 2, 2)
    raw_vars = [
        "temperature_2m",
        "wind_u_10m",
        "wind_v_10m",
        "wind_u_100m",
        "wind_v_100m",
        "total_cloud_cover_atmosphere",
        "pressure_reduced_to_mean_sea_level",
        "downward_long_wave_radiation_flux_surface",
        "downward_short_wave_radiation_flux_surface",
        "precipitation_surface",
        "categorical_precipitation_type_surface",
    ]
    data_vars = {}
    for var in raw_vars:
        if var == "pressure_reduced_to_mean_sea_level":
            val = 101325.0
        elif var == "total_cloud_cover_atmosphere":
            val = 50.0
        elif var == "precipitation_surface":
            val = 0.001
        else:
            val = 10.0
        arr = np.full(shape, val, dtype=np.float32)
        if var in (
            "downward_long_wave_radiation_flux_surface",
            "downward_short_wave_radiation_flux_surface",
            "precipitation_surface",
        ):
            # Dynamical has NaN at lead_time=0h because its rates are backward averages
            arr[:, 0, ...] = np.nan
        data_vars[var] = (dims, arr)

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "init_time": np.array(["2024-04-01T00:00"], dtype="datetime64[ns]"),
            "lead_time": np.array(
                [
                    np.timedelta64(0, "h"),
                    np.timedelta64(1, "h"),
                    np.timedelta64(85, "h"),
                    np.timedelta64(120, "h"),
                ]
            ),
            # Include control member 0 and int64 dtype to verify slicing + int16 cast
            "ensemble_member": np.array([0, 1, 2], dtype=np.int64),
            "latitude": np.array([55.0, 50.0], dtype=np.float64),
            "longitude": np.array([-5.0, 0.0], dtype=np.float64),
            "expected_forecast_length": np.timedelta64(360, "h"),
        },
    )


class TestDynamicalDownload(unittest.TestCase):
    def test_open_and_download_validates(self):
        with patch("dynamical_catalog.open", return_value=_mock_catalog_dataset()):
            ds_lazy = open_it(
                nwp_init_time=dt.datetime(2024, 4, 1, 0, 0, tzinfo=dt.UTC),
                bbox_nwse=(60, -10, 48, 2),
            )
            ds = download(ds_lazy)

        DynamicalEcmwfEnsSchema.validate(ds)
        self.assertEqual(list(ds.dims), list(DynamicalEcmwfEnsSchema.dims()))
        self.assertIn("total_precipitation_rate", ds.data_vars)
        self.assertNotIn("precipitation_surface", ds.data_vars)
        self.assertEqual(ds.ensemble_member.dtype, np.int16)
        self.assertEqual(list(ds.ensemble_member.values), [1, 2])
        # 120h sliced out in open_it; 85h dropped after forward-aligning de-accumulated vars
        self.assertEqual(
            list(ds.step.values), [np.timedelta64(0, "h"), np.timedelta64(1, "h")]
        )
        self.assertFalse(bool(ds["downward_short_wave_radiation_flux_surface"].isnull().any()))
        self.assertFalse(bool(ds["total_precipitation_rate"].isnull().any()))


if __name__ == "__main__":
    unittest.main()
