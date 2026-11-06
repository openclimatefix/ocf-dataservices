import unittest

import numpy as np
import xarray as xr
from ceda_data.schema import CedaGlobalWestEuropeSchema, CedaUkvSchema
from pandera.errors import SchemaError


class TestCedaSchemas(unittest.TestCase):
    def _make_ukv_dataset(self, max_step_hours: int = 54) -> xr.Dataset:
        dims = ("init_time", "step", "y_osgb", "x_osgb")
        var_names = [
            "total_cloud_cover_atmosphere",
            "high_cloud_cover",
            "medium_cloud_cover",
            "low_cloud_cover",
            "visibility",
            "relative_humidity_2m",
            "total_precipitation_rate",
            "snow_depth",
            "downward_long_wave_radiation_flux_surface",
            "downward_short_wave_radiation_flux_surface",
            "temperature_2m",
            "wind_u_10m",
            "wind_v_10m",
        ]
        return xr.Dataset(
            data_vars={
                name: (
                    dims,
                    np.full(
                        (1, 2, 2, 2),
                        50.0 if "humidity" in name or "cloud" in name else 0.0,
                        dtype=np.float32,
                    ),
                )
                for name in var_names
            },
            coords={
                "init_time": [np.datetime64("2024-01-01T00:00")],
                "step": [np.timedelta64(0, "h"), np.timedelta64(max_step_hours, "h")],
                "y_osgb": [1223000.0, 1221000.0],
                "x_osgb": [-239000.0, -237000.0],
                "generating_process_id": ("init_time", np.array([45], dtype=np.int16)),
                "grib_tables_version": ("init_time", np.array([19], dtype=np.int16)),
                "grid_first_lat": ("init_time", np.array([48.0], dtype=np.float64)),
                "grid_first_lon": ("init_time", np.array([-12.0], dtype=np.float64)),
                "ingested_at": ("init_time", [np.datetime64("2024-01-01T01:00")]),
            },
        )

    def _make_global_dataset(self, max_step_hours: int = 144) -> xr.Dataset:
        dims = ("init_time", "step", "latitude", "longitude")
        var_names = [
            "total_cloud_cover_atmosphere",
            "high_cloud_cover",
            "medium_cloud_cover",
            "low_cloud_cover",
            "visibility",
            "relative_humidity_2m",
            "pressure_reduced_to_mean_sea_level",
            "snow_depth",
            "downward_short_wave_radiation_flux_surface",
            "temperature_2m",
            "wind_u_10m",
            "wind_v_10m",
        ]
        return xr.Dataset(
            data_vars={
                name: (
                    dims,
                    np.full(
                        (1, 2, 2, 2),
                        101325.0
                        if "pressure" in name
                        else (50.0 if "humidity" in name or "cloud" in name else 10.0),
                        dtype=np.float32,
                    ),
                )
                for name in var_names
            },
            coords={
                "init_time": [np.datetime64("2024-01-01T00:00")],
                "step": [np.timedelta64(0, "h"), np.timedelta64(max_step_hours, "h")],
                "latitude": [55.0, 50.0],
                "longitude": [-5.0, 5.0],
                "generating_process_id": ("init_time", np.array([45], dtype=np.int16)),
                "grib_tables_version": ("init_time", np.array([19], dtype=np.int16)),
                "ingested_at": ("init_time", [np.datetime64("2024-01-01T01:00")]),
            },
        )

    def test_ukv_schema_passes_and_rejects_step_above_54(self) -> None:
        CedaUkvSchema.validate(self._make_ukv_dataset(max_step_hours=54))
        with self.assertRaises(SchemaError):
            CedaUkvSchema.validate(self._make_ukv_dataset(max_step_hours=55))

    def test_global_schema_passes_and_rejects_step_above_144(self) -> None:
        CedaGlobalWestEuropeSchema.validate(self._make_global_dataset(max_step_hours=144))
        with self.assertRaises(SchemaError):
            CedaGlobalWestEuropeSchema.validate(self._make_global_dataset(max_step_hours=145))


if __name__ == "__main__":
    unittest.main()
